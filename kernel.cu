// KanQ — Quantus (QTC) GPU miner, Linux/Ubuntu build (g++/nvcc).
// Async N-stream pipeline; no drain on job change; honest hash counter.
// Colored console output with GPU monitoring (NVML) and live status line.

#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <random>
#include <stdexcept>
#include <stdarg.h>

#include "version.h"
#include "pow2.cuh"
#include "qpow_host.h"

using namespace qpow;

// ===========================================================================
// ANSI colors + Linux terminal setup
// ===========================================================================

#define C_RESET       "\033[0m"
#define C_BOLD        "\033[1m"
#define C_DIM         "\033[2m"
#define C_RED         "\033[31m"
#define C_GREEN       "\033[32m"
#define C_YELLOW      "\033[33m"
#define C_BLUE        "\033[34m"
#define C_MAGENTA     "\033[35m"
#define C_CYAN        "\033[36m"
#define C_WHITE       "\033[37m"
#define C_BOLD_RED    "\033[1;31m"
#define C_BOLD_GREEN  "\033[1;32m"
#define C_BOLD_YELLOW "\033[1;33m"
#define C_BOLD_CYAN   "\033[1;36m"

static bool g_vt = false;

static bool enable_vt() {
    // On Linux the console driver interprets ANSI/VT sequences by default
    // in every common terminal (xterm, tmux, gnome-terminal, ssh pts).
    // Only disable when stdout is not a TTY (e.g. redirected to a file).
    return isatty(STDOUT_FILENO) != 0;
}

// Print a scrolling event line — clears status, prints, adds newline.
static void print_event(const char* fmt, ...) {
    if (g_vt) printf("\r\033[K"); else printf("\r");
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
    fflush(stdout);
}

// Print/refresh the status line — no newline, overwrites in place.
static void print_status(const char* fmt, ...) {
    if (g_vt) printf("\r\033[K"); else printf("\r");
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    fflush(stdout);
}

// ===========================================================================
// NVML — GPU monitoring via dynamic loading (libnvidia-ml.so.1)
// ===========================================================================

struct NvmlUtilization {
    unsigned int gpu;
    unsigned int memory;
};

static void* g_nvml_dll = nullptr;
static void* g_nvml_dev = nullptr;

static int (*pfn_Init)() = nullptr;
static int (*pfn_GetHandle)(unsigned int, void**) = nullptr;
static int (*pfn_GetTemp)(void*, int, unsigned int*) = nullptr;
static int (*pfn_GetClock)(void*, unsigned int, unsigned int*) = nullptr;
static int (*pfn_GetPower)(void*, unsigned int*) = nullptr;
static int (*pfn_GetPowerLimit)(void*, unsigned int*) = nullptr;
static int (*pfn_GetFan)(void*, unsigned int*) = nullptr;
static int (*pfn_GetUtil)(void*, NvmlUtilization*) = nullptr;
static int (*pfn_GetName)(void*, char*, unsigned int) = nullptr;

struct GpuStats {
    bool         available = false;
    unsigned int temp = 0;            // °C
    unsigned int clock_graphics = 0;  // MHz
    unsigned int clock_mem = 0;       // MHz
    unsigned int power_mw = 0;        // mW
    unsigned int power_limit_mw = 0;  // mW
    unsigned int fan_pct = 0;         // %
    unsigned int util_gpu = 0;        // %
    unsigned int util_mem = 0;        // %
    char         name[96] = {};
};

static bool nvml_init(int device_idx) {
    g_nvml_dll = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!g_nvml_dll) g_nvml_dll = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!g_nvml_dll) return false;

    pfn_Init = (int(*)())dlsym(g_nvml_dll, "nvmlInit_v2");
    if (!pfn_Init) pfn_Init = (int(*)())dlsym(g_nvml_dll, "nvmlInit");
    if (!pfn_Init) return false;

    pfn_GetHandle = (int(*)(unsigned int, void**))
        dlsym(g_nvml_dll, "nvmlDeviceGetHandleByIndex_v2");
    if (!pfn_GetHandle)
        pfn_GetHandle = (int(*)(unsigned int, void**))
        dlsym(g_nvml_dll, "nvmlDeviceGetHandleByIndex");

    pfn_GetTemp = (int(*)(void*, int, unsigned int*))
        dlsym(g_nvml_dll, "nvmlDeviceGetTemperature");
    pfn_GetClock = (int(*)(void*, unsigned int, unsigned int*))
        dlsym(g_nvml_dll, "nvmlDeviceGetClockInfo");
    pfn_GetPower = (int(*)(void*, unsigned int*))
        dlsym(g_nvml_dll, "nvmlDeviceGetPowerUsage");
    pfn_GetPowerLimit = (int(*)(void*, unsigned int*))
        dlsym(g_nvml_dll, "nvmlDeviceGetPowerManagementLimit");
    pfn_GetFan = (int(*)(void*, unsigned int*))
        dlsym(g_nvml_dll, "nvmlDeviceGetFanSpeed");
    pfn_GetUtil = (int(*)(void*, NvmlUtilization*))
        dlsym(g_nvml_dll, "nvmlDeviceGetUtilizationRates");
    pfn_GetName = (int(*)(void*, char*, unsigned int))
        dlsym(g_nvml_dll, "nvmlDeviceGetName");

    if (!pfn_Init || pfn_Init() != 0) return false;
    if (!pfn_GetHandle) return false;
    if (pfn_GetHandle((unsigned int)device_idx, &g_nvml_dev) != 0) return false;
    return true;
}

static GpuStats gpu_query() {
    GpuStats s;
    if (!g_nvml_dev) return s;
    s.available = true;
    unsigned int v;
    if (pfn_GetTemp && pfn_GetTemp(g_nvml_dev, 0, &v) == 0)          s.temp = v;
    if (pfn_GetClock) {
        if (pfn_GetClock(g_nvml_dev, 0, &v) == 0) s.clock_graphics = v;   // NVML_CLOCK_GRAPHICS
        if (pfn_GetClock(g_nvml_dev, 2, &v) == 0) s.clock_mem = v;        // NVML_CLOCK_MEM
    }
    if (pfn_GetPower && pfn_GetPower(g_nvml_dev, &v) == 0)            s.power_mw = v;
    if (pfn_GetPowerLimit && pfn_GetPowerLimit(g_nvml_dev, &v) == 0)  s.power_limit_mw = v;
    if (pfn_GetFan && pfn_GetFan(g_nvml_dev, &v) == 0)                s.fan_pct = v;
    if (pfn_GetUtil) {
        NvmlUtilization u;
        if (pfn_GetUtil(g_nvml_dev, &u) == 0) {
            s.util_gpu = u.gpu;
            s.util_mem = u.memory;
        }
    }
    if (pfn_GetName) pfn_GetName(g_nvml_dev, s.name, sizeof(s.name));
    return s;
}

static const char* temp_color(unsigned int t) {
    if (t >= 80) return C_BOLD_RED;
    if (t >= 70) return C_YELLOW;
    return C_GREEN;
}

// ===========================================================================
// Small helpers
// ===========================================================================

static bool hex_to_bytes(const std::string& s, uint8_t* out, size_t n) {
    if (s.size() != n * 2) return false;
    for (size_t i = 0; i < n; i++) {
        auto hv = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
            };
        int hi = hv(s[i * 2]), lo = hv(s[i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

static std::string bytes_to_hex(const uint8_t* b, size_t n) {
    static const char* H = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (size_t i = 0; i < n; i++) { s.push_back(H[b[i] >> 4]); s.push_back(H[b[i] & 0xf]); }
    return s;
}

static bool json_str(const std::string& s, const std::string& key, std::string& out) {
    std::string pat = "\"" + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    p++;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) p++;
    if (p >= s.size()) return false;
    if (s[p] == '"') {
        size_t e = s.find('"', p + 1);
        if (e == std::string::npos) return false;
        out = s.substr(p + 1, e - p - 1);
        return true;
    }
    size_t e = p;
    while (e < s.size() && s[e] != ',' && s[e] != '}' && s[e] != ']' && s[e] != ' ') e++;
    out = s.substr(p, e - p);
    return !out.empty();
}

static uint64_t parse_u64(const std::string& dec) {
    uint64_t v = 0;
    for (char c : dec) if (c >= '0' && c <= '9') v = v * 10 + (uint64_t)(c - '0');
    return v;
}

#define CUDA_CHECK(x)                                                                \
    do {                                                                              \
        cudaError_t e_ = (x);                                                         \
        if (e_ != cudaSuccess)                                                        \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + \
                                     " @" __FILE__ ":" + std::to_string(__LINE__));   \
    } while (0)

// ===========================================================================
// Pool (POSIX sockets)
// ===========================================================================

struct Job {
    std::string job_id, mining_hash, difficulty, extranonce;
    bool valid = false;
};

class Pool {
public:
    ~Pool() { close_fd(); }

    bool connect_to(const std::string& hostport) {
        close_fd();
        auto pos = hostport.rfind(':');
        if (pos == std::string::npos) {
            print_event("%s[%sBAD ADDR%s]%s %s",
                C_BOLD, C_RED, C_RESET, C_DIM, hostport.c_str());
            return false;
        }
        std::string host = hostport.substr(0, pos);
        std::string port = hostport.substr(pos + 1);

        struct addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        int gerr = getaddrinfo(host.c_str(), port.c_str(), &hints, &res);
        if (gerr != 0) {
            print_event("%s[%sDNS FAIL%s]%s %s (%s)",
                C_BOLD, C_RED, C_RESET, C_DIM, host.c_str(), gai_strerror(gerr));
            return false;
        }
        for (auto* p = res; p; p = p->ai_next) {
            int fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
            if (fd < 0) continue;
            if (::connect(fd, p->ai_addr, (socklen_t)p->ai_addrlen) == 0) { fd_ = fd; break; }
            ::close(fd);
        }
        freeaddrinfo(res);
        if (fd_ < 0) {
            print_event("%s[%sCONNECT FAIL%s]%s %s",
                C_BOLD, C_RED, C_RESET, C_DIM, hostport.c_str());
            return false;
        }
        int one = 1;
        setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        int flags = fcntl(fd_, F_GETFL, 0);
        if (flags >= 0) fcntl(fd_, F_SETFL, flags | O_NONBLOCK);
        print_event("%s[%sCONNECT%s]%s %s%s%s",
            C_BOLD, C_GREEN, C_RESET, C_DIM,
            C_WHITE, hostport.c_str(), C_RESET);
        return true;
    }

    void login(const std::string& wallet, const std::string& worker) {
        std::string user = worker.empty() ? wallet : wallet + "." + worker;
        send("{\"id\":" + std::to_string(++id_) +
            ",\"method\":\"login\",\"params\":{\"agent\":\"" CCMINER_AGENT "\",\"login\":\"" +
            user + "\",\"pass\":\"x\"}}");
        print_event("%s[%sLOGIN%s]%s agent=%s user=%s",
            C_BOLD, C_BLUE, C_RESET, C_DIM,
            CCMINER_AGENT, user.c_str());
    }

    void submit(const std::string& job_id, const std::string& nonce_hex, const std::string& result_hex) {
        send("{\"id\":" + std::to_string(++id_) + ",\"method\":\"submit\",\"params\":{" +
            std::string("\"id\":\"") + session_ + "\"," +
            "\"job_id\":\"" + job_id + "\"," +
            "\"nonce\":\"" + nonce_hex + "\"," +
            "\"result\":\"" + result_hex + "\"}}");
        submitted_++;
    }

    void pump() {
        char buf[65536];
        for (;;) {
            ssize_t n = ::recv(fd_, buf, sizeof(buf), 0);
            if (n > 0) {
                inbox_.append(buf, (size_t)n);
                if ((size_t)n < sizeof(buf)) break;
            }
            else if (n == 0) {
                throw std::runtime_error("Pool closed the connection");
            }
            else {
                if (errno == EAGAIN || errno == EWOULDBLOCK ||
                    errno == EINTR || errno == EINPROGRESS) break;
                throw std::runtime_error(std::string("recv failed: ") + strerror(errno));
            }
        }
        size_t p;
        while ((p = inbox_.find('\n')) != std::string::npos) {
            std::string line = inbox_.substr(0, p);
            inbox_.erase(0, p + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) handle(line);
        }
    }

    Job job() { std::lock_guard<std::mutex> lk(m_); return job_; }
    uint64_t job_seq() { std::lock_guard<std::mutex> lk(m_); return job_seq_; }
    int accepted() const { return accepted_; }
    int rejected() const { return rejected_; }
    int submitted() const { return submitted_; }
    const std::string& last_error() const { return last_error_; }

private:
    void close_fd() {
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    }

    void send(const std::string& s) {
        std::string out = s + "\n";
        ssize_t n = ::send(fd_, out.data(), out.size(), MSG_NOSIGNAL);
        if (n < 0) throw std::runtime_error(std::string("send failed: ") + strerror(errno));
    }

    void handle(const std::string& line) {
        if (line.find("\"method\":\"job\"") != std::string::npos) {
            Job j;
            json_str(line, "job_id", j.job_id);
            json_str(line, "mining_hash", j.mining_hash);
            json_str(line, "difficulty", j.difficulty);
            json_str(line, "extranonce", j.extranonce);
            j.valid = !j.job_id.empty() && !j.mining_hash.empty() && !j.difficulty.empty();
            if (j.valid) {
                std::lock_guard<std::mutex> lk(m_);
                job_ = j;
                job_seq_++;
                print_event("%s[%sJOB%s]%s id=%-20s diff=%-10s extr=%s",
                    C_BOLD, C_CYAN, C_RESET, C_DIM,
                    j.job_id.c_str(), j.difficulty.c_str(),
                    j.extranonce.substr(0, 16).c_str());
            }
            return;
        }
        if (line.find("\"job\"") != std::string::npos && line.find("\"result\"") != std::string::npos) {
            Job j;
            json_str(line, "job_id", j.job_id);
            json_str(line, "mining_hash", j.mining_hash);
            json_str(line, "difficulty", j.difficulty);
            json_str(line, "extranonce", j.extranonce);
            j.valid = !j.job_id.empty() && !j.mining_hash.empty();
            std::string sid;
            size_t q = line.find("\"result\"");
            if (q != std::string::npos) json_str(line.substr(q), "id", sid);
            if (j.valid) {
                std::lock_guard<std::mutex> lk(m_);
                session_ = sid;
                job_ = j;
                job_seq_++;
                print_event("%s[%sLOGIN OK%s]%s session=%s job=%s diff=%s",
                    C_BOLD, C_GREEN, C_RESET, C_DIM,
                    sid.c_str(), j.job_id.c_str(), j.difficulty.c_str());
            }
            return;
        }
        std::string msg;
        if (line.find("\"error\":{") != std::string::npos && json_str(line, "message", msg)) {
            rejected_++;
            last_error_ = msg;
            print_event("%s[%sREJECTED%s]%s %s%s%s",
                C_BOLD_RED, C_RESET, C_RESET, C_DIM,
                C_RED, msg.c_str(), C_RESET);
            return;
        }
        if (line.find("\"status\":\"OK\"") != std::string::npos ||
            line.find("\"result\":true") != std::string::npos) {
            accepted_++;
            print_event("%s[%sACCEPTED%s]%s total=%d",
                C_BOLD_GREEN, C_RESET, C_RESET, C_DIM, accepted_);
            return;
        }
        print_event("%s[%sPOOL%s]%s unrecognized: %s",
            C_BOLD, C_YELLOW, C_RESET, C_DIM,
            line.substr(0, 200).c_str());
    }

    int fd_ = -1;
    int id_ = 0;
    int accepted_ = 0, rejected_ = 0, submitted_ = 0;
    std::string inbox_, last_error_, session_;
    Job job_;
    uint64_t job_seq_ = 0;
    std::mutex m_;
};

// ===========================================================================
// CLI
// ===========================================================================

static void usage() {
    printf(
        "%sKanQ %s%s -- Quantus (QTC) GPU miner for Kryptex\n\n"
        "%sUsage:%s\n"
        "  kanq --wallet <addr> [--pool <host:port>] [--worker <name>] [--device <n>]\n"
        "        [--iters <n>] [--blocks <n>] [--streams <n>] [--once] [--dry-run]\n\n"
        "  %s--pool%s     default qtc.kryptex.network:7049\n"
        "  %s--wallet%s   QTC wallet (qz...)\n"
        "  %s--worker%s   worker name (login as wallet.worker, default 'gpu')\n"
        "  %s--device%s   GPU index (default 0; one process per card)\n"
        "  %s--blocks%s   grid blocks (default 4096, 256 threads/block)\n"
        "  %s--iters%s    nonces per thread (default 64)\n"
        "  %s--streams%s  async pipeline depth (default 3)\n"
        "  %s--once%s     exit after first accepted share\n"
        "  %s--dry-run%s  search and verify, don't submit\n"
        "  %s--version%s  show version\n",
        C_BOLD, CCMINER_VERSION, C_RESET,
        C_BOLD_CYAN, C_RESET,
        C_YELLOW, C_RESET, C_YELLOW, C_RESET, C_YELLOW, C_RESET,
        C_YELLOW, C_RESET, C_YELLOW, C_RESET, C_YELLOW, C_RESET,
        C_YELLOW, C_RESET, C_YELLOW, C_RESET, C_YELLOW, C_RESET,
        C_YELLOW, C_RESET);
}

static int real_main(int argc, char** argv);

int main(int argc, char** argv) {
    // Silence SIGPIPE (we already use MSG_NOSIGNAL, but keep it bulletproof).
    signal(SIGPIPE, SIG_IGN);
    g_vt = enable_vt();
    try {
        return real_main(argc, argv);
    }
    catch (const std::exception& e) {
        if (g_vt) printf("\r\033[K");
        fprintf(stderr, "%sFatal: %s%s\n", C_BOLD_RED, e.what(), C_RESET);
        return 1;
    }
}

// ===========================================================================
// Per-slot state
// ===========================================================================

struct Slot {
    cudaStream_t stream = nullptr;
    cudaEvent_t  ev = nullptr;
    u32* d_res = nullptr;
    u32* h_res = nullptr;
    uint64_t idx_base = 0;
    Job job;
    uint8_t header[32] = { 0 };
    uint8_t target_full[64] = { 0 };
    uint8_t nonce_tmpl[64] = { 0 };
};

// ===========================================================================
// Main
// ===========================================================================

static int real_main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    std::string pool = "qtc.kryptex.network:7049";
    std::string wallet, worker = "gpu";
    uint32_t iters = 64, blocks = 4096, nstreams = 3;
    int device = 0;
    const uint32_t block_size = 256;
    bool once = false, dry = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](std::string& dst) { if (i + 1 < argc) dst = argv[++i]; };
        if (a == "--pool") next(pool);
        else if (a == "--wallet") next(wallet);
        else if (a == "--worker") next(worker);
        else if (a == "--device") { std::string t; next(t); device = atoi(t.c_str()); }
        else if (a == "--iters") { std::string t; next(t); iters = (uint32_t)strtoul(t.c_str(), nullptr, 10); }
        else if (a == "--blocks") { std::string t; next(t); blocks = (uint32_t)strtoul(t.c_str(), nullptr, 10); }
        else if (a == "--streams") { std::string t; next(t); nstreams = (uint32_t)strtoul(t.c_str(), nullptr, 10); }
        else if (a == "--once")   once = true;
        else if (a == "--dry-run") dry = true;
        else if (a == "--version") { printf("%sKanQ %s%s\n", C_BOLD, CCMINER_VERSION, C_RESET); return 0; }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "%sUnknown arg: %s%s\n\n", C_RED, a.c_str(), C_RESET); usage(); return 1; }
    }
    if (wallet.empty() || iters == 0 || blocks == 0) { usage(); return 1; }
    if (nstreams < 1) nstreams = 1;
    if (nstreams > 4) nstreams = 4;

    printf("%sKanQ %s%s\n", C_BOLD, CCMINER_VERSION, C_RESET);

    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    bool nvml_ok = nvml_init(device);
    GpuStats gpu_init;
    if (nvml_ok) gpu_init = gpu_query();

    const uint64_t per_launch = (uint64_t)blocks * block_size * iters;

    printf("%sGPU #%d:%s %s%s%s  SMs=%d  grid=%ux%u  iters/thr=%u  streams=%u  (%.3g nonces/round)\n",
        C_BOLD, device, C_RESET,
        C_CYAN, prop.name, C_RESET,
        prop.multiProcessorCount, blocks, block_size, iters, nstreams,
        (double)per_launch);

    if (nvml_ok && gpu_init.name[0]) {
        printf("  %sNVML:%s %s  %spower limit:%s %uW\n",
            C_DIM, C_RESET, gpu_init.name,
            C_DIM, C_RESET, gpu_init.power_limit_mw / 1000);
    }
    else {
        printf("  %sNVML: unavailable — GPU stats disabled%s\n", C_DIM, C_RESET);
    }

    printf("%sPool:%s %-26s  %swallet:%s %s  %sworker:%s %s\n\n",
        C_BOLD, C_RESET, pool.c_str(),
        C_BOLD, C_RESET, wallet.c_str(),
        C_BOLD, C_RESET, worker.c_str());

    // ---- CUDA resources (allocated once, reused across reconnects) ----
    const size_t res_bytes = (1 + MAX_HITS) * sizeof(u32);
    std::vector<Slot> slots(nstreams);

    for (uint32_t i = 0; i < nstreams; i++) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&slots[i].stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&slots[i].d_res, res_bytes));
        CUDA_CHECK(cudaEventCreateWithFlags(&slots[i].ev, cudaEventDisableTiming));
        slots[i].h_res = (u32*)malloc(res_bytes);
        memset(slots[i].h_res, 0, res_bytes);
    }

    MiningParams params{};
    params.total_threads = blocks * block_size;
    params.nonces_per_thread = iters;

    uint8_t nonce_tmpl_global[64];
    memset(nonce_tmpl_global, 0, sizeof(nonce_tmpl_global));
    {
        std::random_device rd;
        for (int i = 0; i < 8; i++) nonce_tmpl_global[4 + i] = (uint8_t)rd();
    }

    // ---- Reconnect loop ----
    bool first_connect = true;

    for (;;) {
        if (!first_connect) {
            print_event("%s[%sRECONNECT%s]%s waiting 5s before reconnect...",
                C_BOLD_YELLOW, C_RESET, C_RESET, C_DIM);
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
        first_connect = false;

        Pool p;

        // -- connect with retry --
        bool connected = false;
        for (int attempt = 0; attempt < 5; attempt++) {
            if (p.connect_to(pool)) { connected = true; break; }
            int delay = 3 * (attempt + 1);
            print_event("%s[%sRECONNECT%s]%s retry %d/%d in %ds",
                C_BOLD_YELLOW, C_RESET, C_RESET, C_DIM,
                attempt + 1, 5, delay);
            std::this_thread::sleep_for(std::chrono::seconds(delay));
        }
        if (!connected) {
            print_event("%s[%sFATAL%s]%s could not connect after 5 attempts, retrying in 15s",
                C_BOLD_RED, C_RESET, C_RESET, C_DIM);
            std::this_thread::sleep_for(std::chrono::seconds(15));
            continue;
        }

        p.login(wallet, worker);

        // -- wait for first job --
        Job job;
        auto t0 = std::chrono::steady_clock::now();
        bool got_job = false;
        bool need_reconnect = false;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 30) {
            try {
                p.pump();
            }
            catch (const std::exception& e) {
                print_event("%s[%sDISCONNECT%s]%s %s",
                    C_BOLD_RED, C_RESET, C_RESET, C_DIM, e.what());
                need_reconnect = true;
                break;
            }
            job = p.job();
            if (job.valid) { got_job = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (need_reconnect) continue;
        if (!got_job) {
            print_event("%s[%sTIMEOUT%s]%s no job in 30s, reconnecting",
                C_BOLD_YELLOW, C_RESET, C_RESET, C_DIM);
            continue;
        }

        // -- prepare mining state --
        uint64_t last_seq = 0;
        Job cur;
        double hashes_total = 0;
        uint64_t launches = 0, gpu_false = 0;
        uint64_t launch_idx_base = 0;
        auto bench_start = std::chrono::steady_clock::now();
        auto last_status = bench_start;
        int last_acc = 0, last_rej = 0;
        uint32_t cur_slot = 0;

        auto handle_new_job = [&](const Job& j) -> bool {
            uint8_t hdr[32];
            if (!hex_to_bytes(j.mining_hash, hdr, 32)) {
                print_event("%s[%sERROR%s]%s bad mining_hash: %s",
                    C_BOLD_RED, C_RESET, C_RESET, C_DIM, j.mining_hash.c_str());
                return false;
            }
            uint8_t nt[64];
            memcpy(nt, nonce_tmpl_global, 64);
            memset(nt, 0, 4);
            if (j.extranonce.size() < 8 || !hex_to_bytes(j.extranonce.substr(0, 8), nt, 4)) {
                print_event("%s[%sWARN%s]%s bad extranonce: %s",
                    C_BOLD_YELLOW, C_RESET, C_RESET, C_DIM, j.extranonce.c_str());
            }
            uint8_t tf[64];
            qpow_host::target_full512(parse_u64(j.difficulty), tf);
            qpow_host::target_hi_words(tf, params.target_hi);
            qpow_host::prestate_from_input(hdr, nt, params.prestate);
            memcpy(nonce_tmpl_global, nt, 64);
            cur = j;
            return true;
            };

        auto snapshot_slot = [&](Slot& s) {
            uint8_t hdr[32]; hex_to_bytes(cur.mining_hash, hdr, 32);
            uint8_t nt[64];  memcpy(nt, nonce_tmpl_global, 64); memset(nt, 0, 4);
            if (cur.extranonce.size() >= 8) hex_to_bytes(cur.extranonce.substr(0, 8), nt, 4);
            uint8_t tf[64];  qpow_host::target_full512(parse_u64(cur.difficulty), tf);
            memcpy(s.header, hdr, 32);
            memcpy(s.target_full, tf, 64);
            memcpy(s.nonce_tmpl, nt, 64);
            };

        auto process_results = [&](Slot& s) {
            u32 hits = s.h_res[0];
            for (u32 k = 0; k < hits && k < (u32)MAX_HITS; k++) {
                uint64_t idx = s.idx_base + s.h_res[1 + k];
                uint8_t nonce[64];
                memcpy(nonce, s.nonce_tmpl, 64);
                for (int b = 0; b < 8; b++) nonce[56 + b] = (uint8_t)(idx >> (8 * (7 - b)));
                uint8_t in[96], digest[64];
                memcpy(in, s.header, 32);
                memcpy(in + 32, nonce, 64);
                qpow_host::hash_squeeze_twice(in, digest);
                if (!qpow_host::lt_be(digest, s.target_full, 64)) {
                    gpu_false++;
                    print_event("%s[%sFAIL%s]%s GPU candidate failed host verify, total=%llu",
                        C_BOLD_RED, C_RESET, C_RESET, C_DIM,
                        (unsigned long long)gpu_false);
                    continue;
                }
                std::string nonce_hex = bytes_to_hex(nonce, 64);
                std::string result_hex = bytes_to_hex(digest, 64);
                print_event("%s*** %sSHARE FOUND%s ***%s\n"
                    "     %snonce=%s%s\n"
                    "     %sresult=%s%s",
                    C_BOLD, C_BOLD_YELLOW, C_RESET, C_RESET,
                    C_DIM, C_RESET, nonce_hex.c_str(),
                    C_DIM, C_RESET, result_hex.c_str());
                if (dry) {
                    print_event("  %s[DRY-RUN]%s not submitting", C_DIM, C_RESET);
                    if (once) std::exit(0);
                    continue;
                }
                p.pump();
                print_event("  %s[SUBMIT]%s job=%s", C_BLUE, C_RESET, s.job.job_id.c_str());
                p.submit(s.job.job_id, nonce_hex, result_hex);
            }
            };

        if (!handle_new_job(job)) continue;
        last_seq = p.job_seq();

        // -- pre-fill N-1 streams --
        for (uint32_t s = 0; s + 1 < nstreams; s++) {
            Slot& sl = slots[s];
            params.idx_base = launch_idx_base;
            CUDA_CHECK(cudaMemsetAsync(sl.d_res, 0, res_bytes, sl.stream));
            mine_kernel << <blocks, block_size, 0, sl.stream >> > (sl.d_res, params);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(sl.ev, sl.stream));
            sl.idx_base = launch_idx_base;
            sl.job = cur;
            snapshot_slot(sl);
            launch_idx_base += per_launch;
            launches++;
        }

        // -- main pipeline --
        bool mining_reconnect = false;
        for (;;) {
            try {
                p.pump();
            }
            catch (const std::exception& e) {
                print_event("%s[%sDISCONNECT%s]%s %s",
                    C_BOLD_RED, C_RESET, C_RESET, C_DIM, e.what());
                mining_reconnect = true;
                break;
            }

            uint64_t seq = p.job_seq();
            if (seq != last_seq) {
                Job j = p.job();
                if (handle_new_job(j)) {
                    last_seq = seq;
                    launch_idx_base = 0;
                }
                else {
                    last_seq = seq;
                }
            }

            Slot& sl = slots[cur_slot];
            snapshot_slot(sl);
            params.idx_base = launch_idx_base;
            CUDA_CHECK(cudaMemsetAsync(sl.d_res, 0, res_bytes, sl.stream));
            mine_kernel << <blocks, block_size, 0, sl.stream >> > (sl.d_res, params);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(sl.ev, sl.stream));
            sl.idx_base = launch_idx_base;
            sl.job = cur;
            launch_idx_base += per_launch;
            launches++;

            uint32_t old_slot = (cur_slot + 1) % nstreams;
            Slot& os = slots[old_slot];
            CUDA_CHECK(cudaEventSynchronize(os.ev));
            CUDA_CHECK(cudaMemcpyAsync(os.h_res, os.d_res, res_bytes,
                cudaMemcpyDeviceToHost, os.stream));
            CUDA_CHECK(cudaStreamSynchronize(os.stream));

            hashes_total += (double)per_launch;
            process_results(os);

            auto now = std::chrono::steady_clock::now();
            if (p.accepted() != last_acc || p.rejected() != last_rej) {
                last_acc = p.accepted();
                last_rej = p.rejected();
                if (once && last_acc > 0) {
                    printf("\n");
                    return 0;
                }
            }

            double since_status = std::chrono::duration<double>(now - last_status).count();
            if (since_status >= 0.5) {
                last_status = now;
                double el = std::chrono::duration<double>(now - bench_start).count();
                double mh = el > 0 ? hashes_total / el / 1e6 : 0;
                GpuStats gs;
                if (nvml_ok) gs = gpu_query();

                char buf[512];
                if (nvml_ok && gs.available) {
                    const char* tc = temp_color(gs.temp);
                    unsigned int pwr_w = gs.power_mw / 1000;
                    unsigned int plim_w = gs.power_limit_mw / 1000;
                    snprintf(buf, sizeof(buf),
                        "%s%4.0fs%s "
                        "%s%6.1f%s MH/s "
                        "%s%2u C%s "
                        "%uMHz "
                        "%s%uW%s/%uW "
                        "fan:%u%% "
                        "u:%u%% "
                        "%sok:%d%s "
                        "%srej:%d%s "
                        "%sjob:%.10s%s",
                        C_BOLD, (int)el, C_RESET,
                        C_GREEN, mh, C_RESET,
                        tc, gs.temp, C_RESET,
                        gs.clock_graphics,
                        C_MAGENTA, pwr_w, C_RESET, plim_w,
                        gs.fan_pct,
                        gs.util_gpu,
                        C_GREEN, p.accepted(), C_RESET,
                        C_RED, p.rejected(), C_RESET,
                        C_DIM, cur.job_id.c_str(), C_RESET);
                }
                else {
                    snprintf(buf, sizeof(buf),
                        "%s%4.0fs%s "
                        "%s%6.1f%s MH/s "
                        "%sok:%d%s %srej:%d%s "
                        "%sjob:%.10s%s",
                        C_BOLD, (int)el, C_RESET,
                        C_GREEN, mh, C_RESET,
                        C_GREEN, p.accepted(), C_RESET,
                        C_RED, p.rejected(), C_RESET,
                        C_DIM, cur.job_id.c_str(), C_RESET);
                }
                print_status("%s", buf);
            }

            cur_slot = (cur_slot + 1) % nstreams;
        }

        // mining loop exited — either disconnect or --once handled
        if (mining_reconnect) continue;
        // unreachable in normal flow (only --once returns directly)
    }
}