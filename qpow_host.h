// Quantus QPoW — host-side EXACT implementation (Windows/MSVC).
#pragma once
#include <cstdint>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "constants.cuh"

namespace qpow_host {

    constexpr uint64_t P = 0xFFFFFFFF00000001ULL;
    constexpr uint64_t EPS = 0xFFFFFFFFULL;
    constexpr int WIDTH = 12;
    constexpr int RATE = 8;

    inline uint32_t bswap32(uint32_t v) {
#ifdef _MSC_VER
        return _byteswap_ulong(v);
#else
        return __builtin_bswap32(v);
#endif
    }

    // (a + b) mod p
    inline uint64_t add(uint64_t a, uint64_t b) {
        uint64_t lo = a + b;
        uint64_t carry = (lo < a) ? 1ULL : 0ULL;
        uint64_t r = lo + carry * EPS;
        if (r < lo) r += EPS;
        if (r >= P) r -= P;
        return r;
    }

    // (a * b) mod p
    inline uint64_t mul(uint64_t a, uint64_t b) {
#ifdef _MSC_VER
        uint64_t hi;
        uint64_t lo = _umul128(a, b, &hi);
#else
        unsigned __int128 prod = (unsigned __int128)a * b;
        uint64_t lo = (uint64_t)prod;
        uint64_t hi = (uint64_t)(prod >> 64);
#endif
        uint32_t hh = (uint32_t)(hi >> 32), hl = (uint32_t)hi;
        uint64_t t = lo - (uint64_t)hh;
        if (lo < (uint64_t)hh) t -= EPS;
        uint64_t b_term = ((uint64_t)hl << 32) - (uint64_t)hl;
        uint64_t r = t + b_term;
        if (r < t) r += EPS;
        return r;
    }

    inline uint64_t sbox(uint64_t x) {
        uint64_t x2 = mul(x, x);
        uint64_t x3 = mul(x2, x);
        uint64_t x4 = mul(x2, x2);
        return mul(x3, x4);
    }

    inline void external_linear_layer(uint64_t* s) {
        for (int c = 0; c < WIDTH / 4; c++) {
            int i = c * 4;
            uint64_t t01 = add(s[i], s[i + 1]);
            uint64_t t23 = add(s[i + 2], s[i + 3]);
            uint64_t t0123 = add(t01, t23);
            uint64_t t01123 = add(t0123, s[i + 1]);
            uint64_t t01233 = add(t0123, s[i + 3]);
            uint64_t s3 = add(t01233, add(s[i], s[i]));
            uint64_t s1 = add(t01123, add(s[i + 2], s[i + 2]));
            uint64_t s0 = add(t01123, t01);
            uint64_t s2 = add(t01233, t23);
            s[i] = s0; s[i + 1] = s1; s[i + 2] = s2; s[i + 3] = s3;
        }
        uint64_t sums[4];
        for (int k = 0; k < 4; k++) sums[k] = add(add(s[k], s[k + 4]), s[k + 8]);
        for (int i = 0; i < WIDTH; i++) s[i] = add(s[i], sums[i % 4]);
    }

    inline void internal_linear_layer(uint64_t* s) {
        uint64_t total = 0;
        for (int i = 0; i < WIDTH; i++) total = add(total, s[i]);
        for (int i = 0; i < WIDTH; i++) s[i] = add(total, mul(s[i], H_MATRIX_DIAG[i]));
    }

    inline void permute_after_initial(uint64_t* s) {
        for (int r = 0; r < 4; r++) {
            for (int i = 0; i < WIDTH; i++) s[i] = sbox(s[i]);
            external_linear_layer(s);
            if (r + 1 < 4) {
                for (int i = 0; i < WIDTH; i++) s[i] = add(s[i], H_INITIAL_RC[r + 1][i]);
            }
        }
        for (int r = 0; r < 22; r++) {
            s[0] = sbox(add(s[0], H_INTERNAL_RC[r]));
            internal_linear_layer(s);
        }
        for (int r = 0; r < 4; r++) {
            for (int i = 0; i < WIDTH; i++) s[i] = sbox(add(s[i], H_TERMINAL_RC[r][i]));
            external_linear_layer(s);
        }
    }

    inline void permute(uint64_t* s) {
        external_linear_layer(s);
        for (int i = 0; i < WIDTH; i++) s[i] = add(s[i], H_INITIAL_RC[0][i]);
        permute_after_initial(s);
    }

    inline uint32_t load_le32(const uint8_t* p) {
        return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    }

    inline void store_digest32(const uint64_t* s, uint8_t* out32) {
        for (int i = 0; i < 4; i++)
            for (int b = 0; b < 8; b++) out32[i * 8 + b] = (uint8_t)(s[i] >> (8 * b));
    }

    inline void hash_squeeze_twice(const uint8_t* in96, uint8_t* out64) {
        uint64_t s[WIDTH] = { 0 };
        for (int blk = 0; blk < 3; blk++) {
            for (int i = 0; i < RATE; i++) s[i] = add(s[i], load_le32(in96 + (blk * RATE + i) * 4));
            permute(s);
        }
        s[0] = add(s[0], 1);
        s[1] = add(s[1], 1);
        permute(s);
        store_digest32(s, out64);
        permute(s);
        store_digest32(s, out64 + 32);
    }

    inline void prestate_from_input(const uint8_t* header32, const uint8_t* nonce_first56, uint64_t* pre) {
        uint64_t s[WIDTH] = { 0 };
        for (int i = 0; i < RATE; i++) s[i] = add(s[i], load_le32(header32 + i * 4));
        permute(s);
        for (int i = 0; i < RATE; i++) s[i] = add(s[i], load_le32(nonce_first56 + i * 4));
        permute(s);
        for (int i = 0; i < 6; i++) s[i] = add(s[i], load_le32(nonce_first56 + 32 + i * 4));
        external_linear_layer(s);
        for (int i = 0; i < WIDTH; i++) s[i] = add(s[i], H_INITIAL_RC[0][i]);
        memcpy(pre, s, sizeof(s));
    }

    inline void hash_from_prestate(const uint64_t* pre, uint64_t idx, uint8_t* out64) {
        uint64_t s[WIDTH];
        memcpy(s, pre, sizeof(s));
        uint64_t x6 = bswap32((uint32_t)(idx >> 32));
        uint64_t x7 = bswap32((uint32_t)idx);
        uint64_t c[4] = { x6 + x7, 3 * x6 + x7, 2 * x6 + 3 * x7, x6 + 2 * x7 };
        for (int i = 0; i < 4; i++) {
            s[i] = add(s[i], c[i]);
            s[4 + i] = add(s[4 + i], 2 * c[i]);
            s[8 + i] = add(s[8 + i], c[i]);
        }
        permute_after_initial(s);
        s[0] = add(s[0], 1);
        s[1] = add(s[1], 1);
        permute(s);
        store_digest32(s, out64);
        permute(s);
        store_digest32(s, out64 + 32);
    }

    // target = (2^512 - 1) / difficulty. difficulty is uint64_t (< 2^64).
    inline void target_full512(uint64_t difficulty, uint8_t* out64) {
        if (difficulty == 0) difficulty = 1;
        uint64_t limbs[8];
        for (int i = 0; i < 8; i++) limbs[i] = ~0ULL;

        uint64_t rem = 0;
        for (int i = 7; i >= 0; i--) {
#ifdef _MSC_VER
            uint64_t q = _udiv128(rem, limbs[i], difficulty, &rem);
            limbs[i] = q;
#else
            unsigned __int128 cur = ((unsigned __int128)rem << 64) | limbs[i];
            limbs[i] = (uint64_t)(cur / difficulty);
            rem = (uint64_t)(cur % difficulty);
#endif
        }
        for (int k = 0; k < 8; k++) {
            uint64_t v = limbs[7 - k];
            for (int b = 0; b < 8; b++) out64[k * 8 + b] = (uint8_t)(v >> (8 * (7 - b)));
        }
    }

    inline bool lt_be(const uint8_t* a, const uint8_t* b, size_t n) {
        for (size_t i = 0; i < n; i++) if (a[i] != b[i]) return a[i] < b[i];
        return false;
    }

    inline void target_hi_words(const uint8_t* target64, uint32_t* w8) {
        for (int k = 0; k < 8; k++)
            w8[k] = ((uint32_t)target64[4 * k] << 24) | ((uint32_t)target64[4 * k + 1] << 16) |
            ((uint32_t)target64[4 * k + 2] << 8) | (uint32_t)target64[4 * k + 3];
    }

} // namespace qpow_host