#pragma once
#include <cstdint>

#include "constants.cuh"

namespace qpow {

    typedef uint32_t u32;
    typedef uint64_t u64;

    constexpr u64 P = 0xFFFFFFFF00000001ULL;
    constexpr u64 EPS = 0xFFFFFFFFULL;
    constexpr u32 EPS32 = 0xFFFFFFFFu;

    constexpr int WIDTH = 12;
    constexpr int RATE = 8;
    constexpr int MAX_HITS = 8;

#ifndef QPOW_NONCE_BATCH
#define QPOW_NONCE_BATCH 2
#endif
    constexpr int NB = QPOW_NONCE_BATCH;

    // ---------------------------------------------------------------------------
    // Goldilocks field arithmetic
    // ---------------------------------------------------------------------------

    __device__ __forceinline__ u64 gf_add(u64 a, u64 b) {
        u64 r;
        asm("{\n\t"
            ".reg .u64 c;\n\t"
            "add.cc.u64 %0, %1, %2;\n\t"
            "addc.u64 c, 0, 0;\n\t"
            "mad.lo.u64 %0, c, %3, %0;\n\t"
            "}" : "=l"(r) : "l"(a), "l"(b), "l"(EPS));
        return r;
    }

    __device__ __forceinline__ void mul64wide(u64 a, u64 b, u32& r0, u32& r1, u32& r2, u32& r3) {
        u64 lo, hi;
        asm("mul.lo.u64 %0, %2, %3;\n\t"
            "mul.hi.u64 %1, %2, %3;\n\t"
            : "=l"(lo), "=l"(hi) : "l"(a), "l"(b));
        r0 = (u32)lo; r1 = (u32)(lo >> 32);
        r2 = (u32)hi; r3 = (u32)(hi >> 32);
    }

    __device__ __forceinline__ u64 reduce128(u32 r0, u32 r1, u32 r2, u32 r3) {
        u32 o0, o1;
        asm("{\n\t"
            ".reg .u32 c;\n\t"
            "mad.lo.cc.u32 %0, %4, %6, %2;\n\t"
            "madc.hi.cc.u32 %1, %4, %6, %3;\n\t"
            "addc.u32 c, %5, 0;\n\t"
            "addc.u32 %1, %1, 0;\n\t"
            "sub.cc.u32 %0, %0, c;\n\t"
            "subc.u32 %1, %1, 0;\n\t"
            "}" : "=&r"(o0), "=&r"(o1)
            : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(EPS32));
        return ((u64)o1 << 32) | (u64)o0;
    }


    __device__ __forceinline__ u64 gf_mul(u64 a, u64 b) {
        u32 o0, o1;
        asm("{\n\t"
            ".reg .u64 lo, hi;\n\t"
            ".reg .u32 r0, r1, r2, r3, c;\n\t"
            "mul.lo.u64 lo, %2, %3;\n\t"
            "mul.hi.u64 hi, %2, %3;\n\t"
            "mov.b64 {r0, r1}, lo;\n\t"
            "mov.b64 {r2, r3}, hi;\n\t"
            "mad.lo.cc.u32 %0, r2, %4, r0;\n\t"
            "madc.hi.cc.u32 %1, r2, %4, r1;\n\t"
            "addc.u32 c, r3, 0;\n\t"
            "addc.u32 %1, %1, 0;\n\t"
            "sub.cc.u32 %0, %0, c;\n\t"
            "subc.u32 %1, %1, 0;\n\t"
            "}" : "=&r"(o0), "=&r"(o1) : "l"(a), "l"(b), "r"(EPS32));
        return ((u64)o1 << 32) | (u64)o0;
    }

    __device__ __forceinline__ u64 gf_sqr(u64 a) {
        u32 a0 = (u32)a, a1 = (u32)(a >> 32);
        u64 ll = (u64)a0 * a0;
        u64 lh = (u64)a0 * a1;
        u64 hh = (u64)a1 * a1;
        u64 mid = lh << 1;
        u32 mid_top = (u32)(lh >> 63);
        u32 r0 = (u32)ll, r1, r2, r3;
        asm("{\n\t"
            "add.cc.u32 %0, %3, %4;\n\t"
            "addc.cc.u32 %1, %5, %6;\n\t"
            "addc.u32 %2, %7, %8;\n\t"
            "}" : "=&r"(r1), "=&r"(r2), "=&r"(r3)
            : "r"((u32)(ll >> 32)), "r"((u32)mid), "r"((u32)hh),
            "r"((u32)(mid >> 32)), "r"((u32)(hh >> 32)), "r"(mid_top));
        return reduce128(r0, r1, r2, r3);
    }

    __device__ __forceinline__ u64 gf_sbox(u64 x) {
        u64 x2 = gf_sqr(x);
        u64 x3 = gf_mul(x2, x);
        u64 x4 = gf_sqr(x2);
        return gf_mul(x4, x3);
    }

    __device__ __forceinline__ u64 gf_canon(u64 a) {
        return a - ((a >= P) ? P : 0ULL);
    }

    // ---------------------------------------------------------------------------
    // Wide64
    // ---------------------------------------------------------------------------

    struct Wide64 { u64 lo, hi; };

    __device__ __forceinline__ Wide64 wide_from(u64 x) { return { x, 0 }; }

    __device__ __forceinline__ void wide_add(Wide64& w, u64 x) {
        asm("{\n\t"
            "add.cc.u64 %0, %0, %2;\n\t"
            "addc.u64 %1, %1, 0;\n\t"
            "}" : "+l"(w.lo), "+l"(w.hi) : "l"(x));
    }

    __device__ __forceinline__ void wide_add_wide(Wide64& w, const Wide64& x) {
        asm("{\n\t"
            "add.cc.u64 %0, %0, %2;\n\t"
            "addc.u64 %1, %1, %3;\n\t"
            "}" : "+l"(w.lo), "+l"(w.hi) : "l"(x.lo), "l"(x.hi));
    }

    __device__ __forceinline__ u64 wide_reduce(const Wide64& w) {
        return reduce128((u32)w.lo, (u32)(w.lo >> 32), (u32)w.hi, (u32)(w.hi >> 32));
    }

    __device__ __forceinline__ void add128_wide(u32& r0, u32& r1, u32& r2, u32& r3, const Wide64& w) {
        u32 w0 = (u32)w.lo, w1 = (u32)(w.lo >> 32), w2 = (u32)w.hi;
        asm("{\n\t"
            "add.cc.u32 %0, %0, %4;\n\t"
            "addc.cc.u32 %1, %1, %5;\n\t"
            "addc.cc.u32 %2, %2, %6;\n\t"
            "addc.u32 %3, %3, 0;\n\t"
            "}" : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3)
            : "r"(w0), "r"(w1), "r"(w2));
    }

    // ---------------------------------------------------------------------------
    // ext_layer / int_round_p (single nonce, unchanged)
    // ---------------------------------------------------------------------------

    __device__ __forceinline__ void ext_layer(u64* state, const u64* rc12) {
        Wide64 y[12];
#pragma unroll
        for (int chunk = 0; chunk < 3; chunk++) {
            int o = chunk * 4;
            u64 x0 = state[o], x1 = state[o + 1], x2 = state[o + 2], x3 = state[o + 3];
            Wide64 t01 = wide_from(x0); wide_add(t01, x1);
            Wide64 t23 = wide_from(x2); wide_add(t23, x3);
            Wide64 t0123 = t01; wide_add_wide(t0123, t23);
            Wide64 t01123 = t0123; wide_add(t01123, x1);
            Wide64 t01233 = t0123; wide_add(t01233, x3);
            y[o + 3] = t01233; wide_add(y[o + 3], x0); wide_add(y[o + 3], x0);
            y[o + 1] = t01123; wide_add(y[o + 1], x2); wide_add(y[o + 1], x2);
            y[o] = t01123; wide_add_wide(y[o], t01);
            y[o + 2] = t01233; wide_add_wide(y[o + 2], t23);
        }
        Wide64 sums[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            sums[k] = y[k];
            wide_add_wide(sums[k], y[k + 4]);
            wide_add_wide(sums[k], y[k + 8]);
        }
#pragma unroll
        for (int i = 0; i < 12; i++) {
            Wide64 w = y[i];
            wide_add_wide(w, sums[i % 4]);
            wide_add(w, rc12[i]);
            state[i] = wide_reduce(w);
        }
    }

    __device__ __forceinline__ u64 int_round_p(u64* state, u64 x, u64 rc0) {
        Wide64 s; s.lo = state[1]; s.hi = 0;
        Wide64 sb;
        sb.lo = state[2];  sb.hi = 0; wide_add(sb, state[3]);  wide_add_wide(s, sb);
        sb.lo = state[4];  sb.hi = 0; wide_add(sb, state[5]);  wide_add_wide(s, sb);
        sb.lo = state[6];  sb.hi = 0; wide_add(sb, state[7]);  wide_add_wide(s, sb);
        sb.lo = state[8];  sb.hi = 0; wide_add(sb, state[9]);  wide_add_wide(s, sb);
        sb.lo = state[10]; sb.hi = 0; wide_add(sb, state[11]); wide_add_wide(s, sb);
        wide_add(s, x);

        Wide64 s0 = s; wide_add(s0, rc0);

        u32 r0, r1, r2, r3;
        mul64wide(x, MATRIX_DIAG[0], r0, r1, r2, r3);
        add128_wide(r0, r1, r2, r3, s0);
        u64 out0 = reduce128(r0, r1, r2, r3);
#pragma unroll
        for (int i = 1; i < 12; i++) {
            mul64wide(state[i], MATRIX_DIAG[i], r0, r1, r2, r3);
            add128_wide(r0, r1, r2, r3, s);
            state[i] = reduce128(r0, r1, r2, r3);
        }
        return out0;
    }

    // ---------------------------------------------------------------------------
    // Batched ext_layer with explicit interleaving across NB chains
    // ---------------------------------------------------------------------------

    __device__ __forceinline__ void ext_layer_batch(u64 st[NB][WIDTH], const u64* rc12) {
        Wide64 y[NB][12];

#pragma unroll
        for (int chunk = 0; chunk < 3; chunk++) {
            int o = chunk * 4;

            // t01[b] = x0 + x1
            Wide64 t01[NB];
#pragma unroll
            for (int b = 0; b < NB; b++) {
                t01[b] = wide_from(st[b][o]);
                wide_add(t01[b], st[b][o + 1]);
            }

            // t23[b] = x2 + x3
            Wide64 t23[NB];
#pragma unroll
            for (int b = 0; b < NB; b++) {
                t23[b] = wide_from(st[b][o + 2]);
                wide_add(t23[b], st[b][o + 3]);
            }

            // t0123[b] = t01 + t23
            Wide64 t0123[NB];
#pragma unroll
            for (int b = 0; b < NB; b++) {
                t0123[b] = t01[b];
                wide_add_wide(t0123[b], t23[b]);
            }

            // t01123[b] = t0123 + x1
            Wide64 t01123[NB];
#pragma unroll
            for (int b = 0; b < NB; b++) {
                t01123[b] = t0123[b];
                wide_add(t01123[b], st[b][o + 1]);
            }

            // t01233[b] = t0123 + x3
            Wide64 t01233[NB];
#pragma unroll
            for (int b = 0; b < NB; b++) {
                t01233[b] = t0123[b];
                wide_add(t01233[b], st[b][o + 3]);
            }

            // y[o+3] = t01233 + x0 + x0
#pragma unroll
            for (int b = 0; b < NB; b++) {
                y[b][o + 3] = t01233[b];
                wide_add(y[b][o + 3], st[b][o]);
                wide_add(y[b][o + 3], st[b][o]);
            }

            // y[o+1] = t01123 + x2 + x2
#pragma unroll
            for (int b = 0; b < NB; b++) {
                y[b][o + 1] = t01123[b];
                wide_add(y[b][o + 1], st[b][o + 2]);
                wide_add(y[b][o + 1], st[b][o + 2]);
            }

            // y[o] = t01123 + t01
#pragma unroll
            for (int b = 0; b < NB; b++) {
                y[b][o] = t01123[b];
                wide_add_wide(y[b][o], t01[b]);
            }

            // y[o+2] = t01233 + t23
#pragma unroll
            for (int b = 0; b < NB; b++) {
                y[b][o + 2] = t01233[b];
                wide_add_wide(y[b][o + 2], t23[b]);
            }
        }

        // sums[b][k] = y[b][k] + y[b][k+4] + y[b][k+8]
        Wide64 sums[NB][4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
#pragma unroll
            for (int b = 0; b < NB; b++) {
                sums[b][k] = y[b][k];
                wide_add_wide(sums[b][k], y[b][k + 4]);
                wide_add_wide(sums[b][k], y[b][k + 8]);
            }
        }

        // state[i] = reduce(y[i] + sums[i%4] + rc[i])
#pragma unroll
        for (int i = 0; i < 12; i++) {
#pragma unroll
            for (int b = 0; b < NB; b++) {
                Wide64 w = y[b][i];
                wide_add_wide(w, sums[b][i % 4]);
                wide_add(w, rc12[i]);
                st[b][i] = wide_reduce(w);
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Batched int_round_p with explicit interleaving across NB chains
    // ---------------------------------------------------------------------------

    __device__ __forceinline__ void int_round_p_batch(u64 st[NB][WIDTH], u64 x[NB], u64 rc0) {
        // Step 1: s[b] = state[1] + sum of pairs (2,3)..(10,11) + x
        Wide64 s[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) { s[b].lo = st[b][1]; s[b].hi = 0; }

#pragma unroll
        for (int i = 2; i < 12; i += 2) {
#pragma unroll
            for (int b = 0; b < NB; b++) {
                Wide64 sb; sb.lo = st[b][i]; sb.hi = 0;
                wide_add(sb, st[b][i + 1]);
                wide_add_wide(s[b], sb);
            }
        }

#pragma unroll
        for (int b = 0; b < NB; b++) wide_add(s[b], x[b]);

        // Step 2: s0[b] = s[b] + rc0
        Wide64 s0[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) { s0[b] = s[b]; wide_add(s0[b], rc0); }

        // Step 3: out0[b] = reduce(mul(x[b], DIAG[0]) + s0[b])
        u32 r0[NB], r1[NB], r2[NB], r3[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) mul64wide(x[b], MATRIX_DIAG[0], r0[b], r1[b], r2[b], r3[b]);
#pragma unroll
        for (int b = 0; b < NB; b++) add128_wide(r0[b], r1[b], r2[b], r3[b], s0[b]);
#pragma unroll
        for (int b = 0; b < NB; b++) x[b] = reduce128(r0[b], r1[b], r2[b], r3[b]);

        // Step 4: state[i] = reduce(mul(state[i], DIAG[i]) + s) for i=1..11
#pragma unroll
        for (int i = 1; i < 12; i++) {
#pragma unroll
            for (int b = 0; b < NB; b++) mul64wide(st[b][i], MATRIX_DIAG[i], r0[b], r1[b], r2[b], r3[b]);
#pragma unroll
            for (int b = 0; b < NB; b++) add128_wide(r0[b], r1[b], r2[b], r3[b], s[b]);
#pragma unroll
            for (int b = 0; b < NB; b++) st[b][i] = reduce128(r0[b], r1[b], r2[b], r3[b]);
        }
    }

    // ---------------------------------------------------------------------------
    // Batch versions — with interleaving optimization
    // ---------------------------------------------------------------------------

    __device__ __forceinline__ void permute_after_initial_batch(u64 st[NB][WIDTH]) {
        // 4 full rounds
        for (int r = 0; r < 4; r++) {
#pragma unroll
            for (int i = 0; i < 12; i++)
#pragma unroll
                for (int b = 0; b < NB; b++) st[b][i] = gf_sbox(st[b][i]);
            ext_layer_batch(st, RC_INITIAL_EXT[r + 1]);
        }

        // 21 partial rounds — NB independent chains interleaved
        u64 x[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) x[b] = gf_sbox(st[b][0]);
        for (int r = 0; r < 21; r++) {
            int_round_p_batch(st, x, RC_INTERNAL_EXT[r + 1]);
#pragma unroll
            for (int b = 0; b < NB; b++) x[b] = gf_sbox(x[b]);
        }
        int_round_p_batch(st, x, 0ULL);
#pragma unroll
        for (int b = 0; b < NB; b++) st[b][0] = x[b];

#pragma unroll
        for (int i = 0; i < 12; i++)
#pragma unroll
            for (int b = 0; b < NB; b++) st[b][i] = gf_add(st[b][i], RC_TERMINAL_EXT[0][i]);

        // 4 final rounds
        for (int r = 0; r < 4; r++) {
#pragma unroll
            for (int i = 0; i < 12; i++)
#pragma unroll
                for (int b = 0; b < NB; b++) st[b][i] = gf_sbox(st[b][i]);
            ext_layer_batch(st, RC_TERMINAL_EXT[r + 1]);
        }
    }

    __device__ __forceinline__ void permute_twice_after_initial_batch(u64 st[NB][WIDTH]) {
        // pass 0: permute_after_initial, then add (1,1) into st[0], st[1]
        permute_after_initial_batch(st);
#pragma unroll
        for (int b = 0; b < NB; b++) st[b][0] = gf_add(st[b][0], 1ULL);
#pragma unroll
        for (int b = 0; b < NB; b++) st[b][1] = gf_add(st[b][1], 1ULL);
        ext_layer_batch(st, RC_INITIAL_EXT[0]);
        // pass 1: permute_after_initial
        permute_after_initial_batch(st);
    }

    // ---------------------------------------------------------------------------
    // Single-nonce permute (kept for hash_squeeze_twice self-test)
    // ---------------------------------------------------------------------------

#ifndef QPOW_UNROLL
#define QPOW_UNROLL 0
#endif
#if QPOW_UNROLL >= 1
#define QPOW_UNROLL_FULL_ROUNDS _Pragma("unroll")
#define QPOW_UNROLL_PARTIAL_ROUNDS _Pragma("unroll")
#define QPOW_UNROLL_TERMINAL_ROUNDS _Pragma("unroll")
#else
#define QPOW_UNROLL_FULL_ROUNDS _Pragma("unroll 2")
#define QPOW_UNROLL_PARTIAL_ROUNDS _Pragma("unroll 3")
#define QPOW_UNROLL_TERMINAL_ROUNDS _Pragma("unroll 1")
#endif
#if QPOW_UNROLL >= 2
#define QPOW_UNROLL_PASSES _Pragma("unroll")
#else
#define QPOW_UNROLL_PASSES _Pragma("unroll 1")
#endif

    __device__ __forceinline__ void permute_after_initial(u64* state) {
        QPOW_UNROLL_FULL_ROUNDS
            for (int r = 0; r < 4; r++) {
#pragma unroll
                for (int i = 0; i < 12; i++) state[i] = gf_sbox(state[i]);
                ext_layer(state, RC_INITIAL_EXT[r + 1]);
            }
        u64 x = gf_sbox(state[0]);
        QPOW_UNROLL_PARTIAL_ROUNDS
            for (int r = 0; r < 21; r++) {
                x = gf_sbox(int_round_p(state, x, RC_INTERNAL_EXT[r + 1]));
            }
        state[0] = int_round_p(state, x, 0ULL);
#pragma unroll
        for (int i = 0; i < 12; i++) state[i] = gf_add(state[i], RC_TERMINAL_EXT[0][i]);
        QPOW_UNROLL_TERMINAL_ROUNDS
            for (int r = 0; r < 4; r++) {
#pragma unroll
                for (int i = 0; i < 12; i++) state[i] = gf_sbox(state[i]);
                ext_layer(state, RC_TERMINAL_EXT[r + 1]);
            }
    }

    __device__ __forceinline__ void permute(u64* state) {
        ext_layer(state, RC_INITIAL_EXT[0]);
        permute_after_initial(state);
    }

    __device__ __forceinline__ void permute_twice_after_initial(u64* state) {
        QPOW_UNROLL_PASSES
            for (int pass = 0; pass < 2; pass++) {
                if (pass != 0) ext_layer(state, RC_INITIAL_EXT[0]);
                permute_after_initial(state);
                if (pass == 0) {
                    state[0] = gf_add(state[0], 1ULL);
                    state[1] = gf_add(state[1], 1ULL);
                }
            }
    }

    // ---------------------------------------------------------------------------
    // Sponge (self-test)
    // ---------------------------------------------------------------------------

    __device__ __forceinline__ u32 load_le32(const uint8_t* p) {
        return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
    }

    __device__ __forceinline__ void store_digest32(const u64* state, uint8_t* out32) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            u64 v = gf_canon(state[i]);
#pragma unroll
            for (int b = 0; b < 8; b++) out32[i * 8 + b] = (uint8_t)(v >> (8 * b));
        }
    }

    __device__ __forceinline__ void hash_squeeze_twice(const uint8_t* data, uint8_t* out64) {
        u64 s[WIDTH];
#pragma unroll
        for (int i = 0; i < WIDTH; i++) s[i] = 0;
#pragma unroll 1
        for (int blk = 0; blk < 3; blk++) {
#pragma unroll
            for (int i = 0; i < RATE; i++) s[i] = gf_add(s[i], (u64)load_le32(data + (blk * RATE + i) * 4));
            permute(s);
        }
        s[0] = gf_add(s[0], 1ULL);
        s[1] = gf_add(s[1], 1ULL);
        permute(s);
        store_digest32(s, out64);
        permute(s);
        store_digest32(s, out64 + 32);
    }

    // ---------------------------------------------------------------------------
    // Mining path
    // ---------------------------------------------------------------------------

    __device__ __forceinline__ u32 bswap32(u32 v) {
        return __byte_perm(v, 0, 0x0123);
    }

    __device__ __forceinline__ void absorb_idx(u64* st, u64 idx) {
        u64 x6 = (u64)bswap32((u32)(idx >> 32));
        u64 x7 = (u64)bswap32((u32)idx);
        u64 x6_2 = x6 + x6, x6_3 = x6_2 + x6, x6_4 = x6_2 + x6_2, x6_6 = x6_3 + x6_3;
        u64 x7_2 = x7 + x7, x7_3 = x7_2 + x7, x7_4 = x7_2 + x7_2, x7_6 = x7_3 + x7_3;
        u64 c0 = x6 + x7, c1 = x6_3 + x7, c2 = x6_2 + x7_3, c3 = x6 + x7_2;
        st[0] = gf_add(st[0], c0);
        st[1] = gf_add(st[1], c1);
        st[2] = gf_add(st[2], c2);
        st[3] = gf_add(st[3], c3);
        st[4] = gf_add(st[4], x6_2 + x7_2);
        st[5] = gf_add(st[5], x6_6 + x7_2);
        st[6] = gf_add(st[6], x6_4 + x7_6);
        st[7] = gf_add(st[7], x6_2 + x7_4);
        st[8] = gf_add(st[8], c0);
        st[9] = gf_add(st[9], c1);
        st[10] = gf_add(st[10], c2);
        st[11] = gf_add(st[11], c3);
    }

    // Single-nonce (kept for compatibility)
    __device__ __forceinline__ void first_squeeze_from_prestate(const u64* pre, u64 idx, u64* out4) {
        u64 st[WIDTH];
#pragma unroll
        for (int i = 0; i < WIDTH; i++) st[i] = pre[i];
        absorb_idx(st, idx);
        permute_twice_after_initial(st);
#pragma unroll
        for (int i = 0; i < 4; i++) out4[i] = gf_canon(st[i]);
    }

    // Batched: NB nonces in parallel — operations interleave.
    __device__ __forceinline__ void first_squeeze_from_prestate_batch(
        const u64* pre, u64 idx_base, u32 idx0, u64 out4[NB][4])
    {
        u64 st[NB][WIDTH];
#pragma unroll
        for (int b = 0; b < NB; b++) {
#pragma unroll
            for (int i = 0; i < WIDTH; i++) st[b][i] = pre[i];
            absorb_idx(st[b], idx_base + (u64)(idx0 + b));
        }
        permute_twice_after_initial_batch(st);
#pragma unroll
        for (int b = 0; b < NB; b++)
#pragma unroll
            for (int i = 0; i < 4; i++) out4[b][i] = gf_canon(st[b][i]);
    }

    __device__ __forceinline__ bool first_squeeze_le_target(const u64* out4, const u32* tgt_hi) {
#pragma unroll
        for (int k = 0; k < 8; k++) {
            u64 v = out4[k >> 1];
            u32 h = bswap32((k & 1) ? (u32)(v >> 32) : (u32)v);
            u32 t = tgt_hi[k];
            if (h != t) return h < t;
        }
        return true;
    }

    struct MiningParams {
        u64 prestate[WIDTH];
        u32 target_hi[8];
        u64 idx_base;
        u32 total_threads;
        u32 nonces_per_thread;
    };

#ifndef QPOW_MIN_BLOCKS
#define QPOW_MIN_BLOCKS 1
#endif

    __global__ void __launch_bounds__(256, QPOW_MIN_BLOCKS) mine_kernel(u32* results, const MiningParams params) {
        u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= params.total_threads) return;

        u32 tgt[8];
#pragma unroll
        for (int i = 0; i < 8; i++) tgt[i] = params.target_hi[i];

        u32 base = tid * params.nonces_per_thread;

        // Process NB nonces per iteration
        for (u32 j = 0; j + NB <= params.nonces_per_thread; j += NB) {
#ifdef QPOW_LOCKSTEP
            __syncthreads();
#endif
            u64 out4[NB][4];
            first_squeeze_from_prestate_batch(params.prestate, params.idx_base, base + j, out4);

#pragma unroll
            for (int b = 0; b < NB; b++) {
                if (first_squeeze_le_target(out4[b], tgt)) {
                    u32 logical = base + j + b;
                    u32 slot = atomicAdd(&results[0], 1u);
                    if (slot < (u32)MAX_HITS) results[1 + slot] = logical;
                }
            }
        }

        // Tail (if nonces_per_thread % NB != 0)
        for (u32 j = params.nonces_per_thread - (params.nonces_per_thread % NB);
            j < params.nonces_per_thread; j++) {
            u64 out4[4];
            first_squeeze_from_prestate(params.prestate, params.idx_base + (u64)(base + j), out4);
            if (first_squeeze_le_target(out4, tgt)) {
                u32 logical = base + j;
                u32 slot = atomicAdd(&results[0], 1u);
                if (slot < (u32)MAX_HITS) results[1 + slot] = logical;
            }
        }
    }

} // namespace qpow
