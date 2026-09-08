/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * GCC function multiversioning, made ThreadSanitizer-compatible.
 *
 * exp_smooth, fft, and psd select their SIMD kernels with GCC's native C++ function
 * multiversioning: several same-name overloads carrying different [[gnu::target(...)]]
 * attributes, which GCC resolves once at load time. That is the intended mechanism and the
 * one this fleet prefers -- it keeps the dispatch out of the source and lets the compiler
 * own the ISA decision.
 *
 * It has exactly one problem: GCC implements it with an IFUNC resolver, and IFUNC resolvers
 * run during dynamic relocation, BEFORE the ThreadSanitizer runtime initializes. Any binary
 * that links or dlopen()s a multiversioned module under TSan dies inside ld.so, before main:
 *
 *     #1 work<double>::process(...) [clone .resolver] at psd/component.cpp:128
 *     #2 elf_machine_lazy_rel (skip_ifunc=...) at ../sysdeps/x86_64/dl-machine.h:581
 *
 * So under TSan only -- and never in a shipping build -- the non-default versions are
 * compiled out and the default (scalar) version stands alone, emitting no resolver. The
 * fleet's race coverage is about concurrency, not SIMD, so the scalar path is the right
 * thing to instrument anyway.
 *
 * Usage:
 *
 *     COMPS_FMV_DEFAULT
 *     auto kernel(...) -> void { ...scalar... }
 *
 *     #if COMPS_FMV_ENABLED
 *     [[gnu::target("avx2,fma")]]
 *     auto kernel(...) -> void { ...avx2... }
 *     #endif
 *
 * ASan/UBSan are unaffected and need no guard.
 */
#pragma once

#if defined(__SANITIZE_THREAD__)
#  define COMPS_FMV_ENABLED 0
   // A lone function with target("default") and no siblings would still be a versioned
   // function, so drop the attribute entirely rather than leaving a one-version set.
#  define COMPS_FMV_DEFAULT
#else
#  define COMPS_FMV_ENABLED 1
#  define COMPS_FMV_DEFAULT [[gnu::target("default")]]
#endif
