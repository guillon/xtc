/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
/*
 * Minimal TVM runtime init function.
 *
 * Initialize each TVM API call slot with the definition
 * from tvm_runtime shared lib (or custom implementation).
 */
#include <stdint.h>

/* We omit actuall function prototype as we only assign the slots there */
extern void TVMBackendAllocWorkspace();
extern void TVMBackendFreeWorkspace();
extern void TVMBackendParallelBarrier();
extern void TVMBackendParallelLaunch();

void (*__TVMBackendAllocWorkspace)();
void (*__TVMBackendFreeWorkspace)();
void (*__TVMBackendParallelBarrier)();
void (*__TVMBackendParallelLaunch)();

void xtc_tvm_init_runtime()
{
    __TVMBackendAllocWorkspace = TVMBackendAllocWorkspace;
    __TVMBackendFreeWorkspace = TVMBackendFreeWorkspace;
    __TVMBackendParallelBarrier = TVMBackendParallelBarrier;
    __TVMBackendParallelLaunch = TVMBackendParallelLaunch;
}
