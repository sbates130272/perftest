/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * Data validation for ROCm device memory.
 *
 * Upstream perftest validates host memory (src/host_validation.c, SIMD
 * polling threads) and CUDA memory (src/cuda_memory.c driving kernels from
 * cuda_kernels.cu).  There is no ROCm equivalent, so --data_validation is
 * refused for --use_rocm at parameter-parse time.
 *
 * This is a *snapshot* validator, deliberately not a port of either.  Device
 * memory cannot be polled in place the way host_validation.c polls host
 * memory, and implementing the live ring-buffer scheme would mean HIP kernels
 * and a staging pipeline.  Instead the whole validation region is copied
 * device->host once, after the run, and every chunk's final contents are
 * checked against the pattern its marker names.
 *
 * What that does and does not buy:
 *
 *   + It catches address-translation and offset faults -- a transfer landing
 *     at the wrong place writes the wrong bytes everywhere, which is exactly
 *     what a bandwidth test cannot see.
 *   + It needs no kernels and no second thread.
 *   - It validates the final state of each chunk, not every cycle, so
 *     bytes_validated is much lower than the host/CUDA backends report for
 *     the same run.  Transient corruption that a later cycle overwrites is
 *     not detected.
 *
 * Report it as what it is; do not compare its bytes_validated against the
 * host path's.
 */

#ifndef ROCM_VALIDATION_H
#define ROCM_VALIDATION_H

#include "memory.h"
#include "validation_common.h"

#ifdef __cplusplus
extern "C" {
#endif

int rocm_validation_init(struct memory_ctx *ctx,
			 const struct validation_config *cfg);
int rocm_validation_start(struct memory_ctx *ctx);
int rocm_validation_stop(struct memory_ctx *ctx,
			 struct data_validation_result *result);
void rocm_validation_destroy(struct memory_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* ROCM_VALIDATION_H */
