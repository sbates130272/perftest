/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * C interface to the HIP validation kernel in rocm_kernels.hip -- a port of
 * cuda_kernels.cu.
 *
 * The signatures mirror the CUDA plugin's exported entry points one for one;
 * the names carry a rocm_kernel_ prefix because these are linked into
 * libperftest.a rather than dlopen'd from libperftest_kernels.so, so they
 * share a namespace with everything else and must not collide with either
 * the CUDA plugin or the memory_ctx hooks in rocm_validation.c.
 */

#ifndef ROCM_KERNELS_H
#define ROCM_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *rocm_kernel_validation_strerror(int err);

int rocm_kernel_validation_init(void *buffer_base,
				uint64_t markers_offset,
				uint64_t recv_slots_offset,
				uint64_t payload_size,
				uint32_t tx_depth,
				uint32_t num_qps,
				uint32_t chunks_per_qp,
				int validation_mode,
				int device_id,
				int debug_enabled);

int rocm_kernel_validation_start(void *params, int num_blocks,
				 int threads_per_block);
int rocm_kernel_validation_stop(void);
int rocm_kernel_validation_get_stats(uint64_t *chunks_validated,
				     uint64_t *bytes_validated,
				     uint64_t *errors_found,
				     uint64_t *markers_scanned,
				     uint64_t *markers_hit,
				     uint64_t *skipped_steps,
				     uint64_t *race_overwrites,
				     uint64_t *dma_stale_retries,
				     uint64_t *queue_full_drops,
				     uint64_t *stale_work_skips);
int rocm_kernel_validation_get_error(uint32_t *qp_id, uint32_t *chunk_id,
				     uint64_t *byte_offset, uint8_t *expected,
				     uint8_t *actual);
int rocm_kernel_validation_destroy(void);
int rocm_kernel_validation_is_running(void);
int rocm_kernel_validation_check_status(void);

/* 1 if a HIP device is usable, 0 otherwise. */
int rocm_kernels_available(void);

/*
 * EXPERIMENT (not for merge): one-shot post-DMA compare, run in VRAM after
 * the traffic rather than staging the buffer through host memory. Selected
 * with PERFTEST_ROCM_VALIDATION=oneshot. lfsr_mode selects lock-and-predict
 * over the LFSR stream instead of the constant-byte patterns.
 *
 * Declared here rather than at the call site so the prototype is checked
 * against the definition in rocm_kernels.hip.
 */
int rocm_kernel_oneshot_validate(void *markers, void *slots,
				 uint32_t num_qps, uint32_t chunks_per_qp,
				 uint64_t ops_per_chunk, uint64_t payload_size,
				 int lfsr_mode,
				 uint64_t *chunks, uint64_t *bytes,
				 uint64_t *errors, uint64_t *no_lock,
				 uint32_t *err_chunk, uint64_t *err_off,
				 uint8_t *err_expected, uint8_t *err_actual);

#ifdef __cplusplus
}
#endif

#endif /* ROCM_KERNELS_H */
