/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * C interface to the HIP validation kernels in rocm_kernels.hip.
 *
 * Deliberately free of HIP types so rocm_validation.c, an ordinary C file
 * compiled by $(CC), can call it. The .hip side is compiled by hipcc and
 * linked in; see the ROCM_KERNELS block in Makefile.am.
 */

#ifndef ROCM_KERNELS_H
#define ROCM_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum RocmChunkState {
	ROCM_CHUNK_EMPTY    = 0,  /* marker == 0: no cycle ever completed */
	ROCM_CHUNK_CLEAN    = 1,  /* every byte matched the marker's pattern */
	ROCM_CHUNK_MISMATCH = 2   /* first mismatch recorded below */
};

/*
 * One per chunk, filled on the device and copied back. Everything the host
 * needs to classify a mismatch is here, so the payload itself never leaves
 * device memory.
 */
struct RocmChunkResult {
	uint64_t marker;         /* tail marker value the compare used */
	uint64_t chunk_bytes;    /* ops_per_chunk * payload_size */
	uint64_t first_mismatch; /* byte offset within the chunk */
	uint32_t qp_id;
	uint32_t chunk_id;
	uint32_t state;          /* enum RocmChunkState */
	uint8_t  expected;
	uint8_t  actual;
	uint8_t  fwd[4];         /* bytes [first+1 .. first+4], for the epoch guard */
	uint8_t  retry_matched;  /* re-read of the mismatched byte now matches */
	uint8_t  pad;
};

/*
 * Validate every chunk on the device. host_results must have room for
 * num_qps * chunks_per_qp entries. Returns 0 on success, -1 on a HIP error.
 */
int rocm_kernels_validate(const void *device_buffer,
			  uint64_t markers_offset,
			  uint64_t slots_offset,
			  uint64_t payload_size,
			  uint32_t ops_per_chunk,
			  uint32_t num_qps,
			  uint32_t chunks_per_qp,
			  struct RocmChunkResult *host_results);

/*
 * Fault injection: XOR 0xFF into `count` device bytes starting at `offset`,
 * `stride` apart. For tests that need to prove the validator reports failure.
 */
int rocm_kernels_poke(void *device_buffer, uint64_t offset,
		      uint32_t count, uint64_t stride);

/* 1 if a HIP device is usable, 0 otherwise. */
int rocm_kernels_available(void);

#ifdef __cplusplus
}
#endif

#endif /* ROCM_KERNELS_H */
