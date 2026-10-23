/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * HIP validation kernels for perftest ROCm memory.
 *
 * The comparison runs on the GPU, as it does for CUDA (src/cuda_kernels.cu):
 * the received payload is never copied to the host to be checked. Only a
 * per-chunk result record comes back, which is a few hundred bytes however
 * large the ring is.
 *
 * This is the at-stop tier of the ROCm validator, not its default. The
 * default is rocm_kernels_persistent.hip, which mirrors cuda_kernels.cu: a
 * persistent kernel runs for the duration of the transfer, a monitor block
 * scanning tail markers and pushing work into a lock-free queue that
 * validator blocks drain, so each cycle is checked while the next is still
 * arriving. That design needs a GPU that can hold a spinning kernel resident
 * for a whole run at no cost to the transfer, and needs to preempt waves so
 * the poll loops retire. Where that does not hold -- a functional simulator,
 * or any device where a resident poller competes with the DMA it is watching
 * -- PERFTEST_ROCM_VALIDATION=kernel selects this file instead: one launch at
 * stop, one block per chunk, every chunk compared in parallel on the device.
 *
 * What that costs: only the final state of each chunk is seen, so transient
 * corruption that a later cycle overwrites is not caught, and
 * bytes_validated is correspondingly lower. What it buys: no interference
 * with the measurement, and no large device->host copy per run.
 *
 * PERFTEST_ROCM_VALIDATION=staged is the third tier, comparing on the host,
 * and is what a build without hipcc falls back to.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "rocm_kernels.h"

/*
 * One block per chunk, blockDim threads striding the chunk.
 *
 * The first mismatching byte is reduced with atomicMin rather than taken by
 * whichever thread happens to report first: the classifier downstream keys on
 * the *offset* of the first mismatch to tell an overwrite race from real
 * corruption, so an arbitrary mismatch would misclassify.
 */
__global__ void rocm_validate_chunks_kernel(const uint8_t *buf,
					    uint64_t markers_offset,
					    uint64_t slots_offset,
					    uint64_t payload_size,
					    uint32_t ops_per_chunk,
					    uint32_t chunks_per_qp,
					    struct RocmChunkResult *out)
{
	const uint32_t chunk_idx = blockIdx.x;
	const uint32_t q = chunk_idx / chunks_per_qp;
	const uint32_t c = chunk_idx % chunks_per_qp;

	const uint64_t *markers = (const uint64_t *)(buf + markers_offset);
	const uint64_t marker = markers[chunk_idx];

	__shared__ unsigned long long s_first;

	if (threadIdx.x == 0)
		s_first = ~0ULL;
	__syncthreads();

	const uint64_t chunk_bytes = (uint64_t)ops_per_chunk * payload_size;

	if (marker != 0) {
		/* Patterns are consecutive 0x01,0x02,0x03 -- see
		 * validation_cycle_to_pattern() in validation_common.h. Kept
		 * as arithmetic here so the kernel needs no host header. */
		const uint8_t want =
			(uint8_t)(((marker - 1) % 3) + 1);

		const uint64_t slots_per_qp =
			(uint64_t)ops_per_chunk * chunks_per_qp;
		const uint8_t *base = buf + slots_offset +
			((uint64_t)q * slots_per_qp + (uint64_t)c * ops_per_chunk) *
			payload_size;

		for (uint64_t b = threadIdx.x; b < chunk_bytes; b += blockDim.x) {
			if (base[b] != want)
				atomicMin(&s_first, (unsigned long long)b);
		}
	}

	__syncthreads();

	if (threadIdx.x != 0)
		return;

	struct RocmChunkResult *r = &out[chunk_idx];
	r->marker = marker;
	r->chunk_bytes = chunk_bytes;
	r->qp_id = q;
	r->chunk_id = c;

	if (marker == 0) {
		r->state = ROCM_CHUNK_EMPTY;
		r->first_mismatch = 0;
		r->expected = 0;
		r->actual = 0;
		for (int i = 0; i < 4; i++)
			r->fwd[i] = 0;
		return;
	}

	const uint8_t want = (uint8_t)(((marker - 1) % 3) + 1);

	if (s_first == ~0ULL) {
		r->state = ROCM_CHUNK_CLEAN;
		r->first_mismatch = 0;
		r->expected = want;
		r->actual = want;
		for (int i = 0; i < 4; i++)
			r->fwd[i] = 0;
		return;
	}

	const uint64_t slots_per_qp = (uint64_t)ops_per_chunk * chunks_per_qp;
	const uint8_t *base = buf + slots_offset +
		((uint64_t)q * slots_per_qp + (uint64_t)c * ops_per_chunk) *
		payload_size;

	const uint64_t off = (uint64_t)s_first;

	r->state = ROCM_CHUNK_MISMATCH;
	r->first_mismatch = off;
	r->expected = want;
	r->actual = base[off];
	r->retry_matched = 0;   /* snapshot: nothing is still writing */

	/*
	 * Four bytes past the mismatch, for the host-side epoch guard. Read
	 * here rather than fetched later so the classifier sees the same
	 * device state the compare did.
	 */
	for (int i = 0; i < 4; i++)
		r->fwd[i] = (off + 1 + i < chunk_bytes) ? base[off + 1 + i] : 0;
}

extern "C" int rocm_kernels_validate(const void *device_buffer,
				     uint64_t markers_offset,
				     uint64_t slots_offset,
				     uint64_t payload_size,
				     uint32_t ops_per_chunk,
				     uint32_t num_qps,
				     uint32_t chunks_per_qp,
				     struct RocmChunkResult *host_results)
{
	const uint32_t total_chunks = num_qps * chunks_per_qp;
	struct RocmChunkResult *d_results = NULL;
	const size_t bytes = (size_t)total_chunks * sizeof(*d_results);
	hipError_t err;

	if (!device_buffer || !host_results || total_chunks == 0)
		return -1;

	err = hipMalloc((void **)&d_results, bytes);
	if (err != hipSuccess) {
		fprintf(stderr, "rocm_kernels: hipMalloc(%zu) failed: %s\n",
			bytes, hipGetErrorString(err));
		return -1;
	}

	err = hipMemset(d_results, 0, bytes);
	if (err != hipSuccess) {
		fprintf(stderr, "rocm_kernels: hipMemset failed: %s\n",
			hipGetErrorString(err));
		hipFree(d_results);
		return -1;
	}

	/*
	 * 256 threads is a compromise, not a tuned number: wide enough that a
	 * real GPU covers a multi-MiB chunk in few strides, narrow enough not
	 * to ask a weaker device to schedule a full occupancy wave for what is
	 * a memcmp.
	 */
	hipLaunchKernelGGL(rocm_validate_chunks_kernel,
			   dim3(total_chunks), dim3(256), 0, 0,
			   (const uint8_t *)device_buffer,
			   markers_offset, slots_offset, payload_size,
			   ops_per_chunk, chunks_per_qp, d_results);

	err = hipGetLastError();
	if (err != hipSuccess) {
		fprintf(stderr, "rocm_kernels: launch failed: %s\n",
			hipGetErrorString(err));
		hipFree(d_results);
		return -1;
	}

	err = hipDeviceSynchronize();
	if (err != hipSuccess) {
		fprintf(stderr, "rocm_kernels: synchronize failed: %s\n",
			hipGetErrorString(err));
		hipFree(d_results);
		return -1;
	}

	err = hipMemcpy(host_results, d_results, bytes, hipMemcpyDeviceToHost);
	hipFree(d_results);

	if (err != hipSuccess) {
		fprintf(stderr, "rocm_kernels: result copy failed: %s\n",
			hipGetErrorString(err));
		return -1;
	}

	return 0;
}

/*
 * Flip bytes in *device* memory, for proving the kernel validator can fail.
 * The staged validator can corrupt its host copy; this path has no host copy
 * to corrupt, so the poke has to reach the device. Only ever called after a
 * run has finished.
 */
extern "C" int rocm_kernels_poke(void *device_buffer, uint64_t offset,
				 uint32_t count, uint64_t stride)
{
	uint8_t *p = (uint8_t *)device_buffer + offset;

	for (uint32_t i = 0; i < count; i++) {
		uint8_t v;

		if (hipMemcpy(&v, p + (uint64_t)i * stride, 1,
			      hipMemcpyDeviceToHost) != hipSuccess)
			return -1;
		v ^= 0xFF;
		if (hipMemcpy(p + (uint64_t)i * stride, &v, 1,
			      hipMemcpyHostToDevice) != hipSuccess)
			return -1;
	}
	return 0;
}

extern "C" int rocm_kernels_available(void)
{
	int count = 0;

	return (hipGetDeviceCount(&count) == hipSuccess && count > 0) ? 1 : 0;
}
