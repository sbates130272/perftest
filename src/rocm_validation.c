/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * Snapshot data validation for ROCm device memory.  See rocm_validation.h
 * for why this is a snapshot rather than a port of the host/CUDA validators.
 *
 * Buffer layout is the one compute_validation_layout() builds and
 * host_validation.c consumes:
 *
 *   [3 pattern buffers] [markers] [recv_slots] [atomic_returns]
 *
 * with, per QP q and chunk c:
 *
 *   marker index  = q * chunks_per_qp + c           (uint64_t each)
 *   slot base     = recv_slots
 *                   + (q * ops_per_chunk * chunks_per_qp
 *                      + c * ops_per_chunk) * payload_size
 *   chunk bytes   = ops_per_chunk * payload_size
 *   expected byte = validation_cycle_to_pattern(marker - 1)
 *
 * The addressing is duplicated from host_validation.c rather than shared
 * because that file keeps it in a file-scope singleton tied to its thread
 * pool; lifting it out is a bigger change than this needs.  If either side
 * moves, they must move together.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * rocm_memory.h pulls in config.h, which is where configure's
 * AC_DEFINE([__HIP_PLATFORM_AMD__]) lands.  The HIP headers #error out
 * without it, so this include must come first -- same ordering as
 * rocm_memory.c.
 */
#include "rocm_memory.h"
#include "rocm_validation.h"
#include "cuda_validation.h"   /* VALIDATION_OK and the shared structures */
#include <hip/hip_runtime_api.h>

#ifdef HAVE_ROCM_KERNELS
#include "rocm_kernels.h"
#include "rocm_kernels_persistent.h"
#endif

#define RVLOG(fmt, ...) \
	do { if (g_rv.debug) \
		printf(VALIDATION_LOG_PREFIX " rocm: " fmt "\n", ##__VA_ARGS__); \
	} while (0)

static struct {
	int initialized;
	int debug;
	void *device_base;      /* start of the validation region in device memory */
	uint8_t *host_copy;     /* staging buffer, device_span bytes */
	uint64_t device_span;   /* bytes copied back at stop */
	uint64_t markers_offset;
	uint64_t recv_slots_offset;
	uint64_t payload_size;
	uint32_t ops_per_chunk;
	uint32_t num_qps;
	uint32_t chunks_per_qp;
	int validation_mode;
	int device_id;
#ifdef HAVE_ROCM_KERNELS
	void *persist;          /* non-NULL while the live kernel is running */
#endif
} g_rv;

int rocm_validation_init(struct memory_ctx *ctx,
			 const struct validation_config *cfg)
{
	(void)ctx;

	if (!cfg || !cfg->buffer_base)
		return -1;
	if (cfg->payload_size == 0 || cfg->ops_per_chunk == 0 ||
	    cfg->num_qps == 0 || cfg->chunks_per_qp == 0)
		return -1;

	memset(&g_rv, 0, sizeof(g_rv));
	g_rv.debug             = cfg->debug_enabled;
	g_rv.device_base       = cfg->buffer_base;
	g_rv.markers_offset    = cfg->markers_offset;
	g_rv.recv_slots_offset = cfg->recv_slots_offset;
	g_rv.payload_size      = cfg->payload_size;
	g_rv.ops_per_chunk     = cfg->ops_per_chunk;
	g_rv.num_qps           = cfg->num_qps;
	g_rv.chunks_per_qp     = cfg->chunks_per_qp;
	g_rv.validation_mode   = cfg->validation_mode;
	g_rv.device_id         = 0;   /* set from the memory ctx below */

	/*
	 * Everything through the end of recv_slots.  atomic_returns sits
	 * beyond it and is not checked here, so it is deliberately excluded
	 * rather than guessed at -- cfg carries no is_read_mode.
	 */
	uint64_t slots_bytes;
	if (validation_safe_mul_u64(cfg->payload_size, cfg->ops_per_chunk,
				    &slots_bytes) != 0)
		return -1;
	if (validation_safe_mul_u64(slots_bytes, cfg->chunks_per_qp,
				    &slots_bytes) != 0)
		return -1;
	if (validation_safe_mul_u64(slots_bytes, cfg->num_qps,
				    &slots_bytes) != 0)
		return -1;
	g_rv.device_span = cfg->recv_slots_offset + slots_bytes;

	g_rv.host_copy = malloc(g_rv.device_span);
	if (!g_rv.host_copy) {
		fprintf(stderr, VALIDATION_LOG_PREFIX
			" rocm: cannot allocate %llu-byte staging buffer\n",
			(unsigned long long)g_rv.device_span);
		return -1;
	}

	g_rv.initialized = 1;
	RVLOG("init: base=%p span=%llu markers@%llu slots@%llu payload=%llu "
	      "ops_per_chunk=%u qps=%u chunks_per_qp=%u",
	      g_rv.device_base, (unsigned long long)g_rv.device_span,
	      (unsigned long long)g_rv.markers_offset,
	      (unsigned long long)g_rv.recv_slots_offset,
	      (unsigned long long)g_rv.payload_size,
	      g_rv.ops_per_chunk, g_rv.num_qps, g_rv.chunks_per_qp);
	return 0;
}

int rocm_validation_start(struct memory_ctx *ctx)
{
	(void)ctx;

	if (!g_rv.initialized)
		return -1;

#ifdef HAVE_ROCM_KERNELS
	/*
	 * Live validation by default, driven exactly as cuda_memory.c drives
	 * the CUDA plugin: init the context, then launch 1 monitor block plus
	 * N validator blocks. PERFTEST_ROCM_VALIDATION selects "kernel" (one
	 * GPU compare at stop) or "staged" (one host compare at stop) instead.
	 */
	const char *mode = getenv("PERFTEST_ROCM_VALIDATION");

	if ((!mode || !strcmp(mode, "persistent")) &&
	    !getenv("PERFTEST_ROCM_VALIDATION_STAGED") &&
	    rocm_kernels_available()) {
		int rc = rocm_kernel_validation_init(g_rv.device_base,
						     g_rv.markers_offset,
						     g_rv.recv_slots_offset,
						     g_rv.payload_size,
						     g_rv.ops_per_chunk,
						     g_rv.num_qps,
						     g_rv.chunks_per_qp,
						     g_rv.validation_mode,
						     g_rv.device_id,
						     g_rv.debug);
		if (rc == VALIDATION_OK) {
			/*
			 * Defaults (0) reproduce the CUDA geometry. A device
			 * that cannot retire 33 blocks x 1024 spinning
			 * threads -- a functional simulator, say -- can ask
			 * for a smaller launch without changing the kernel.
			 */
			const char *b = getenv("PERFTEST_ROCM_VALIDATION_BLOCKS");
			const char *t = getenv("PERFTEST_ROCM_VALIDATION_THREADS");

			rc = rocm_kernel_validation_start(NULL,
							  b ? atoi(b) : 0,
							  t ? atoi(t) : 0);
			if (rc == VALIDATION_OK) {
				g_rv.persist = (void *)1;
				printf(VALIDATION_LOG_PREFIX
				       " rocm: live GPU validation\n");
				return 0;
			}
			rocm_kernel_validation_destroy();
		}
		fprintf(stderr, VALIDATION_LOG_PREFIX
			" rocm: live validation unavailable (%s), falling back\n",
			rocm_kernel_validation_strerror(rc));
		g_rv.persist = NULL;
	}
#endif
	return 0;
}

static int rocm_validation_stop_staged(struct data_validation_result *result)
{

	if (!g_rv.initialized || !result)
		return -1;

	memset(result, 0, sizeof(*result));

	hipError_t err = hipMemcpy(g_rv.host_copy, g_rv.device_base,
				   g_rv.device_span, hipMemcpyDeviceToHost);
	if (err != hipSuccess) {
		fprintf(stderr, VALIDATION_LOG_PREFIX
			" rocm: hipMemcpy(DeviceToHost) failed: %d\n", err);
		return -1;
	}

	/*
	 * Fault injection, for proving the checker can fail. A validator that
	 * only ever passes is indistinguishable from one that does not look.
	 * PERFTEST_ROCM_VALIDATION_FAULT=<n> flips n bytes in the staged copy
	 * -- never in device memory -- before the comparison runs.
	 */
	const char *fault = getenv("PERFTEST_ROCM_VALIDATION_FAULT");
	if (fault) {
		uint64_t n = strtoull(fault, NULL, 0);
		for (uint64_t i = 0; i < n && i < g_rv.device_span; i++) {
			uint64_t off = g_rv.recv_slots_offset + i * 4096;
			if (off >= g_rv.device_span)
				break;
			g_rv.host_copy[off] ^= 0xFF;
		}
		RVLOG("fault injection: flipped %llu byte(s)",
		      (unsigned long long)n);
	}

	const uint64_t *markers =
		(const uint64_t *)(g_rv.host_copy + g_rv.markers_offset);
	const uint8_t *slots = g_rv.host_copy + g_rv.recv_slots_offset;
	const uint64_t chunk_bytes =
		(uint64_t)g_rv.ops_per_chunk * g_rv.payload_size;
	const uint64_t slots_per_qp =
		(uint64_t)g_rv.ops_per_chunk * g_rv.chunks_per_qp;

	uint64_t checked_bytes = 0, checked_chunks = 0, errors = 0;
	uint64_t races = 0;
	int first_error_recorded = 0;

	for (uint32_t q = 0; q < g_rv.num_qps; q++) {
		for (uint32_t c = 0; c < g_rv.chunks_per_qp; c++) {
			uint64_t marker = markers[q * g_rv.chunks_per_qp + c];

			/*
			 * marker == 0 means the chunk never completed a cycle.
			 * Skipping is not a pass: if no chunk completed, the
			 * "checked_chunks == 0" test below fails the run.
			 */
			if (marker == 0)
				continue;

			uint8_t want = validation_cycle_to_pattern(marker - 1);
			const uint8_t *base = slots +
				(q * slots_per_qp + (uint64_t)c * g_rv.ops_per_chunk) *
				g_rv.payload_size;

			/*
			 * First mismatch only, then classify -- the same
			 * shape the host and CUDA backends use. Checking every
			 * byte independently is wrong here: a run that stops
			 * mid-cycle leaves the head of a chunk on the new
			 * pattern and the tail on the previous one, which is
			 * an overwrite race, not corruption. The shared
			 * classifier's epoch guard is what tells them apart.
			 */
			uint64_t off = 0;
			int mismatch = 0;
			for (; off < chunk_bytes; off++) {
				if (base[off] != want) {
					mismatch = 1;
					break;
				}
			}

			if (mismatch) {
				uint8_t fwd[4] = {0, 0, 0, 0};
				int fwd_valid = (off + 4 < chunk_bytes);

				if (fwd_valid)
					memcpy(fwd, &base[off + 1], 4);

				/*
				 * retry_matched = 0: this is a snapshot taken
				 * after the run, so there is no second read to
				 * compare against and the STALE branch cannot
				 * apply.
				 */
				enum validation_mismatch_type kind =
					classify_validation_mismatch(off, chunk_bytes,
								     g_rv.payload_size,
								     want, base[off],
								     0, fwd, fwd_valid);

				if (kind == VALIDATION_MISMATCH_REAL) {
					errors++;
					if (!first_error_recorded) {
						result->error_qp_id = q;
						result->error_chunk_id = c;
						result->error_byte_offset =
							(uint64_t)(base - g_rv.host_copy) + off;
						result->error_expected = want;
						result->error_actual = base[off];
						first_error_recorded = 1;
						RVLOG("real mismatch qp=%u chunk=%u off=%llu "
						      "got=0x%02x want=0x%02x",
						      q, c, (unsigned long long)off,
						      base[off], want);
					}
				} else {
					races++;
					RVLOG("overwrite race qp=%u chunk=%u off=%llu",
					      q, c, (unsigned long long)off);
				}
			}

			checked_bytes += mismatch ? off : chunk_bytes;
			checked_chunks++;
		}
	}

	result->race_overwrites = races;
	result->bytes_validated = checked_bytes;
	result->chunks_validated = checked_chunks;
	result->errors_found = errors;
	result->passed = (errors == 0 && checked_chunks > 0) ? 1 : 0;

	if (checked_chunks == 0)
		fprintf(stderr, VALIDATION_LOG_PREFIX
			" rocm: no chunk completed a cycle; nothing validated\n");

	RVLOG("stop: chunks=%llu bytes=%llu errors=%llu passed=%d",
	      (unsigned long long)checked_chunks,
	      (unsigned long long)checked_bytes,
	      (unsigned long long)errors, result->passed);
	return 0;
}

void rocm_validation_destroy(struct memory_ctx *ctx)
{
	(void)ctx;
	free(g_rv.host_copy);
	memset(&g_rv, 0, sizeof(g_rv));
}

#ifdef HAVE_ROCM_KERNELS
/*
 * GPU path: the compare runs in rocm_validate_chunks_kernel and only a
 * per-chunk record comes back. Classification stays on the host so it uses
 * the same classify_validation_mismatch() the host and CUDA backends use.
 */
static int rocm_validation_stop_kernel(struct data_validation_result *result)
{
	const uint32_t total_chunks = g_rv.num_qps * g_rv.chunks_per_qp;
	struct RocmChunkResult *res;
	uint64_t checked_bytes = 0, checked_chunks = 0, errors = 0, races = 0;
	int first_error_recorded = 0;

	res = calloc(total_chunks, sizeof(*res));
	if (!res)
		return -1;

	const char *fault = getenv("PERFTEST_ROCM_VALIDATION_FAULT");
	if (fault) {
		uint64_t n = strtoull(fault, NULL, 0);

		if (n && rocm_kernels_poke(g_rv.device_base,
					   g_rv.recv_slots_offset,
					   (uint32_t)n, 4096) != 0) {
			fprintf(stderr, VALIDATION_LOG_PREFIX
				" rocm: fault injection failed\n");
			free(res);
			return -1;
		}
		RVLOG("fault injection: flipped %llu device byte(s)",
		      (unsigned long long)n);
	}

	if (rocm_kernels_validate(g_rv.device_base, g_rv.markers_offset,
				  g_rv.recv_slots_offset, g_rv.payload_size,
				  g_rv.ops_per_chunk, g_rv.num_qps,
				  g_rv.chunks_per_qp, res) != 0) {
		free(res);
		return -1;
	}

	for (uint32_t i = 0; i < total_chunks; i++) {
		struct RocmChunkResult *r = &res[i];

		if (r->state == ROCM_CHUNK_EMPTY)
			continue;

		if (r->state == ROCM_CHUNK_CLEAN) {
			checked_bytes += r->chunk_bytes;
			checked_chunks++;
			continue;
		}

		int fwd_valid = (r->first_mismatch + 4 < r->chunk_bytes);
		enum validation_mismatch_type kind =
			classify_validation_mismatch(r->first_mismatch,
						     r->chunk_bytes,
						     g_rv.payload_size,
						     r->expected, r->actual,
						     0, r->fwd, fwd_valid);

		if (kind == VALIDATION_MISMATCH_REAL) {
			errors++;
			if (!first_error_recorded) {
				result->error_qp_id = r->qp_id;
				result->error_chunk_id = r->chunk_id;
				result->error_byte_offset = r->first_mismatch;
				result->error_expected = r->expected;
				result->error_actual = r->actual;
				first_error_recorded = 1;
				RVLOG("real mismatch qp=%u chunk=%u off=%llu "
				      "got=0x%02x want=0x%02x",
				      r->qp_id, r->chunk_id,
				      (unsigned long long)r->first_mismatch,
				      r->actual, r->expected);
			}
		} else {
			races++;
		}

		checked_bytes += r->first_mismatch;
		checked_chunks++;
	}

	free(res);

	result->bytes_validated = checked_bytes;
	result->chunks_validated = checked_chunks;
	result->errors_found = errors;
	result->race_overwrites = races;
	result->passed = (errors == 0 && checked_chunks > 0) ? 1 : 0;

	if (checked_chunks == 0)
		fprintf(stderr, VALIDATION_LOG_PREFIX
			" rocm: no chunk completed a cycle; nothing validated\n");

	RVLOG("stop (gpu): chunks=%llu bytes=%llu errors=%llu races=%llu",
	      (unsigned long long)checked_chunks,
	      (unsigned long long)checked_bytes,
	      (unsigned long long)errors, (unsigned long long)races);
	return 0;
}
#endif /* HAVE_ROCM_KERNELS */

#ifdef HAVE_ROCM_KERNELS
/*
 * Join the live validator and translate its stats into perftest's result
 * struct. The counter set is the CUDA plugin's, so the fields line up one
 * for one with what the CUDA backend reports.
 */
static int rocm_validation_stop_persistent(struct data_validation_result *result)
{
	uint64_t chunks = 0, bytes = 0, errors = 0;
	uint64_t scanned = 0, hit = 0, skipped = 0;
	uint64_t races = 0, retries = 0, qfull = 0, stale = 0;
	int rc;

	rc = rocm_kernel_validation_stop();
	g_rv.persist = NULL;
	if (rc != VALIDATION_OK) {
		fprintf(stderr, VALIDATION_LOG_PREFIX " rocm: stop failed: %s\n",
			rocm_kernel_validation_strerror(rc));
		rocm_kernel_validation_destroy();
		return -1;
	}

	rocm_kernel_validation_get_stats(&chunks, &bytes, &errors, &scanned,
					 &hit, &skipped, &races, &retries,
					 &qfull, &stale);

	result->chunks_validated = chunks;
	result->bytes_validated = bytes;
	result->errors_found = errors;
	result->markers_scanned = scanned;
	result->markers_hit = hit;
	result->skipped_steps = skipped;
	result->race_overwrites = races;
	result->dma_stale_retries = retries;
	result->queue_full_drops = qfull;
	result->stale_work_skips = stale;
	result->passed = (errors == 0 && chunks > 0) ? 1 : 0;

	if (errors)
		rocm_kernel_validation_get_error(&result->error_qp_id,
						 &result->error_chunk_id,
						 &result->error_byte_offset,
						 &result->error_expected,
						 &result->error_actual);

	if (chunks == 0)
		fprintf(stderr, VALIDATION_LOG_PREFIX
			" rocm: live validator saw no completed cycle\n");

	rocm_kernel_validation_destroy();
	return 0;
}

#endif /* HAVE_ROCM_KERNELS */

int rocm_validation_stop(struct memory_ctx *ctx,
			 struct data_validation_result *result)
{
	(void)ctx;

	if (!g_rv.initialized || !result)
		return -1;

	memset(result, 0, sizeof(*result));

#ifdef HAVE_ROCM_KERNELS
	if (g_rv.persist)
		return rocm_validation_stop_persistent(result);

	/*
	 * No live kernel: compare once, on the GPU.
	 * PERFTEST_ROCM_VALIDATION_STAGED=1 forces the host-staged path, which
	 * is the reference implementation and the fallback for a device that
	 * cannot run the kernel at all.
	 */
	if (!getenv("PERFTEST_ROCM_VALIDATION_STAGED") &&
	    rocm_kernels_available()) {
		printf(VALIDATION_LOG_PREFIX " rocm: GPU kernel validation\n");
		return rocm_validation_stop_kernel(result);
	}
#endif
	printf(VALIDATION_LOG_PREFIX " rocm: host-staged validation\n");
	return rocm_validation_stop_staged(result);
}
