/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause
 *
 * GPU-initiated RDMA kernels for perftest.
 * Writes WQEs, rings doorbells, and polls CQ
 * completions matching the PVRDMA ABI used by
 * the rocm-ernic emulated device.
 *
 * WQE layout and doorbell encoding match
 * rocm-xio's ernic-queue-pair.hip exactly.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string.h>

/* PVRDMA WQE opcodes */
#define ERNIC_WR_RDMA_WRITE 0
#define ERNIC_WR_RDMA_READ  4

/* Send flags */
#define ERNIC_SEND_SIGNALED (1u << 1)

/* UAR doorbell encoding */
#define ERNIC_UAR_QP_SEND (1u << 30)

/*
 * WQE header: 80 bytes matching pvrdma_sq_wqe_hdr.
 * Must be kept in sync with the rocm-ernic provider
 * and emulator.
 */
struct ernic_wqe_hdr {
	uint64_t wr_id;
	uint32_t num_sge;
	uint32_t total_len;
	uint32_t opcode;
	uint32_t send_flags;
	union {
		uint32_t imm_data;
		uint32_t invalidate_rkey;
	} ex;
	uint32_t reserved;
	union {
		struct {
			uint64_t remote_addr;
			uint32_t rkey;
			uint8_t pad[4];
		} rdma;
		uint8_t _pad[48];
	} wr;
};
static_assert(sizeof(ernic_wqe_hdr) == 80,
	      "ernic_wqe_hdr must be 80 bytes");

struct ernic_sge {
	uint64_t addr;
	uint32_t length;
	uint32_t lkey;
};

/*
 * Write one RDMA Write WQE into the SQ at the given
 * tail index, then advance the tail.
 */
__device__ static void write_rdma_wqe(
	void *sq_base, uint32_t sq_wqe_size,
	uint32_t *tail, uint32_t sq_depth,
	uint64_t remote_addr, uint32_t rkey,
	uint32_t lkey, uint64_t local_addr,
	uint32_t msg_size)
{
	char *wqe_ptr = (char *)sq_base + (*tail * sq_wqe_size);

	struct ernic_wqe_hdr hdr;
	memset(&hdr, 0, sizeof(hdr));
	hdr.wr_id = 0;
	hdr.num_sge = 1;
	hdr.total_len = msg_size;
	hdr.opcode = ERNIC_WR_RDMA_WRITE;
	hdr.send_flags = ERNIC_SEND_SIGNALED;
	hdr.wr.rdma.remote_addr = remote_addr;
	hdr.wr.rdma.rkey = rkey;

	struct ernic_sge sge;
	sge.addr = local_addr;
	sge.length = msg_size;
	sge.lkey = lkey;

	memcpy(wqe_ptr, &hdr, sizeof(hdr));
	memcpy(wqe_ptr + sizeof(hdr), &sge, sizeof(sge));

	*tail = (*tail + 1) % sq_depth;
}

/*
 * Ring the UAR doorbell via direct MMIO write.
 * The pci-mmio-bridge path is not implemented here;
 * it requires the bridge shadow buffer and BDF which
 * would add significant complexity.  For Phase 1 we
 * use the direct UAR path (which works when the GPU
 * can reach the UAR page via hipHostRegister).
 */
__device__ static void ring_doorbell(
	volatile int32_t *sq_ring_prod_tail,
	volatile uint32_t *uar_ptr,
	uint32_t uar_qp_offset,
	uint32_t qp_handle,
	uint32_t sq_depth)
{
	if (sq_ring_prod_tail) {
		int32_t cur = *sq_ring_prod_tail;
		int32_t mask = (int32_t)((sq_depth << 1) - 1);
		int32_t next = (cur + 1) & mask;
		*sq_ring_prod_tail = next;
	}

	uint32_t val = (qp_handle & 0x00FFFFFF) | ERNIC_UAR_QP_SEND;
	__threadfence_system();

	if (uar_ptr) {
		volatile uint32_t *db = uar_ptr + (uar_qp_offset / 4);
		__hip_atomic_store(
			const_cast<uint32_t *>(db), val,
			__ATOMIC_SEQ_CST,
			__HIP_MEMORY_SCOPE_SYSTEM);
	}
}

/*
 * Poll the CQ for a completion.  The CQ ring state
 * lives at the start of the CQ buffer as PvrdmaRingState:
 *   [0] = SQ prod_tail (unused here)
 *   [1] = SQ cons_head (unused here)
 *   [2] = CQ prod_tail (server-written)
 *   [3] = CQ cons_head (GPU-advanced)
 */
__device__ static void poll_cq_until(
	void *cq_base, uint32_t cq_depth,
	uint32_t sq_depth,
	uint32_t *sq_head, uint32_t *sq_tail,
	uint32_t requested)
{
	volatile int32_t *cq_ring = (volatile int32_t *)cq_base;

	while (true) {
		if (cq_ring) {
			int32_t cq_prod = cq_ring[2];
			int32_t cq_cons = cq_ring[3];
			if (cq_prod != cq_cons) {
				uint32_t cq_mask =
					(cq_depth << 1) - 1;
				int32_t new_cons =
					(cq_cons + 1) & cq_mask;
				cq_ring[3] = new_cons;
				__threadfence_system();
				(*sq_head)++;
				if (*sq_head >= sq_depth)
					*sq_head = 0;
			}
		}

		uint32_t consumed =
			(*sq_tail - *sq_head + sq_depth) %
			sq_depth;
		uint32_t available = sq_depth - consumed;
		if (available >= requested)
			return;
	}
}

/*
 * Bandwidth kernel: post `iterations` RDMA Write
 * WQEs as fast as possible, each followed by a
 * doorbell ring.  Poll for all completions at end.
 */
__global__ void rocm_xio_write_bw_gpu_kernel(
	void *sq_base, void *cq_base,
	volatile int32_t *sq_ring_prod_tail,
	volatile uint32_t *gpu_uar_ptr,
	uint32_t uar_qp_offset,
	uint64_t remote_addr, uint32_t rkey,
	uint32_t lkey, void *local_buf,
	uint32_t msg_size, uint64_t iterations,
	uint32_t sq_depth, uint32_t sq_wqe_size,
	uint32_t qp_handle, uint32_t cq_depth)
{
	uint32_t sq_tail = 0;
	uint32_t sq_head = 0;
	uint64_t local_addr = (uint64_t)local_buf;

	for (uint64_t i = 0; i < iterations; i++) {
		poll_cq_until(cq_base, cq_depth,
			      sq_depth, &sq_head, &sq_tail, 1);

		write_rdma_wqe(sq_base, sq_wqe_size,
			       &sq_tail, sq_depth,
			       remote_addr, rkey,
			       lkey, local_addr,
			       msg_size);

		ring_doorbell(sq_ring_prod_tail,
			      gpu_uar_ptr,
			      uar_qp_offset,
			      qp_handle, sq_depth);
	}

	poll_cq_until(cq_base, cq_depth,
		      sq_depth, &sq_head, &sq_tail,
		      sq_depth);
}

/*
 * Latency kernel: post one WQE, wait for completion,
 * record per-iteration timing.  Runs on a single
 * thread.
 */
__global__ void rocm_xio_write_lat_gpu_kernel(
	void *sq_base, void *cq_base,
	volatile int32_t *sq_ring_prod_tail,
	volatile uint32_t *gpu_uar_ptr,
	uint32_t uar_qp_offset,
	uint64_t remote_addr, uint32_t rkey,
	uint32_t lkey, void *local_buf,
	uint32_t msg_size, uint64_t iterations,
	uint32_t sq_depth, uint32_t sq_wqe_size,
	uint32_t qp_handle, uint32_t cq_depth,
	uint64_t *gpu_timestamps)
{
	uint32_t sq_tail = 0;
	uint32_t sq_head = 0;
	uint64_t local_addr = (uint64_t)local_buf;

	for (uint64_t i = 0; i < iterations; i++) {
		uint64_t t0 = clock64();

		poll_cq_until(cq_base, cq_depth,
			      sq_depth, &sq_head, &sq_tail, 1);

		write_rdma_wqe(sq_base, sq_wqe_size,
			       &sq_tail, sq_depth,
			       remote_addr, rkey,
			       lkey, local_addr,
			       msg_size);

		ring_doorbell(sq_ring_prod_tail,
			      gpu_uar_ptr,
			      uar_qp_offset,
			      qp_handle, sq_depth);

		poll_cq_until(cq_base, cq_depth,
			      sq_depth, &sq_head, &sq_tail,
			      sq_depth);

		uint64_t t1 = clock64();
		gpu_timestamps[i] = t1 - t0;
	}
}

/*
 * Host-side launcher functions called from
 * rocm_xio_memory.c.  These set up HIP events
 * for timing, launch the kernels, and collect
 * results.
 */
extern "C" {

int rocm_xio_launch_write_bw_kernel(
	void *sq_base, void *cq_base,
	volatile int32_t *sq_ring_prod_tail,
	volatile uint32_t *gpu_uar_ptr,
	uint32_t uar_qp_offset,
	uint64_t remote_addr, uint32_t rkey,
	uint32_t lkey, void *local_buf,
	uint32_t msg_size, uint64_t iterations,
	uint32_t sq_depth, uint32_t sq_wqe_size,
	uint32_t qp_handle, uint32_t cq_depth,
	double *elapsed_usec)
{
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);

	hipEventRecord(start, 0);

	rocm_xio_write_bw_gpu_kernel<<<1, 1>>>(
		sq_base, cq_base,
		sq_ring_prod_tail, gpu_uar_ptr,
		uar_qp_offset,
		remote_addr, rkey,
		lkey, local_buf,
		msg_size, iterations,
		sq_depth, sq_wqe_size,
		qp_handle, cq_depth);

	hipEventRecord(stop, 0);
	hipEventSynchronize(stop);

	float ms = 0.0f;
	hipEventElapsedTime(&ms, start, stop);
	*elapsed_usec = (double)ms * 1000.0;

	hipEventDestroy(start);
	hipEventDestroy(stop);

	hipError_t err = hipGetLastError();
	if (err != hipSuccess) {
		fprintf(stderr,
			"rocm-xio: BW kernel error: %s\n",
			hipGetErrorString(err));
		return 1;
	}

	return 0;
}

int rocm_xio_launch_write_lat_kernel(
	void *sq_base, void *cq_base,
	volatile int32_t *sq_ring_prod_tail,
	volatile uint32_t *gpu_uar_ptr,
	uint32_t uar_qp_offset,
	uint64_t remote_addr, uint32_t rkey,
	uint32_t lkey, void *local_buf,
	uint32_t msg_size, uint64_t iterations,
	uint32_t sq_depth, uint32_t sq_wqe_size,
	uint32_t qp_handle, uint32_t cq_depth,
	double *iter_usec)
{
	uint64_t *gpu_ts = NULL;
	hipMalloc(&gpu_ts, iterations * sizeof(uint64_t));
	hipMemset(gpu_ts, 0, iterations * sizeof(uint64_t));

	rocm_xio_write_lat_gpu_kernel<<<1, 1>>>(
		sq_base, cq_base,
		sq_ring_prod_tail, gpu_uar_ptr,
		uar_qp_offset,
		remote_addr, rkey,
		lkey, local_buf,
		msg_size, iterations,
		sq_depth, sq_wqe_size,
		qp_handle, cq_depth,
		gpu_ts);

	hipDeviceSynchronize();

	hipError_t err = hipGetLastError();
	if (err != hipSuccess) {
		fprintf(stderr,
			"rocm-xio: LAT kernel error: %s\n",
			hipGetErrorString(err));
		hipFree(gpu_ts);
		return 1;
	}

	uint64_t *host_ts = (uint64_t *)malloc(
		iterations * sizeof(uint64_t));
	hipMemcpy(host_ts, gpu_ts,
		  iterations * sizeof(uint64_t),
		  hipMemcpyDeviceToHost);

	/*
	 * Convert GPU clock ticks to microseconds.
	 * hipDeviceGetAttribute with
	 * hipDeviceAttributeClockRate returns KHz.
	 */
	int clock_khz = 0;
	hipDeviceGetAttribute(&clock_khz,
			      hipDeviceAttributeClockRate, 0);
	double ticks_per_us = (double)clock_khz / 1000.0;

	for (uint64_t i = 0; i < iterations; i++) {
		if (ticks_per_us > 0.0)
			iter_usec[i] = (double)host_ts[i] /
				       ticks_per_us;
		else
			iter_usec[i] = 0.0;
	}

	free(host_ts);
	hipFree(gpu_ts);
	return 0;
}

} /* extern "C" */
