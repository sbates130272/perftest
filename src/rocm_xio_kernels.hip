/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause
 *
 * GPU-initiated RDMA kernels for perftest.
 * Writes WQEs, rings doorbells, and polls CQ
 * completions matching the PVRDMA ABI used by
 * the rocm-ernic emulated device.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string.h>
#include <cstdio>
#include <unistd.h>

#define ERNIC_WR_RDMA_WRITE 0
#define ERNIC_WR_RDMA_READ  4
#define ERNIC_SEND_SIGNALED (1u << 1)
#define ERNIC_UAR_QP_SEND   (1u << 30)

#define PCI_MMIO_BRIDGE_CMD_WRITE     1
#define PCI_MMIO_BRIDGE_STATUS_PENDING 0

/*
 * pci-mmio-bridge structures matching QEMU device
 * and rocm-xio xio.h.  Inlined here so the GPU
 * linker doesn't need librocm-xio.a device code.
 */
struct pci_mmio_bridge_ring_meta {
	uint32_t producer_idx;
	uint32_t consumer_idx;
	uint32_t queue_depth;
	uint32_t reserved;
} __attribute__((packed));

struct pci_mmio_bridge_command {
	uint16_t target_bdf;
	uint8_t target_bar;
	uint8_t reserved1;
	uint32_t offset;
	uint64_t value;
	uint8_t command;
	uint8_t size;
	uint8_t status;
	uint8_t reserved2;
	uint32_t sequence;
} __attribute__((packed));

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
 * Inline bridge command -- matches
 * xio::genPciMmioBridgeCmd exactly.
 */
__device__ static void gen_bridge_cmd(
	void *shadow, uint16_t target_bdf,
	uint8_t target_bar, uint32_t offset,
	uint64_t value, uint8_t command,
	uint8_t size)
{
	volatile struct pci_mmio_bridge_ring_meta *meta =
		(volatile struct pci_mmio_bridge_ring_meta *)shadow;
	uint32_t slot = meta->producer_idx % meta->queue_depth;
	volatile struct pci_mmio_bridge_command *cmd =
		(volatile struct pci_mmio_bridge_command *)(
			(volatile uint8_t *)shadow +
			sizeof(struct pci_mmio_bridge_ring_meta) +
			slot * sizeof(struct pci_mmio_bridge_command));
	cmd->target_bdf = target_bdf;
	cmd->target_bar = target_bar;
	cmd->reserved1 = 0;
	cmd->offset = offset;
	cmd->value = value;
	cmd->command = command;
	cmd->size = size;
	cmd->status = PCI_MMIO_BRIDGE_STATUS_PENDING;
	cmd->reserved2 = 0;
	cmd->sequence = slot;
	__threadfence_system();
	meta->producer_idx = meta->producer_idx + 1;
	__threadfence_system();
}

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

__device__ static void ring_doorbell(
	volatile int32_t *sq_ring_prod_tail,
	volatile uint32_t *uar_ptr,
	uint32_t uar_qp_offset,
	uint32_t qp_handle,
	uint32_t sq_depth,
	void *mmio_bridge_shadow,
	uint16_t ernic_target_bdf,
	uint32_t uar_bar_offset)
{
	if (sq_ring_prod_tail) {
		int32_t cur = *sq_ring_prod_tail;
		int32_t mask = (int32_t)((sq_depth << 1) - 1);
		int32_t next = (cur + 1) & mask;
		*sq_ring_prod_tail = next;
	}

	uint32_t val = (qp_handle & 0x00FFFFFF) | ERNIC_UAR_QP_SEND;
	__threadfence_system();

	if (mmio_bridge_shadow) {
		gen_bridge_cmd(mmio_bridge_shadow,
			       ernic_target_bdf, 2,
			       uar_bar_offset, val,
			       PCI_MMIO_BRIDGE_CMD_WRITE, 4);
	} else if (uar_ptr) {
		volatile uint32_t *db = uar_ptr + (uar_qp_offset / 4);
		__hip_atomic_store(
			const_cast<uint32_t *>(db), val,
			__ATOMIC_SEQ_CST,
			__HIP_MEMORY_SCOPE_SYSTEM);
	}
}

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

__global__ void rocm_xio_write_bw_gpu_kernel(
	void *sq_base, void *cq_base,
	volatile int32_t *sq_ring_prod_tail,
	volatile uint32_t *gpu_uar_ptr,
	uint32_t uar_qp_offset,
	uint64_t remote_addr, uint32_t rkey,
	uint32_t lkey, void *local_buf,
	uint32_t msg_size, uint64_t iterations,
	uint32_t sq_depth, uint32_t sq_wqe_size,
	uint32_t qp_handle, uint32_t cq_depth,
	void *mmio_bridge_shadow,
	uint16_t ernic_target_bdf,
	uint32_t uar_bar_offset)
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
			      gpu_uar_ptr, uar_qp_offset,
			      qp_handle, sq_depth,
			      mmio_bridge_shadow,
			      ernic_target_bdf,
			      uar_bar_offset);
	}

	poll_cq_until(cq_base, cq_depth,
		      sq_depth, &sq_head, &sq_tail,
		      sq_depth);
}

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
	void *mmio_bridge_shadow,
	uint16_t ernic_target_bdf,
	uint32_t uar_bar_offset,
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
			      gpu_uar_ptr, uar_qp_offset,
			      qp_handle, sq_depth,
			      mmio_bridge_shadow,
			      ernic_target_bdf,
			      uar_bar_offset);

		poll_cq_until(cq_base, cq_depth,
			      sq_depth, &sq_head, &sq_tail,
			      sq_depth);

		uint64_t t1 = clock64();
		gpu_timestamps[i] = t1 - t0;
	}
}

/*
 * Host-side functions.  xio.h is only needed here
 * for the setup API (not device code).
 */
#include "xio.h"

extern "C" {

int rocm_xio_setup_mmio_bridge(
	const char *rdma_dev_name,
	uint64_t uar_mmap_offset,
	uint32_t uar_qp_offset,
	void **shadow_gpu_out,
	uint16_t *ernic_bdf_out,
	uint32_t *uar_bar_offset_out)
{
	*shadow_gpu_out = NULL;
	*ernic_bdf_out = 0;
	*uar_bar_offset_out = 0;

	uint16_t bridge_bdf = 0;
	int ret = xio::detectPciMmioBridgeBdf(&bridge_bdf);
	if (ret != 0) {
		fprintf(stderr,
			"rocm-xio: detectPciMmioBridgeBdf "
			"failed: %d\n", ret);
		return -1;
	}

	void *shadow_cpu = NULL;
	ret = xio::mapMmioBridgeShadowBuffer(bridge_bdf,
					     &shadow_cpu);
	if (ret != 0 || !shadow_cpu) {
		fprintf(stderr,
			"rocm-xio: mapMmioBridgeShadowBuffer "
			"failed: %d\n", ret);
		return -1;
	}

	const size_t shadow_size = 8192;
	void *shadow_gpu = NULL;
	ret = xio::registerMmioBridgeShadowBufferForGpu(
		shadow_cpu, shadow_size, &shadow_gpu);
	if (ret != 0 || !shadow_gpu) {
		fprintf(stderr,
			"rocm-xio: "
			"registerMmioBridgeShadowBufferForGpu "
			"failed: %d\n", ret);
		return -1;
	}

	char res_path[256];
	snprintf(res_path, sizeof(res_path),
		 "/sys/class/infiniband/%s/device/resource",
		 rdma_dev_name);
	FILE *rf = fopen(res_path, "r");
	unsigned long bar2_base = 0;
	if (rf) {
		unsigned long s, e, f;
		if (fscanf(rf, "%lx %lx %lx", &s, &e, &f) == 3 &&
		    fscanf(rf, "%lx %lx %lx", &s, &e, &f) == 3 &&
		    fscanf(rf, "%lx %lx %lx", &bar2_base, &e, &f) == 3) {
		}
		fclose(rf);
	}

	uint32_t bar_offset = 0;
	if (bar2_base && uar_mmap_offset >= bar2_base) {
		uint64_t off64 = (uar_mmap_offset - bar2_base) +
				 uar_qp_offset;
		if (off64 <= UINT32_MAX)
			bar_offset = (uint32_t)off64;
		else
			fprintf(stderr,
				"rocm-xio: UAR BAR2 offset "
				"0x%lx exceeds 32-bit\n",
				(unsigned long)off64);
	}

	char sysfs_link[256];
	snprintf(sysfs_link, sizeof(sysfs_link),
		 "/sys/class/infiniband/%s/device",
		 rdma_dev_name);
	char link_buf[256];
	ssize_t llen = readlink(sysfs_link, link_buf,
				sizeof(link_buf) - 1);
	uint16_t ernic_bdf = 0;
	if (llen > 0) {
		link_buf[llen] = '\0';
		const char *p = strrchr(link_buf, '/');
		if (p)
			p++;
		else
			p = link_buf;
		unsigned b = 0, d = 0, f = 0;
		if (sscanf(p, "%*x:%x:%x.%x", &b, &d, &f) == 3)
			ernic_bdf = (uint16_t)((b << 8) |
					       (d << 3) | f);
	}

	if (ernic_bdf == 0 || bar_offset == 0) {
		fprintf(stderr,
			"rocm-xio: bridge cannot enable: "
			"bdf=0x%04x, bar_offset=0x%x\n",
			ernic_bdf, bar_offset);
		return -1;
	}

	*shadow_gpu_out = shadow_gpu;
	*ernic_bdf_out = ernic_bdf;
	*uar_bar_offset_out = bar_offset;

	fprintf(stderr,
		"rocm-xio: pci-mmio-bridge enabled "
		"(bridge=0x%04x, ernic=0x%04x, "
		"bar_offset=0x%x, shadow=%p)\n",
		bridge_bdf, ernic_bdf,
		bar_offset, shadow_gpu);

	return 0;
}

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
	void *mmio_bridge_shadow,
	uint16_t ernic_target_bdf,
	uint32_t uar_bar_offset,
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
		qp_handle, cq_depth,
		mmio_bridge_shadow,
		ernic_target_bdf,
		uar_bar_offset);

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
	void *mmio_bridge_shadow,
	uint16_t ernic_target_bdf,
	uint32_t uar_bar_offset,
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
		mmio_bridge_shadow,
		ernic_target_bdf,
		uar_bar_offset,
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
