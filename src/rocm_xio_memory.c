/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause
 *
 * rocm-xio GPU-initiated RDMA memory backend for
 * perftest.  Allocates GPU-accessible buffers and
 * manages DV resources (CQ, QP, UAR) for
 * GPU-initiated WQE posting.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_ROCM_XIO

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <infiniband/verbs.h>

#include "rocm_xio_memory.h"
#include "perftest_parameters.h"
#include "perftest_resources.h"
#include "get_clock.h"

#define ACCEL_PAGE_SIZE (64 * 1024)

#define HIP_CHECK(call)                                         \
	do {                                                    \
		hipError_t _err = (call);                       \
		if (_err != hipSuccess) {                       \
			fprintf(stderr,                         \
				"rocm-xio: HIP error %d "       \
				"(%s) at %s:%d\n",              \
				_err,                           \
				hipGetErrorString(_err),         \
				__FILE__, __LINE__);            \
			return FAILURE;                         \
		}                                               \
	} while (0)

/*
 * DV struct definitions: must match the rocm_ernic
 * provider header exactly.
 */
struct rocm_ernic_dv_umem_attr {
	void *addr;
	size_t size;
	uint32_t access_flags;
	int dmabuf_fd;
	uint64_t comp_mask;
};

struct rocm_ernic_dv_umem {
	struct ibv_context *context;
	void *addr;
	size_t size;
	uint32_t access_flags;
	int dmabuf_fd;
};

struct rocm_ernic_dv_cq_init_attr {
	uint64_t cq_handle;
	void *umem_handle;
	uint32_t ncqe;
	uint64_t comp_mask;
};

struct rocm_ernic_dv_qp_init_attr {
	uint32_t qp_type;
	uint32_t max_send_wr;
	uint32_t max_recv_wr;
	uint32_t max_send_sge;
	uint32_t max_recv_sge;
	uint32_t max_inline_data;
	struct ibv_cq *send_cq;
	struct ibv_cq *recv_cq;
	uint64_t qp_handle;
	void *sq_umem_handle;
	uint32_t sq_len;
	uint32_t sq_wqe_size;
	void *rq_umem_handle;
	uint32_t rq_len;
	uint32_t rq_wqe_size;
	uint64_t comp_mask;
};

struct rocm_ernic_dv_qp_attr {
	uint32_t qpn;
	uint32_t qp_handle;
	uint32_t sq_depth;
	uint32_t rq_depth;
	uint32_t sq_wqe_size;
	uint32_t rq_wqe_size;
	void *uar_ptr;
	uint32_t uar_qp_offset;
	uint32_t uar_cq_offset;
	uint64_t uar_mmap_offset;
};

typedef struct rocm_ernic_dv_umem *(*fn_umem_reg)(
	struct ibv_context *, struct rocm_ernic_dv_umem_attr *);
typedef int (*fn_umem_dereg)(struct rocm_ernic_dv_umem *);
typedef struct ibv_cq *(*fn_create_cq)(
	struct ibv_context *, struct rocm_ernic_dv_cq_init_attr *);
typedef int (*fn_destroy_cq)(struct ibv_cq *);
typedef struct ibv_qp *(*fn_create_qp)(
	struct ibv_pd *, struct rocm_ernic_dv_qp_init_attr *);
typedef int (*fn_destroy_qp)(struct ibv_qp *);
typedef int (*fn_modify_qp)(struct ibv_qp *, struct ibv_qp_attr *, int);
typedef uint32_t (*fn_get_cq_id)(struct ibv_cq *);
typedef int (*fn_get_qp_attr)(struct ibv_qp *,
			      struct rocm_ernic_dv_qp_attr *);

struct dv_funcs {
	fn_umem_reg umem_reg;
	fn_umem_dereg umem_dereg;
	fn_create_cq create_cq;
	fn_destroy_cq destroy_cq;
	fn_create_qp create_qp;
	fn_destroy_qp destroy_qp;
	fn_modify_qp modify_qp;
	fn_get_cq_id get_cq_id;
	fn_get_qp_attr get_qp_attr;
};

struct rocm_xio_dv_state {
	void *dl_handle;
	struct dv_funcs dv;

	void *cq_buf;
	size_t cq_buf_len;
	struct rocm_ernic_dv_umem *cq_umem;
	struct ibv_cq *dv_send_cq;
	struct ibv_cq *dv_recv_cq;
	uint32_t cq_depth;

	void *sq_buf;
	size_t sq_buf_len;
	struct rocm_ernic_dv_umem *sq_umem;
	void *rq_buf;
	size_t rq_buf_len;
	struct rocm_ernic_dv_umem *rq_umem;
	uint32_t sq_depth;
	uint32_t sq_wqe_size;
	uint32_t qp_handle;

	void *uar_ptr;
	uint32_t uar_qp_offset;
	uint32_t uar_cq_offset;
	uint64_t uar_mmap_offset;

	volatile uint32_t *gpu_uar_ptr;
	volatile int32_t *sq_ring_prod_tail;
	void *sq_wqe_base;
	uint32_t uar_bar_offset;

	int pci_mmio_bridge;

	/* pci-mmio-bridge state */
	void *mmio_bridge_shadow;
	uint16_t ernic_target_bdf;

	/* RDMA device name for sysfs lookups */
	char rdma_dev_name[64];
};

struct rocm_xio_memory_ctx {
	struct memory_ctx base;
	int device_id;
	int use_pci_mmio_bridge;
};

static int rocm_xio_init(struct memory_ctx *ctx)
{
	struct rocm_xio_memory_ctx *xctx =
		container_of(ctx, struct rocm_xio_memory_ctx, base);
	int count = 0;
	hipError_t err = hipGetDeviceCount(&count);

	if (err != hipSuccess) {
		fprintf(stderr, "rocm-xio: hipGetDeviceCount failed: %d\n", err);
		return FAILURE;
	}
	if (xctx->device_id >= count) {
		fprintf(stderr,
			"rocm-xio: requested device %d but only %d found\n",
			xctx->device_id, count);
		return FAILURE;
	}
	HIP_CHECK(hipSetDevice(xctx->device_id));

	hipDeviceProp_t prop = {0};
	HIP_CHECK(hipGetDeviceProperties(&prop, xctx->device_id));

	printf("rocm-xio: Using GPU %d: %s (PCI 0x%x)\n",
	       xctx->device_id, prop.name, prop.pciBusID);

	return SUCCESS;
}

static int rocm_xio_destroy(struct memory_ctx *ctx)
{
	struct rocm_xio_memory_ctx *xctx =
		container_of(ctx, struct rocm_xio_memory_ctx, base);
	free(xctx);
	return SUCCESS;
}

static int rocm_xio_allocate_buffer(struct memory_ctx *ctx, int alignment,
				    uint64_t size, int *dmabuf_fd,
				    uint64_t *dmabuf_offset, void **addr,
				    bool *can_init)
{
	(void)ctx;
	(void)alignment;
	void *buf = NULL;
	size_t buf_size = (size + ACCEL_PAGE_SIZE - 1) & ~(ACCEL_PAGE_SIZE - 1);
	hipError_t err;

	err = hipHostMalloc(&buf, buf_size, hipHostMallocDefault);
	if (err != hipSuccess) {
		fprintf(stderr, "rocm-xio: hipHostMalloc %lu failed: %d\n",
			(unsigned long)buf_size, err);
		return FAILURE;
	}
	memset(buf, 0, buf_size);

	*addr = buf;
	*dmabuf_fd = 0;
	*dmabuf_offset = 0;
	*can_init = true;

	printf("rocm-xio: allocated %lu bytes at %p\n",
	       (unsigned long)buf_size, buf);
	return SUCCESS;
}

static int rocm_xio_free_buffer(struct memory_ctx *ctx, int dmabuf_fd,
				void *addr, uint64_t size)
{
	(void)ctx;
	(void)dmabuf_fd;
	(void)size;
	hipHostFree(addr);
	return SUCCESS;
}

bool rocm_xio_memory_supported(void)
{
	return true;
}

struct memory_ctx *rocm_xio_memory_create(struct perftest_parameters *params)
{
	struct rocm_xio_memory_ctx *ctx;

	ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return NULL;

	ctx->base.init = rocm_xio_init;
	ctx->base.destroy = rocm_xio_destroy;
	ctx->base.allocate_buffer = rocm_xio_allocate_buffer;
	ctx->base.free_buffer = rocm_xio_free_buffer;
	ctx->base.copy_host_to_buffer = memcpy;
	ctx->base.copy_buffer_to_host = memcpy;
	ctx->base.copy_buffer_to_buffer = memcpy;
	ctx->device_id = params->rocm_device_id;
	ctx->use_pci_mmio_bridge = params->rocm_xio_pci_mmio_bridge;

	return &ctx->base;
}

/*
 * DV resource management.
 */

static int dv_dlopen(struct rocm_xio_dv_state *st)
{
	int flags = RTLD_LAZY;
	void *h = dlopen("librocm_ernic.so", flags);

	if (!h)
		h = dlopen("/usr/lib/libibverbs/librocm_ernic.so", flags);
	if (!h)
		h = dlopen("/usr/local/lib/librocm_ernic.so", flags);
	if (!h) {
		fprintf(stderr, "rocm-xio: dlopen librocm_ernic.so: %s\n",
			dlerror());
		return FAILURE;
	}
	st->dl_handle = h;

#define DV_LOAD(field, name) do {                               \
	st->dv.field = (typeof(st->dv.field))dlsym(h, name);   \
	if (!st->dv.field) {                                    \
		fprintf(stderr,                                 \
			"rocm-xio: dlsym %s: %s\n",            \
			name, dlerror());                       \
		return FAILURE;                                 \
	}                                                       \
} while (0)

	DV_LOAD(umem_reg, "rocm_ernic_dv_umem_reg");
	DV_LOAD(umem_dereg, "rocm_ernic_dv_umem_dereg");
	DV_LOAD(create_cq, "rocm_ernic_dv_create_cq");
	DV_LOAD(destroy_cq, "rocm_ernic_dv_destroy_cq");
	DV_LOAD(create_qp, "rocm_ernic_dv_create_qp");
	DV_LOAD(destroy_qp, "rocm_ernic_dv_destroy_qp");
	DV_LOAD(modify_qp, "rocm_ernic_dv_modify_qp");
	DV_LOAD(get_cq_id, "rocm_ernic_dv_get_cq_id");
	DV_LOAD(get_qp_attr, "rocm_ernic_dv_get_qp_attr");
#undef DV_LOAD

	return SUCCESS;
}

static int dv_create_cq(struct rocm_xio_dv_state *st,
			struct ibv_context *ibctx, int ncqe,
			struct ibv_cq **out_cq, const char *label)
{
	uint32_t cqe_size = 64;
	uint32_t depth = (uint32_t)ncqe;
	size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
	size_t alloc_len = depth * cqe_size;

	if (alloc_len < pgsz)
		alloc_len = pgsz;
	alloc_len = (alloc_len + pgsz - 1) & ~(pgsz - 1);

	void *buf = NULL;
	if (posix_memalign(&buf, pgsz, alloc_len) != 0) {
		fprintf(stderr, "rocm-xio: posix_memalign %s failed\n",
			label);
		return FAILURE;
	}
	memset(buf, 0, alloc_len);
	mlock(buf, alloc_len);
	hipError_t herr = hipHostRegister(buf, alloc_len,
					  hipHostRegisterDefault);
	if (herr != hipSuccess)
		fprintf(stderr, "rocm-xio: hipHostRegister %s: %s\n",
			label, hipGetErrorString(herr));

	struct rocm_ernic_dv_umem_attr ua;
	memset(&ua, 0, sizeof(ua));
	ua.addr = buf;
	ua.size = alloc_len;
	ua.access_flags = IBV_ACCESS_LOCAL_WRITE;
	ua.dmabuf_fd = -1;

	struct rocm_ernic_dv_umem *umem = st->dv.umem_reg(ibctx, &ua);
	if (!umem) {
		fprintf(stderr, "rocm-xio: umem_reg %s failed\n", label);
		hipHostFree(buf);
		return FAILURE;
	}

	struct rocm_ernic_dv_cq_init_attr ca;
	memset(&ca, 0, sizeof(ca));
	ca.umem_handle = umem;
	ca.ncqe = depth;

	*out_cq = st->dv.create_cq(ibctx, &ca);
	if (!*out_cq) {
		fprintf(stderr, "rocm-xio: create_cq %s failed\n", label);
		st->dv.umem_dereg(umem);
		hipHostFree(buf);
		return FAILURE;
	}

	if (strcmp(label, "SCQ") == 0) {
		st->cq_buf = buf;
		st->cq_buf_len = alloc_len;
		st->cq_umem = umem;
		st->cq_depth = depth;
	}

	printf("rocm-xio: created DV CQ %s (depth=%u, buf=%p)\n",
	       label, depth, buf);
	return SUCCESS;
}

/*
 * Extern: bridge setup in rocm_xio_kernels.hip
 * (C++ / rocm-xio API calls).
 */
extern int rocm_xio_setup_mmio_bridge(
	const char *rdma_dev_name,
	uint64_t uar_mmap_offset,
	uint32_t uar_qp_offset,
	void **shadow_gpu_out,
	uint16_t *ernic_bdf_out,
	uint32_t *uar_bar_offset_out);

int rocm_xio_create_dv_resources(struct pingpong_context *ctx,
				 struct perftest_parameters *user_param)
{
	struct rocm_xio_dv_state *st;
	size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);

	st = calloc(1, sizeof(*st));
	if (!st)
		return FAILURE;

	st->pci_mmio_bridge = user_param->rocm_xio_pci_mmio_bridge;

	/* Save RDMA device name for sysfs lookups */
	if (user_param->ib_devname)
		snprintf(st->rdma_dev_name, sizeof(st->rdma_dev_name),
			 "%s", user_param->ib_devname);

	if (dv_dlopen(st) != SUCCESS) {
		free(st);
		return FAILURE;
	}

	/* Create send CQ via DV */
	if (dv_create_cq(st, ctx->context,
			 user_param->tx_depth * user_param->num_of_qps,
			 &ctx->send_cq, "SCQ") != SUCCESS) {
		free(st);
		return FAILURE;
	}

	/* Create recv CQ via DV */
	struct ibv_cq *recv_cq = NULL;
	if (dv_create_cq(st, ctx->context,
			 user_param->rx_depth * user_param->num_of_qps,
			 &recv_cq, "RCQ") != SUCCESS) {
		free(st);
		return FAILURE;
	}
	st->dv_send_cq = ctx->send_cq;
	st->dv_recv_cq = recv_cq;
	ctx->recv_cq = recv_cq;

	/* Allocate SQ buffer: page 0 = ring state, rest = WQEs */
	uint32_t sq_wqe_sz = 128;
	uint32_t sq_depth = (uint32_t)user_param->tx_depth;
	uint32_t sq_len = sq_depth * sq_wqe_sz;

	if (sq_len < pgsz)
		sq_len = (uint32_t)pgsz;
	sq_len += (uint32_t)pgsz;

	void *sq_ptr = NULL;
	if (posix_memalign(&sq_ptr, pgsz, sq_len) != 0) {
		fprintf(stderr, "rocm-xio: posix_memalign SQ failed\n");
		free(st);
		return FAILURE;
	}
	memset(sq_ptr, 0, sq_len);
	mlock(sq_ptr, sq_len);
	hipError_t herr = hipHostRegister(sq_ptr, sq_len,
					  hipHostRegisterDefault);
	if (herr != hipSuccess)
		fprintf(stderr, "rocm-xio: hipHostRegister SQ: %s\n",
			hipGetErrorString(herr));

	/* Write marker to page 0 to verify UMEM mapping */
	volatile int32_t *marker = (volatile int32_t *)sq_ptr;
	marker[0] = 0xDEAD;
	marker[1] = 0xBEEF;
	printf("rocm-xio: SQ page0 marker written: "
	       "[0]=0x%x [1]=0x%x at %p\n",
	       marker[0], marker[1], sq_ptr);

	struct rocm_ernic_dv_umem_attr sq_ua;
	memset(&sq_ua, 0, sizeof(sq_ua));
	sq_ua.addr = sq_ptr;
	sq_ua.size = sq_len;
	sq_ua.access_flags = IBV_ACCESS_LOCAL_WRITE;
	sq_ua.dmabuf_fd = -1;

	st->sq_umem = st->dv.umem_reg(ctx->context, &sq_ua);
	if (!st->sq_umem) {
		fprintf(stderr, "rocm-xio: umem_reg SQ failed\n");
		hipHostFree(sq_ptr);
		free(st);
		return FAILURE;
	}
	st->sq_buf = sq_ptr;
	st->sq_buf_len = sq_len;
	st->sq_depth = sq_depth;
	st->sq_wqe_size = sq_wqe_sz;
	st->sq_ring_prod_tail = (volatile int32_t *)sq_ptr;
	st->sq_wqe_base = (char *)sq_ptr + pgsz;

	/* Allocate RQ buffer (minimal) */
	uint32_t rq_wqe_sz = 64;
	uint32_t rq_len = (uint32_t)pgsz;

	void *rq_ptr = NULL;
	if (posix_memalign(&rq_ptr, pgsz, rq_len) != 0) {
		fprintf(stderr, "rocm-xio: posix_memalign RQ failed\n");
		free(st);
		return FAILURE;
	}
	memset(rq_ptr, 0, rq_len);
	mlock(rq_ptr, rq_len);

	struct rocm_ernic_dv_umem_attr rq_ua;
	memset(&rq_ua, 0, sizeof(rq_ua));
	rq_ua.addr = rq_ptr;
	rq_ua.size = rq_len;
	rq_ua.access_flags = IBV_ACCESS_LOCAL_WRITE;
	rq_ua.dmabuf_fd = -1;

	st->rq_umem = st->dv.umem_reg(ctx->context, &rq_ua);
	if (!st->rq_umem) {
		fprintf(stderr, "rocm-xio: umem_reg RQ failed\n");
		hipHostFree(rq_ptr);
		free(st);
		return FAILURE;
	}
	st->rq_buf = rq_ptr;
	st->rq_buf_len = rq_len;

	/* Create QP via DV */
	struct rocm_ernic_dv_qp_init_attr qp_attr;
	memset(&qp_attr, 0, sizeof(qp_attr));
	qp_attr.qp_type = IBV_QPT_RC;
	qp_attr.max_send_wr = sq_depth;
	qp_attr.max_recv_wr = 0;
	qp_attr.max_send_sge = 1;
	qp_attr.max_recv_sge = 0;
	qp_attr.max_inline_data = 0;
	qp_attr.send_cq = ctx->send_cq;
	qp_attr.recv_cq = recv_cq;
	qp_attr.sq_umem_handle = st->sq_umem;
	qp_attr.sq_len = sq_len;
	qp_attr.sq_wqe_size = sq_wqe_sz;
	qp_attr.rq_umem_handle = st->rq_umem;
	qp_attr.rq_len = rq_len;
	qp_attr.rq_wqe_size = rq_wqe_sz;

	ctx->qp[0] = st->dv.create_qp(ctx->pd, &qp_attr);
	if (!ctx->qp[0]) {
		fprintf(stderr, "rocm-xio: create_qp failed\n");
		free(st);
		return FAILURE;
	}

	/* Get QP attributes for doorbell info */
	struct rocm_ernic_dv_qp_attr qa;
	memset(&qa, 0, sizeof(qa));
	st->dv.get_qp_attr(ctx->qp[0], &qa);
	st->uar_ptr = qa.uar_ptr;
	st->uar_qp_offset = qa.uar_qp_offset;
	st->uar_cq_offset = qa.uar_cq_offset;
	st->uar_mmap_offset = qa.uar_mmap_offset;
	st->qp_handle = qa.qp_handle;

	/* Map UAR for GPU access (fallback path) */
	if (st->uar_ptr) {
		herr = hipHostRegister(st->uar_ptr, (size_t)getpagesize(),
				       hipHostRegisterDefault);
		if (herr == hipSuccess) {
			void *dev_uar = NULL;
			herr = hipHostGetDevicePointer(&dev_uar, st->uar_ptr, 0);
			if (herr == hipSuccess)
				st->gpu_uar_ptr = (volatile uint32_t *)dev_uar;
			else
				fprintf(stderr,
					"rocm-xio: hipHostGetDevicePointer(uar): %s\n",
					hipGetErrorString(herr));
		} else {
			fprintf(stderr,
				"rocm-xio: hipHostRegister(uar): %s\n",
				hipGetErrorString(herr));
		}
	}

	/* Set up pci-mmio-bridge for doorbell routing */
	if (st->pci_mmio_bridge) {
		void *shadow_gpu = NULL;
		uint16_t ernic_bdf = 0;
		uint32_t bar_offset = 0;
		int bret = rocm_xio_setup_mmio_bridge(
			st->rdma_dev_name,
			st->uar_mmap_offset,
			st->uar_qp_offset,
			&shadow_gpu, &ernic_bdf, &bar_offset);
		if (bret == 0) {
			st->mmio_bridge_shadow = shadow_gpu;
			st->ernic_target_bdf = ernic_bdf;
			st->uar_bar_offset = bar_offset;
		} else {
			fprintf(stderr,
				"rocm-xio: pci-mmio-bridge setup "
				"failed, falling back to direct UAR\n");
		}
	}

	/* Dump ring state at page 0 of SQ buffer */
	volatile int32_t *rs = (volatile int32_t *)st->sq_buf;
	printf("rocm-xio: SQ ring state at %p: "
	       "[0]=%d [1]=%d [2]=%d [3]=%d\n",
	       st->sq_buf, rs[0], rs[1], rs[2], rs[3]);
	printf("rocm-xio: SQ WQE base=%p, prod_tail_ptr=%p\n",
	       st->sq_wqe_base, (void *)st->sq_ring_prod_tail);

	printf("rocm-xio: DV QP created (QPN=%u, handle=%u, "
	       "SQ depth=%u, UAR=%p, bridge=%s)\n",
	       ctx->qp[0]->qp_num, st->qp_handle,
	       st->sq_depth, st->uar_ptr,
	       st->mmio_bridge_shadow ? "enabled" : "disabled");

	ctx->rocm_xio_dv = st;
	return SUCCESS;
}

void rocm_xio_destroy_dv_resources(struct pingpong_context *ctx)
{
	struct rocm_xio_dv_state *st;

	if (!ctx || !ctx->rocm_xio_dv)
		return;

	st = (struct rocm_xio_dv_state *)ctx->rocm_xio_dv;

	if (st->dv_send_cq)
		st->dv.destroy_cq(st->dv_send_cq);
	if (st->dv_recv_cq)
		st->dv.destroy_cq(st->dv_recv_cq);

	if (st->sq_umem)
		st->dv.umem_dereg(st->sq_umem);
	if (st->rq_umem)
		st->dv.umem_dereg(st->rq_umem);
	if (st->cq_umem)
		st->dv.umem_dereg(st->cq_umem);

	if (st->sq_buf)
		hipHostFree(st->sq_buf);
	if (st->rq_buf)
		hipHostFree(st->rq_buf);
	if (st->cq_buf)
		hipHostFree(st->cq_buf);

	if (st->dl_handle)
		dlclose(st->dl_handle);

	free(st);
	ctx->rocm_xio_dv = NULL;
}

/*
 * Extern declarations for the HIP kernel launcher
 * functions defined in rocm_xio_kernels.hip.
 */
extern int rocm_xio_launch_write_bw_kernel(
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
	double *elapsed_usec);

extern int rocm_xio_launch_write_lat_kernel(
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
	double *iter_usec);

int rocm_xio_run_iter_bw(struct pingpong_context *ctx,
			 struct perftest_parameters *user_param)
{
	struct rocm_xio_dv_state *st =
		(struct rocm_xio_dv_state *)ctx->rocm_xio_dv;
	if (!st) {
		fprintf(stderr, "rocm-xio: DV state not initialized\n");
		return FAILURE;
	}

	double elapsed_usec = 0.0;

	volatile int32_t *rs = (volatile int32_t *)st->sq_buf;
	printf("rocm-xio: pre-launch ring state: "
	       "[0]=%d [1]=%d [2]=%d [3]=%d\n",
	       rs[0], rs[1], rs[2], rs[3]);
	printf("rocm-xio: remote_addr=0x%llx rkey=0x%x "
	       "lkey=0x%x local_buf=%p size=%u iters=%lu\n",
	       (unsigned long long)ctx->wr[0].wr.rdma.remote_addr,
	       ctx->wr[0].wr.rdma.rkey,
	       ctx->mr[0]->lkey, ctx->buf[0],
	       (uint32_t)user_param->size,
	       (unsigned long)user_param->iters);

	user_param->tposted[0] = get_cycles();

	int rc = rocm_xio_launch_write_bw_kernel(
		st->sq_wqe_base, st->cq_buf,
		st->sq_ring_prod_tail, st->gpu_uar_ptr,
		st->uar_qp_offset,
		ctx->wr[0].wr.rdma.remote_addr,
		ctx->wr[0].wr.rdma.rkey,
		ctx->mr[0]->lkey, ctx->buf[0],
		(uint32_t)user_param->size,
		user_param->iters,
		st->sq_depth, st->sq_wqe_size,
		st->qp_handle, st->cq_depth,
		st->mmio_bridge_shadow,
		st->ernic_target_bdf,
		st->uar_bar_offset,
		&elapsed_usec);

	user_param->tcompleted[0] = get_cycles();

	if (rc != SUCCESS)
		return FAILURE;

	return SUCCESS;
}

int rocm_xio_run_iter_lat_write(struct pingpong_context *ctx,
				struct perftest_parameters *user_param)
{
	struct rocm_xio_dv_state *st =
		(struct rocm_xio_dv_state *)ctx->rocm_xio_dv;
	if (!st) {
		fprintf(stderr, "rocm-xio: DV state not initialized\n");
		return FAILURE;
	}

	double *iter_usec = calloc(user_param->iters, sizeof(double));
	if (!iter_usec)
		return FAILURE;

	int rc = rocm_xio_launch_write_lat_kernel(
		st->sq_wqe_base, st->cq_buf,
		st->sq_ring_prod_tail, st->gpu_uar_ptr,
		st->uar_qp_offset,
		ctx->wr[0].wr.rdma.remote_addr,
		ctx->wr[0].wr.rdma.rkey,
		ctx->mr[0]->lkey, ctx->buf[0],
		(uint32_t)user_param->size,
		user_param->iters,
		st->sq_depth, st->sq_wqe_size,
		st->qp_handle, st->cq_depth,
		st->mmio_bridge_shadow,
		st->ernic_target_bdf,
		st->uar_bar_offset,
		iter_usec);

	if (rc != SUCCESS) {
		free(iter_usec);
		return FAILURE;
	}

	for (uint64_t i = 0; i < user_param->iters; i++) {
		user_param->tposted[i] = 0;
		user_param->tcompleted[i] = iter_usec[i];
	}

	free(iter_usec);
	return SUCCESS;
}

#endif /* HAVE_ROCM_XIO */
