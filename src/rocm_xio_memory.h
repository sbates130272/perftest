/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause
 *
 * rocm-xio GPU-initiated RDMA memory backend for
 * perftest.  Provides DV-based QP/CQ creation and
 * GPU-initiated data path via HIP kernels.
 */

#ifndef ROCM_XIO_MEMORY_H
#define ROCM_XIO_MEMORY_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "memory.h"
#include "config.h"

struct perftest_parameters;
struct pingpong_context;

#ifdef HAVE_ROCM_XIO

bool rocm_xio_memory_supported(void);
struct memory_ctx *rocm_xio_memory_create(struct perftest_parameters *params);

/*
 * DV resource creation: replaces standard ibv_create_cq/qp
 * when --use_rocm_xio is active.
 */
int rocm_xio_create_dv_resources(struct pingpong_context *ctx,
				 struct perftest_parameters *user_param);
void rocm_xio_destroy_dv_resources(struct pingpong_context *ctx);

/*
 * GPU-initiated iteration loops: replace run_iter_bw / run_iter_lat_write.
 */
int rocm_xio_run_iter_bw(struct pingpong_context *ctx,
			 struct perftest_parameters *user_param);
int rocm_xio_run_iter_lat_write(struct pingpong_context *ctx,
				struct perftest_parameters *user_param);

#else /* !HAVE_ROCM_XIO */

static inline bool rocm_xio_memory_supported(void) {
	return false;
}

static inline struct memory_ctx *rocm_xio_memory_create(
	struct perftest_parameters *params) {
	(void)params;
	return NULL;
}

static inline int rocm_xio_create_dv_resources(
	struct pingpong_context *ctx,
	struct perftest_parameters *user_param) {
	(void)ctx; (void)user_param;
	return 1;
}

static inline void rocm_xio_destroy_dv_resources(
	struct pingpong_context *ctx) {
	(void)ctx;
}

static inline int rocm_xio_run_iter_bw(
	struct pingpong_context *ctx,
	struct perftest_parameters *user_param) {
	(void)ctx; (void)user_param;
	return 1;
}

static inline int rocm_xio_run_iter_lat_write(
	struct pingpong_context *ctx,
	struct perftest_parameters *user_param) {
	(void)ctx; (void)user_param;
	return 1;
}

#endif /* HAVE_ROCM_XIO */

#endif /* ROCM_XIO_MEMORY_H */
