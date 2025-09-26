// SPDX-License-Identifier: Apache-2.0

#include "arg.h"
#include "error.h"
#include "log.h"
#include "matmul.h"
#include "op.h"
#include "plugin.h"
#include "prof.h"
#include "session.h"
#include <inttypes.h>
#include <stdint.h>

#define _unused __attribute__((unused))

static struct vaccel_prof_region rknn_op_stats =
	VACCEL_PROF_REGION_INIT("vaccel_rknn_op");



typedef int (*matmul_create_fn_t)(struct vaccel_session *sess, vaccel_matmul_ctx* ctx,
			vaccel_matmul_info* info, vaccel_matmul_io_attr* io_attr);

int vaccel_matmul_create(struct vaccel_session *sess, vaccel_matmul_ctx* ctx,
			vaccel_matmul_info* info, vaccel_matmul_io_attr* io_attr)
{
	int ret;

	if (!sess)
		return VACCEL_EINVAL;

	vaccel_debug("session:%" PRId64
		     " Looking for plugin implementing matmul_create",
		     sess->id);

	vaccel_prof_region_start(&rknn_op_stats);

	matmul_create_fn_t plugin_matmul_create =
		plugin_get_op_func(VACCEL_OP_MATMUL_CREATE, sess->hint);
	if (!plugin_matmul_create) {
		ret = VACCEL_ENOTSUP;
		goto out;
	}

	ret = plugin_matmul_create(sess, ctx, info, io_attr);

	out:
		vaccel_prof_region_stop(&rknn_op_stats);

	return ret;

}

int vaccel_matmul_create_unpack(struct vaccel_session *sess, struct vaccel_arg *read,
			int nr_read, struct vaccel_arg *write, int nr_write)
{
	if (nr_read != 1) {
		vaccel_error("Wrong number of read arguments in matmul_create: %d",
			     nr_read);
		return VACCEL_EINVAL;
	}

	if (nr_write != 2) {
		vaccel_error("Wrong number of write arguments in matmul_create: %d",
			     nr_write);
		return VACCEL_EINVAL;
	}

	vaccel_matmul_info* info = read[0].buf;

	vaccel_matmul_ctx* ctx = write[0].buf;
	vaccel_matmul_io_attr* io_attr = write[1].buf;

	return vaccel_matmul_create(sess, ctx, info, io_attr);
}


typedef int (*vaccel_create_mem_fn_t)(struct vaccel_session *sess, vaccel_matmul_ctx ctx, uint32_t size, vaccel_tensor_mem* result);

int vaccel_create_mem(struct vaccel_session *sess, vaccel_matmul_ctx ctx, uint32_t size, vaccel_tensor_mem* result)
{
	int ret;

	if (!sess)
		return VACCEL_EINVAL;

	vaccel_debug("session:%" PRId64
		     " Looking for plugin implementing create_mem",
		     sess->id);

	vaccel_prof_region_start(&rknn_op_stats);

	vaccel_create_mem_fn_t plugin_create_mem =
		plugin_get_op_func(VACCEL_OP_CREATE_MEM, sess->hint);
	if (!plugin_create_mem) {
		ret = VACCEL_ENOTSUP;
		goto out;
	}

	ret = plugin_create_mem(sess, ctx, size, result);

	out:
		vaccel_prof_region_stop(&rknn_op_stats);

	return ret;

}

int vaccel_create_mem_unpack(struct vaccel_session *sess, struct vaccel_arg *read,
			int nr_read, struct vaccel_arg *write, int nr_write)
{
	if (nr_read != 2) {
		vaccel_error("Wrong number of read arguments in create_mem: %d",
			     nr_read);
		return VACCEL_EINVAL;
	}

	if (nr_write != 1) {
		vaccel_error("Wrong number of write arguments in create_mem: %d",
			     nr_write);
		return VACCEL_EINVAL;
	}

	vaccel_matmul_ctx ctx = *(vaccel_matmul_ctx*)read[0].buf;
	uint32_t size = *(uint32_t*)read[1].buf;

	vaccel_tensor_mem* result = write[0].buf;

	return vaccel_create_mem(sess, ctx, size, result);
}

typedef int (*vaccel_destroy_mem_fn_t)(struct vaccel_session *sess, vaccel_matmul_ctx ctx, vaccel_tensor_mem *mem);

int vaccel_destroy_mem(struct vaccel_session *sess, vaccel_matmul_ctx ctx, vaccel_tensor_mem *mem)
{
	int ret;

	if (!sess)
		return VACCEL_EINVAL;

	vaccel_debug("session:%" PRId64
		     " Looking for plugin implementing destroy_mem",
		     sess->id);

	vaccel_prof_region_start(&rknn_op_stats);

	vaccel_destroy_mem_fn_t plugin_destroy_mem =
		plugin_get_op_func(VACCEL_OP_DESTROY_MEM, sess->hint);
	if (!plugin_destroy_mem) {
		ret = VACCEL_ENOTSUP;
		goto out;
	}

	ret = plugin_destroy_mem(sess, ctx, mem);

	out:
		vaccel_prof_region_stop(&rknn_op_stats);

	return ret;

}

int vaccel_destroy_mem_unpack(struct vaccel_session *sess, struct vaccel_arg *read,
			int nr_read, _unused struct vaccel_arg *write, int nr_write)
{
	if (nr_read != 2) {
		vaccel_error("Wrong number of read arguments in destroy_mem: %d",
			     nr_read);
		return VACCEL_EINVAL;
	}

	if (nr_write != 0) {
		vaccel_error("Wrong number of write arguments in destroy_mem: %d",
			     nr_write);
		return VACCEL_EINVAL;
	}

	vaccel_matmul_ctx ctx = *(vaccel_matmul_ctx*)read[0].buf;
	vaccel_tensor_mem* mem = read[1].buf;


	return vaccel_destroy_mem(sess, ctx, mem);
}

typedef int (*vaccel_matmul_destroy_fn_t)(struct vaccel_session *sess, vaccel_matmul_ctx ctx);

int vaccel_matmul_destroy(struct vaccel_session *sess, vaccel_matmul_ctx ctx)
{
	int ret;

	if (!sess)
		return VACCEL_EINVAL;

	vaccel_debug("session:%" PRId64
		     " Looking for plugin implementing matmul_destroy",
		     sess->id);

	vaccel_prof_region_start(&rknn_op_stats);

	vaccel_matmul_destroy_fn_t plugin_matmul_destroy =
		plugin_get_op_func(VACCEL_OP_MATMUL_DESTROY, sess->hint);
	if (!plugin_matmul_destroy) {
		ret = VACCEL_ENOTSUP;
		goto out;
	}

	ret = plugin_matmul_destroy(sess, ctx);

	out:
		vaccel_prof_region_stop(&rknn_op_stats);

	return ret;

}

int vaccel_matmul_destroy_unpack(struct vaccel_session *sess, struct vaccel_arg *read,
			int nr_read, _unused struct vaccel_arg *write, int nr_write)
{
	if (nr_read != 1) {
		vaccel_error("Wrong number of read arguments in matmul_destroy: %d",
			     nr_read);
		return VACCEL_EINVAL;
	}

	if (nr_write != 0) {
		vaccel_error("Wrong number of write arguments in matmul_destroy: %d",
			     nr_write);
		return VACCEL_EINVAL;
	}

	vaccel_matmul_ctx ctx = *(vaccel_matmul_ctx*)read[0].buf;


	return vaccel_matmul_destroy(sess, ctx);
}

typedef int (*vaccel_matmul_set_io_mem_fn_t)(struct vaccel_session *sess, vaccel_matmul_ctx ctx, vaccel_tensor_mem_handle* mem, vaccel_matmul_tensor_attr* attr);

int vaccel_matmul_set_io_mem(struct vaccel_session *sess, vaccel_matmul_ctx ctx, vaccel_tensor_mem_handle* mem, vaccel_matmul_tensor_attr* attr)
{
	int ret;

	if (!sess)
		return VACCEL_EINVAL;

	vaccel_debug("session:%" PRId64
		     " Looking for plugin implementing matmul_set_io_mem",
		     sess->id);

	vaccel_prof_region_start(&rknn_op_stats);

	vaccel_matmul_set_io_mem_fn_t plugin_matmul_set_io_mem =
		plugin_get_op_func(VACCEL_OP_MATMUL_SET_IO, sess->hint);
	if (!plugin_matmul_set_io_mem) {
		ret = VACCEL_ENOTSUP;
		goto out;
	}

	ret = plugin_matmul_set_io_mem(sess, ctx, mem, attr);

	out:
		vaccel_prof_region_stop(&rknn_op_stats);

	return ret;

}

int vaccel_matmul_set_io_mem_unpack(struct vaccel_session *sess, struct vaccel_arg *read,
			int nr_read, _unused struct vaccel_arg *write, int nr_write)
{
	if (nr_read != 3) {
		vaccel_error("Wrong number of read arguments in matmul_set_io_mem: %d",
			     nr_read);
		return VACCEL_EINVAL;
	}

	if (nr_write != 0) {
		vaccel_error("Wrong number of write arguments in matmul_set_io_mem: %d",
			     nr_write);
		return VACCEL_EINVAL;
	}

	vaccel_matmul_ctx ctx = *(vaccel_matmul_ctx*)read[0].buf;
	vaccel_tensor_mem_handle* mem = *(vaccel_tensor_mem_handle**)read[1].buf;
	vaccel_matmul_tensor_attr* attr = read[2].buf;


	return vaccel_matmul_set_io_mem(sess, ctx, mem, attr);
}

typedef int (*vaccel_matmul_set_core_mask_fn_t)(struct vaccel_session *sess, vaccel_matmul_ctx ctx, vaccel_core_mask core_mask);

int vaccel_matmul_set_core_mask(struct vaccel_session *sess, vaccel_matmul_ctx ctx, vaccel_core_mask core_mask)
{
	int ret;

	if (!sess)
		return VACCEL_EINVAL;

	vaccel_debug("session:%" PRId64
		     " Looking for plugin implementing matmul_set_core_mask",
		     sess->id);

	vaccel_prof_region_start(&rknn_op_stats);

	vaccel_matmul_set_core_mask_fn_t plugin_matmul_set_core_mask =
		plugin_get_op_func(VACCEL_OP_MATMUL_SET_CORE_MASK, sess->hint);
	if (!plugin_matmul_set_core_mask) {
		ret = VACCEL_ENOTSUP;
		goto out;
	}

	ret = plugin_matmul_set_core_mask(sess, ctx, core_mask);

	out:
		vaccel_prof_region_stop(&rknn_op_stats);

	return ret;

}

int vaccel_matmul_set_core_mask_unpack(struct vaccel_session *sess, struct vaccel_arg *read,
			int nr_read, _unused struct vaccel_arg *write, int nr_write)
{
	if (nr_read != 2) {
		vaccel_error("Wrong number of read arguments in matmul_set_core_mask: %d",
			     nr_read);
		return VACCEL_EINVAL;
	}

	if (nr_write != 0) {
		vaccel_error("Wrong number of write arguments in matmul_set_core_mask: %d",
			     nr_write);
		return VACCEL_EINVAL;
	}

	vaccel_matmul_ctx ctx = *(vaccel_matmul_ctx*)read[0].buf;
	vaccel_core_mask core_mask = *(vaccel_core_mask*)read[1].buf;


	return vaccel_matmul_set_core_mask(sess, ctx, core_mask);
}

typedef int (*vaccel_matmul_run_fn_t)(struct vaccel_session *sess, vaccel_matmul_ctx ctx);

int vaccel_matmul_run(struct vaccel_session *sess, vaccel_matmul_ctx ctx)
{
	int ret;

	if (!sess)
		return VACCEL_EINVAL;

	vaccel_debug("session:%" PRId64
		     " Looking for plugin implementing matmul_run",
		     sess->id);

	vaccel_prof_region_start(&rknn_op_stats);

	vaccel_matmul_run_fn_t plugin_matmul_run =
		plugin_get_op_func(VACCEL_OP_MATMUL_RUN, sess->hint);
	if (!plugin_matmul_run) {
		ret = VACCEL_ENOTSUP;
		goto out;
	}

	ret = plugin_matmul_run(sess, ctx);

	out:
		vaccel_prof_region_stop(&rknn_op_stats);

	return ret;

}

int vaccel_matmul_run_unpack(struct vaccel_session *sess, struct vaccel_arg *read,
			int nr_read, _unused struct vaccel_arg *write, int nr_write)
{
	if (nr_read != 1) {
		vaccel_error("Wrong number of read arguments in matmul_run: %d",
			     nr_read);
		return VACCEL_EINVAL;
	}

	if (nr_write != 0) {
		vaccel_error("Wrong number of write arguments in matmul_run: %d",
			     nr_write);
		return VACCEL_EINVAL;
	}

	vaccel_matmul_ctx ctx = *(vaccel_matmul_ctx*)read[0].buf;


	return vaccel_matmul_run(sess, ctx);
}

typedef int (*vaccel_matmul_set_matrix_fn_t)(struct vaccel_session *sess, vaccel_tensor_mem_handle* dst, void* src, size_t nbytes);
int vaccel_matmul_set_matrix(struct vaccel_session *sess, vaccel_tensor_mem_handle* dst, void* src, size_t nbytes)
{
	int ret;

	if (!sess)
		return VACCEL_EINVAL;

	vaccel_debug("session:%" PRId64
		     " Looking for plugin implementing matmul_set_matrix",
		     sess->id);

	vaccel_prof_region_start(&rknn_op_stats);

	vaccel_matmul_set_matrix_fn_t plugin_matmul_set_matrix=
		plugin_get_op_func(VACCEL_OP_MATMUL_SET_MATRIX, sess->hint);
	if (!plugin_matmul_set_matrix) {
		ret = VACCEL_ENOTSUP;
		goto out;
	}

	ret = plugin_matmul_set_matrix(sess, dst, src, nbytes);

out:
	vaccel_prof_region_stop(&rknn_op_stats);

	return ret;
}

int vaccel_matmul_set_matrix_unpack(struct vaccel_session *sess, struct vaccel_arg *read,
			     int nr_read, _unused struct vaccel_arg *write, int nr_write)
{
	if (nr_read != 3) {
		vaccel_error("Wrong number of read arguments in matmul_set_matrix: %d",
			     nr_read);
		return VACCEL_EINVAL;
	}

	if (nr_write != 0) {
		vaccel_error("Wrong number of write arguments in matmul_set_matrix: %d",
			     nr_write);
		return VACCEL_EINVAL;
	}

	vaccel_tensor_mem_handle* dst = *(vaccel_tensor_mem_handle**)read[0].buf;
	void* src = read[1].buf;
	size_t nbytes = *(size_t*)read[2].buf;


	return vaccel_matmul_set_matrix(sess, dst, src, nbytes);
}

typedef int (*vaccel_matmul_get_matrix_fn_t)(struct vaccel_session *sess, void* dst, vaccel_tensor_mem_handle* src, size_t nbytes);
int vaccel_matmul_get_matrix(struct vaccel_session *sess, void* dst, vaccel_tensor_mem_handle* src, size_t nbytes)
{
	int ret;

	if (!sess)
		return VACCEL_EINVAL;

	vaccel_debug("session:%" PRId64
		     " Looking for plugin implementing matmul_get_matrix",
		     sess->id);

	vaccel_prof_region_start(&rknn_op_stats);

	vaccel_matmul_get_matrix_fn_t plugin_matmul_get_matrix=
		plugin_get_op_func(VACCEL_OP_MATMUL_GET_MATRIX, sess->hint);
	if (!plugin_matmul_get_matrix) {
		ret = VACCEL_ENOTSUP;
		goto out;
	}

	ret = plugin_matmul_get_matrix(sess, dst, src, nbytes);

out:
	vaccel_prof_region_stop(&rknn_op_stats);

	return ret;
}

int vaccel_matmul_get_matrix_unpack(struct vaccel_session *sess, struct vaccel_arg *read,
				    int nr_read, struct vaccel_arg *write, int nr_write)
{
	if (nr_read != 2) {
		vaccel_error("Wrong number of read arguments in matmul_set_matrix: %d",
			     nr_read);
		return VACCEL_EINVAL;
	}

	if (nr_write != 1) {
		vaccel_error("Wrong number of write arguments in matmul_set_matrix: %d",
			     nr_write);
		return VACCEL_EINVAL;
	}

	vaccel_tensor_mem_handle* src = *(vaccel_tensor_mem_handle**)read[0].buf;
	size_t nbytes = *(size_t*)read[1].buf;

	void* dst = write[0].buf;

	return vaccel_matmul_get_matrix(sess, dst, src, nbytes);
}

__attribute__((constructor)) static void vaccel_ops_init(void)
{
}

__attribute__((destructor)) static void vaccel_ops_fini(void)
{
	vaccel_prof_region_print(&rknn_op_stats);
	vaccel_prof_region_release(&rknn_op_stats);
}
