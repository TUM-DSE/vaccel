// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "include/vaccel/ops/matmul.h" // IWYU pragma: export
#include "arg.h"
#include "session.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int vaccel_matmul_create_unpack(struct vaccel_session *sess,
				struct vaccel_arg *read, int nr_read,
				struct vaccel_arg *write, int nr_write);

int vaccel_create_mem_unpack(struct vaccel_session *sess,
			     struct vaccel_arg *read, int nr_read,
			     struct vaccel_arg *write, int nr_write);

int vaccel_destroy_mem_unpack(struct vaccel_session *sess,
			      struct vaccel_arg *read, int nr_read,
			      struct vaccel_arg *write, int nr_write);

int vaccel_matmul_destroy_unpack(struct vaccel_session *sess,
				 struct vaccel_arg *read, int nr_read,
				 struct vaccel_arg *write, int nr_write);

int vaccel_matmul_set_io_mem_unpack(struct vaccel_session *sess,
				    struct vaccel_arg *read, int nr_read,
				    struct vaccel_arg *write, int nr_write);

int vaccel_matmul_set_core_mask_unpack(struct vaccel_session *sess,
				       struct vaccel_arg *read, int nr_read,
				       struct vaccel_arg *write, int nr_write);

int vaccel_matmul_run_unpack(struct vaccel_session *sess,
			     struct vaccel_arg *read, int nr_read,
			     struct vaccel_arg *write, int nr_write);

int vaccel_matmul_set_matrix_unpack(struct vaccel_session *sess,
				    struct vaccel_arg *read, int nr_read,
				    struct vaccel_arg *write, int nr_write);

int vaccel_matmul_get_matrix_unpack(struct vaccel_session *sess,
				    struct vaccel_arg *read, int nr_read,
				    struct vaccel_arg *write, int nr_write);

int vaccel_matmul_get_props_unpack(struct vaccel_session *sess,
				    struct vaccel_arg *read, int nr_read,
				    struct vaccel_arg *write, int nr_write);
#ifdef __cplusplus
}
#endif
