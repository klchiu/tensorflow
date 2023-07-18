// Copyright (c) 2011-2023 Columbia University, System Level Design Group
// SPDX-License-Identifier: Apache-2.0
#ifndef __CFG_TF_MULT3_H__
#define __CFG_TF_MULT3_H__

#include "lib_esp.h"
// #include "tf_mult3_stratus.h"
#include <linux/ioctl.h>  // for _IOW()

extern "C" {
struct tf_mult3_stratus_access {
	struct esp_access esp;

	unsigned int tf_length;
	unsigned int tf_src_dst_offset_0;	// output
	unsigned int tf_src_dst_offset_1;	// input 1
	unsigned int tf_src_dst_offset_2;	// input 2
	unsigned int chunk_size;
	unsigned int src_offset;
	unsigned int dst_offset;
};
}

#define TF_MULT3_STRATUS_IOC_ACCESS	_IOW ('S', 0, struct tf_mult3_stratus_access)



// typedef uint64_t token_t;
// typedef float native_t;

#define fx2float             fixed64_to_double
#define float2fx             double_to_fixed64
#define FX_IL           34

struct esp_access esp2_mult3 =
    {
        .contig = NULL,
        .run = 0,
        .p2p_store = 0,
        .p2p_nsrcs = 0,
        .p2p_srcs = {"", "", "", ""},
        .coherence = ACC_COH_NONE,
        .footprint = 0,
        .alloc_policy = CONTIG_ALLOC_PREFERRED,
        .ddr_node = 0,
        .in_place = 0,
        .reuse_factor = 0,
};

struct tf_mult3_stratus_access tf_mult3_cfg_000[] = {
	{
        .esp = esp2_mult3,
		/* <<--descriptor-->> */
		.tf_length = 1024,
		.tf_src_dst_offset_0 = 0,
		.tf_src_dst_offset_1 = 1024,
		.tf_src_dst_offset_2 = 2048,
		.chunk_size = 4096,
		.src_offset = 0,
		.dst_offset = 0,
	}
};

esp_thread_info_t cfg_tf_mult3[] = {
	{
		.run = true,
		.devname = "tf_mult3_stratus.0",
        .hw_buf = NULL,
        .ioctl_req = TF_MULT3_STRATUS_IOC_ACCESS,
        .esp_desc = &(tf_mult3_cfg_000[0].esp),
        .fd = 0,
        .hw_ns = 0,
	}
};

#endif /* __CFG_TF_MULT3_H__ */
