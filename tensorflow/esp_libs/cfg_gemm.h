// Copyright (c) 2011-2023 Columbia University, System Level Design Group
// SPDX-License-Identifier: Apache-2.0
#ifndef __CFG_GEMM_H__
#define __CFG_GEMM_H__

#include "lib_esp.h"
// #include "gemm_stratus.h"
#include <linux/ioctl.h>  // for _IOW()


extern "C" {
struct gemm_stratus_access {
	struct esp_access esp;
	/* <<--regs-->> */
	unsigned do_relu;
	unsigned transpose;
	unsigned ninputs;
	unsigned d3;
	unsigned d2;
	unsigned d1;
	unsigned st_offset;
	unsigned ld_offset1;
	unsigned ld_offset2;
	unsigned src_offset;
	unsigned dst_offset;
};
}

#define GEMM_STRATUS_IOC_ACCESS	_IOW ('S', 0, struct gemm_stratus_access)



// typedef uint64_t token_t;
typedef float native_t;

#define fx2float             fixed64_to_double
#define float2fx             double_to_fixed64
#define FX_IL           34

#define MAX_PRINTED_ERRORS 512

/* <<--params-def-->> */
#define DO_RELU 0
#define TRANSPOSE 1
#define NINPUTS 2
#define D3 8
#define D2 8
#define D1 8
#define ST_OFFSET (NINPUTS * (D1 * D2 + D2 * D3))
#define LD_OFFSET1 0
#define LD_OFFSET2 (NINPUTS * (D1 * D2))

#define NACC 1
#define ACC_TLB_ENTRIES 128
#define ACC_PAGE_SIZE (1 << 20)
#define MAX_SIZE (ACC_PAGE_SIZE * ACC_TLB_ENTRIES)
#define MAX_TESTS 30

struct esp_access esp2_gemm =
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

struct gemm_stratus_access gemm_cfg_000[] = {
	{
        .esp = esp2_gemm,
		/* <<--descriptor-->> */
		.do_relu = DO_RELU,
		.transpose = TRANSPOSE,
		.ninputs = NINPUTS,
		.d3 = D3,
		.d2 = D2,
		.d1 = D1,
		.st_offset = ST_OFFSET,
		.ld_offset1 = LD_OFFSET1,
		.ld_offset2 = LD_OFFSET2,
		.src_offset = 0,
		.dst_offset = 0,
	}
};

esp_thread_info_t cfg_gemm[] = {
	{
		.run = true,
		.devname = "gemm_stratus.0",
		.hw_buf = NULL,
		.ioctl_req = GEMM_STRATUS_IOC_ACCESS,
		.esp_desc = &(gemm_cfg_000[0].esp),
		.fd = 0,
		.hw_ns = 0,
	}
};
#endif /* __CFG_GEMM_H__ */
