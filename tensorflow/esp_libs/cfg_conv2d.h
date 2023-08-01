#ifndef _CFG_CONV2D_H_
#define _CFG_CONV2D_H_

#include "lib_esp.h"

// [humu]: conv2d_stratus.h
#include <linux/ioctl.h>  // for _IOW()

extern "C" {
struct conv2d_stratus_access {
  struct esp_access esp;
  /* <<--regs-->> */
  unsigned n_channels;
  unsigned feature_map_height;
  unsigned feature_map_width;
  unsigned n_filters;
  unsigned filter_dim;
  unsigned is_padded;
  unsigned stride;
  unsigned do_relu;
  unsigned pool_type;
  unsigned batch_size;
  unsigned src_offset;
  unsigned dst_offset;
};
}

#define CONV2D_STRATUS_IOC_ACCESS _IOW('S', 0, struct conv2d_stratus_access)

// [humu]: cfg.h
#ifdef __INT
#if (BITWIDTH == 32)
typedef int token_t;
#elif (BITWIDTH == 64)
typedef long long token_t;
#endif
#endif

#ifdef __FIXED
#if (BITWIDTH == 32)
typedef int token_t;
#define fx2float fixed32_to_float
#define float2fx float_to_fixed32
#define FX_IL 16
#elif (BITWIDTH == 64)
typedef long long token_t;
#define fx2float fixed64_to_double
#define float2fx double_to_fixed64
#define FX_IL 32
#endif
#endif

#ifdef __FLOAT
#if (BITWIDTH == 32)
typedef float token_t;
#elif (BITWIDTH == 64)
typedef double token_t;
#endif
#endif

typedef float native_t;

#define MAX_PRINTED_ERRORS 300
#define REL_ERROR_THRESHOLD 0.01

/* <<--params-def-->> */
#define N_CHANNELS 2
#define FEATURE_MAP_HEIGHT 6
#define FEATURE_MAP_WIDTH 6
#define N_FILTERS 2
#define FILTER_DIM 3
#define IS_PADDED 1
#define STRIDE 1
#define DO_RELU 0
#define POOL_TYPE 0
#define BATCH_SIZE 1

#define NACC 1
#define ACC_TLB_ENTRIES 128
#define ACC_PAGE_SIZE (1 << 20)
#define MAX_SIZE (ACC_PAGE_SIZE * ACC_TLB_ENTRIES)
#define MAX_TESTS 10
#define DMA_RATIO 2

struct esp_access esp2 =
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

struct conv2d_stratus_access conv2d_cfg_000[] = {{
    .esp = esp2,

    /* <<--descriptor-->> */
    .n_channels = N_CHANNELS,
    .feature_map_height = FEATURE_MAP_HEIGHT,
    .feature_map_width = FEATURE_MAP_WIDTH,
    .n_filters = N_FILTERS,
    .filter_dim = FILTER_DIM,
    .is_padded = IS_PADDED,
    .stride = STRIDE,
    .do_relu = DO_RELU,
    .pool_type = POOL_TYPE,
    .batch_size = BATCH_SIZE,
    .src_offset = 0,
    .dst_offset = 0,
}};

esp_thread_info_t cfg_conv2d[] = {{
    .run = true,
    .devname = "conv2d_stratus.0",
    .devname_noid = "conv2d_stratus",
    .puffinname = "Green_Conv2d",
    .hw_buf = NULL,
    .ioctl_req = CONV2D_STRATUS_IOC_ACCESS,
    .esp_desc = &(conv2d_cfg_000[0].esp),
    .fd = 0,
    .hw_ns = 0,
}};

#endif /* _CFG_CONV2D_H_ */
