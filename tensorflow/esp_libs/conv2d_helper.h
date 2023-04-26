#ifndef _CONV2D_HELPER_H_
#define _CONV2D_HELPER_H_

#include "lib_esp.h"
#include "cfg_conv2d.h"

// [humu]: from esp_accelerator.h
#ifndef __KERNEL__
#define __round_mask(x, y) ((y)-1)
#define round_up(x, y) ((((x)-1) | __round_mask(x, y))+1)
#endif


// [humu]: helper functions

static void conv2d_init_parameters(int test, int32_t n_channels, int32_t feature_map_height, int32_t feature_map_width,
                            int32_t n_filters, int32_t filter_dim, int32_t is_padded, int32_t stride, int32_t do_relu,
                            int32_t pool_type, int32_t batch_size, unsigned *in_len, unsigned *weights_len,
                            unsigned *bias_len, unsigned *out_len, unsigned *in_size, unsigned *weights_size,
                            unsigned *bias_size, unsigned *out_size, unsigned *weights_offset, unsigned *bias_offset,
                            unsigned *out_offset, unsigned *size)
{
    int32_t output_h;
    // int32_t output_w;
    int32_t output_pool_h;
    // int32_t output_pool_w;
    int32_t pad_dim = 0;

    if (is_padded) {
        pad_dim = filter_dim / 2;
    }

    output_h      = (feature_map_height + 2 * pad_dim - ((filter_dim - 1) + 1)) / stride + 1;
    output_pool_h = pool_type ? output_h / 2 : output_h;

    // Input data and golden output (aligned to DMA_WIDTH makes your life easier)
    *in_len = round_up(
        batch_size * round_up(n_channels * round_up(feature_map_height * feature_map_width, DMA_RATIO), DMA_RATIO),
        DMA_RATIO);
    *weights_len = round_up(n_filters * n_channels * filter_dim * filter_dim, DMA_RATIO);
    *bias_len    = round_up(n_filters, DMA_RATIO);
    *out_len     = round_up(
        batch_size * round_up(n_filters * round_up(output_pool_h * output_pool_h, DMA_RATIO), DMA_RATIO), DMA_RATIO);

    *in_size        = *in_len * sizeof(token_t);
    *weights_size   = *weights_len * sizeof(token_t);
    *bias_size      = *bias_len * sizeof(token_t);
    *out_size       = *out_len * sizeof(token_t);
    *weights_offset = *in_len;
    *bias_offset    = *in_len + *weights_len;
    *out_offset     = *in_len + *weights_len + *bias_len;
    *size           = *in_size + *weights_size + *bias_size + *out_size;

    conv2d_cfg_000[0].n_channels         = n_channels;
    conv2d_cfg_000[0].feature_map_height = feature_map_height;
    conv2d_cfg_000[0].feature_map_width  = feature_map_width;
    conv2d_cfg_000[0].n_filters          = n_filters;
    conv2d_cfg_000[0].filter_dim         = filter_dim;
    conv2d_cfg_000[0].is_padded          = is_padded;
    conv2d_cfg_000[0].stride             = stride;
    conv2d_cfg_000[0].do_relu            = do_relu;
    conv2d_cfg_000[0].pool_type          = pool_type;
    conv2d_cfg_000[0].batch_size         = batch_size;

    // print test info
    printf("  Prepare test %d parameters\n", test);
    printf("    .n_channels = %d\n", n_channels);
    printf("    .feature_map_height = %d\n", feature_map_height);
    printf("    .feature_map_width = %d\n", feature_map_width);
    printf("    .n_filters = %d\n", n_filters);
    printf("    .filter_dim = %d\n", filter_dim);
    printf("    .is_padded = %d\n", is_padded);
    printf("    .stride = %d\n", stride);
    printf("    .do_relu = %d\n", do_relu);
    printf("    .pool_type = %d\n", pool_type);
    printf("    .batch_size = %d\n", batch_size);
}


#endif /* _CONV2D_HELPER_H_ */
