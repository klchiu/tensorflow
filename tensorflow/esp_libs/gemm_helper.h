#ifndef _GEMM_HELPER_H_
#define _GEMM_HELPER_H_

#include "lib_esp.h"
#include "cfg_gemm.h"

// [humu]: from esp_accelerator.h
#ifndef __KERNEL__
#define __round_mask(x, y) ((y)-1)
#define round_up(x, y) ((((x)-1) | __round_mask(x, y))+1)
#endif

typedef int token_t;


// [humu]: helper functions
static unsigned DMA_WORD_PER_BEAT(unsigned _st)
{
        return (sizeof(void *) / _st);
}

static void gemm_init_parameters(int test, int32_t do_relu, int32_t transpose, int32_t ninputs,
			    int32_t d3, int32_t d2, int32_t d1,
			    unsigned *in_len, unsigned *in1_len, unsigned *out_len,
			    unsigned *in_size, unsigned *out_size, unsigned *size)
{
    int32_t ld_offset1, ld_offset2, st_offset;
    unsigned in2_len;
    
    *in1_len = round_up(ninputs * d1 * d2, DMA_WORD_PER_BEAT(sizeof(token_t)));
    in2_len = round_up(ninputs * d2 * d3, DMA_WORD_PER_BEAT(sizeof(token_t)));
    *in_len = *in1_len + in2_len;
    *out_len = round_up(ninputs * d1 * d3, DMA_WORD_PER_BEAT(sizeof(token_t)));
    *in_size = *in_len * sizeof(token_t);
    *out_size = *out_len * sizeof(token_t);
    *size = *in_size + *out_size;

    ld_offset1 = 0;
    ld_offset2 = *in1_len;
    st_offset = *in_len;

    gemm_cfg_000[0].do_relu = do_relu;
    gemm_cfg_000[0].transpose = transpose;
    gemm_cfg_000[0].ninputs = ninputs;
    gemm_cfg_000[0].d1 = d1;
    gemm_cfg_000[0].d2 = d2;
    gemm_cfg_000[0].d3 = d3;
    gemm_cfg_000[0].ld_offset1 = ld_offset1;
    gemm_cfg_000[0].ld_offset2 = ld_offset2;
    gemm_cfg_000[0].st_offset = st_offset;

    // print test info
    printf("  Prepare test %d parameters\n", test);
    printf("    .do_relu = %d\n", do_relu);
    printf("    .transpose = %d\n", transpose);
    printf("    .ninputs = %d\n", ninputs);
    printf("    .d3 = %d\n", d3);
    printf("    .d2 = %d\n", d2);
    printf("    .d1 = %d\n", d1);
    printf("    .st_offset = %d\n", st_offset);
    printf("    .ld_offset1 = %d\n", ld_offset1);
    printf("    .ld_offset2 = %d\n", ld_offset2);
}


#endif /* _GEMM_HELPER_H_ */
