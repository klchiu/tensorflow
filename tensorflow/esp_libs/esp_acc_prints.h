#ifndef _ESP_ACC_PRINTS_H_
#define _ESP_ACC_PRINTS_H_



#include "lib_esp.h"

#include "cfg_conv2d.h"
#include "cfg_gemm.h"
#include "cfg_tf_add3.h"
#include "cfg_tf_mult3.h"
#include "cfg_tf_sub3.h"


void print_conv2d_cfg(esp_thread_info_t* thread_cfg, conv2d_stratus_access* cfg) {
  fprintf(stderr, "---------------- conv2d ACC Config ----------------\n");
  fprintf(stderr, "    devname =            %s\n", thread_cfg->devname);
  fprintf(stderr, "    n_channels =         %d\n", cfg->n_channels);
  fprintf(stderr, "    feature_map_height = %d\n", cfg->feature_map_height);
  fprintf(stderr, "    feature_map_width =  %d\n", cfg->feature_map_width);
  fprintf(stderr, "    n_filters =          %d\n", cfg->n_filters);
  fprintf(stderr, "    filter_dim =         %d\n", cfg->filter_dim);
  fprintf(stderr, "    is_padded =          %d\n", cfg->is_padded);
  fprintf(stderr, "    stride =             %d\n", cfg->stride);
  fprintf(stderr, "    do_relu =            %d\n", cfg->do_relu);
  fprintf(stderr, "    pool_type =          %d\n", cfg->pool_type);
  fprintf(stderr, "    batch_size =         %d\n", cfg->batch_size);
}

void print_gemm_cfg(esp_thread_info_t* thread_cfg, gemm_stratus_access* cfg) {
  fprintf(stderr, "---------------- gemm ACC Config ----------------\n");
  fprintf(stderr, "    devname =            %s\n", thread_cfg->devname);
  fprintf(stderr, "    do_relu =            %d\n", cfg->do_relu);
  fprintf(stderr, "    transpose =          %d\n", cfg->transpose);
  fprintf(stderr, "    ninputs =            %d\n", cfg->ninputs);
  fprintf(stderr, "    d3 =                 %d\n", cfg->d3);
  fprintf(stderr, "    d2 =                 %d\n", cfg->d2);
  fprintf(stderr, "    d1 =                 %d\n", cfg->d1);
  fprintf(stderr, "    st_offset =          %d\n", cfg->st_offset);
  fprintf(stderr, "    ld_offset1 =         %d\n", cfg->ld_offset1);
  fprintf(stderr, "    ld_offset2 =         %d\n", cfg->ld_offset2);
}

void print_tf_add3_cfg(esp_thread_info_t* thread_cfg, tf_add3_stratus_access* cfg) {
  fprintf(stderr, "---------------- tf_add3 ACC Config ----------------\n");
  fprintf(stderr, "    devname =              %s\n", thread_cfg->devname);
  fprintf(stderr, "    tf_length =            %d\n", cfg->tf_length);
  fprintf(stderr, "    tf_src_dst_offset_0 =  %d\n", cfg->tf_src_dst_offset_0);
  fprintf(stderr, "    tf_src_dst_offset_1 =  %d\n", cfg->tf_src_dst_offset_1);
  fprintf(stderr, "    tf_src_dst_offset_2 =  %d\n", cfg->tf_src_dst_offset_2);
}

void print_tf_sub3_cfg(esp_thread_info_t* thread_cfg, tf_sub3_stratus_access* cfg) {
  fprintf(stderr, "---------------- tf_sub3 ACC Config ----------------\n");
  fprintf(stderr, "    devname =              %s\n", thread_cfg->devname);
  fprintf(stderr, "    tf_length =            %d\n", cfg->tf_length);
  fprintf(stderr, "    tf_src_dst_offset_0 =  %d\n", cfg->tf_src_dst_offset_0);
  fprintf(stderr, "    tf_src_dst_offset_1 =  %d\n", cfg->tf_src_dst_offset_1);
  fprintf(stderr, "    tf_src_dst_offset_2 =  %d\n", cfg->tf_src_dst_offset_2);
}

void print_tf_mult3_cfg(esp_thread_info_t* thread_cfg, tf_mult3_stratus_access* cfg) {
  fprintf(stderr, "---------------- tf_mult3 ACC Config ----------------\n");
  fprintf(stderr, "    devname =              %s\n", thread_cfg->devname);
  fprintf(stderr, "    tf_length =            %d\n", cfg->tf_length);
  fprintf(stderr, "    tf_src_dst_offset_0 =  %d\n", cfg->tf_src_dst_offset_0);
  fprintf(stderr, "    tf_src_dst_offset_1 =  %d\n", cfg->tf_src_dst_offset_1);
  fprintf(stderr, "    tf_src_dst_offset_2 =  %d\n", cfg->tf_src_dst_offset_2);
}




#endif /* _ESP_ACC_PRINTS_H_ */
