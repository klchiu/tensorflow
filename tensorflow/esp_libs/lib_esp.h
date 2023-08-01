#ifndef _LIB_ESP_H_
#define _LIB_ESP_H_

#include <stdio.h>
#include <math.h>

#include "fixed_point.h"

// [humu]: types.h
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed short int int16_t;
typedef unsigned short int uint16_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;

// [humu]: esp_accelerator.h
enum accelerator_coherence {ACC_COH_NONE = 0, ACC_COH_LLC, ACC_COH_RECALL, ACC_COH_FULL, ACC_COH_AUTO};


// [humu]: contig_alloc.h
struct contig_khandle_struct {
	char unused;
};
typedef struct contig_khandle_struct *contig_khandle_t;

enum contig_alloc_policy {
	CONTIG_ALLOC_PREFERRED,
	CONTIG_ALLOC_LEAST_LOADED,
	CONTIG_ALLOC_BALANCED,
};


// [humu]: esp.h
struct esp_access {
	contig_khandle_t contig;
	uint8_t run;
	uint8_t p2p_store;
	uint8_t p2p_nsrcs;
	char p2p_srcs[4][64];
	enum accelerator_coherence coherence;
    unsigned int footprint;
    enum contig_alloc_policy alloc_policy;
    unsigned int ddr_node;
	unsigned int in_place;
	unsigned int reuse_factor;
};

// [humu]: libesp.h
typedef struct esp_accelerator_thread_info {
	bool run;
	char *devname;
	char *devname_noid;
	char *puffinname;
	void *hw_buf;
	int ioctl_req;
	/* Partially Filled-in by ESPLIB */
	struct esp_access *esp_desc;
	/* Filled-in by ESPLIB */
	int fd;
	unsigned long long hw_ns;
} esp_thread_info_t;



#endif /* _LIB_ESP_H_ */
