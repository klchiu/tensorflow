/*
 * Copyright (c) 2011-2023 Columbia University, System Level Design Group
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * esp_api_include.h
 * only the code from TF should include this file
 */


#ifndef _ESP_API_INCLUDE_H_
#define _ESP_API_INCLUDE_H_



#include "lib_esp.h"

typedef unsigned long size_t;


extern "C"
{
    void *esp_alloc(size_t size);
    void esp_run(esp_thread_info_t cfg[], unsigned nacc);
    void esp_free(void *buf);
    void esp_dummy(void *buf);
}



#endif /* _ESP_API_INCLUDE_H_ */
