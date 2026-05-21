// @trace-pilot 0a93042b1acdd0846383b92c73f488584310b601
#ifndef LLAMA_C_CONFIG_H
#define LLAMA_C_CONFIG_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t vocab_size;
    int32_t seq_len;
} Config;

bool config_init(Config *config);
bool config_is_valid(const Config *config);
bool config_read(FILE *file, Config *config);

int32_t config_head_size(const Config *config);
int32_t config_kv_dim(const Config *config);
int32_t config_kv_mul(const Config *config);

void config_print(const Config *config, FILE *out);

#endif
