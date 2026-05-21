// @trace-pilot abb386e0dd72e969b4685301a151d818cead844d
#ifndef LLAMA_C_WEIGHTS_H
#define LLAMA_C_WEIGHTS_H

#include <stdbool.h>
#include <stddef.h>

#include "config.h"

typedef struct {
    const float *token_embedding_table;
    const float *rms_att_weight;
    const float *wq;
    const float *wk;
    const float *wv;
    const float *wo;
    const float *rms_ffn_weight;
    const float *w1;
    const float *w2;
    const float *w3;
    const float *rms_final_weight;
    const float *freq_cis_real;
    const float *freq_cis_imag;
    const float *wcls;
    void *mapped_data;
    size_t mapped_size;
} TransformerWeights;

bool weights_init(TransformerWeights *weights);
bool weights_map(TransformerWeights *weights, const Config *config, const char *path);
void weights_free(TransformerWeights *weights);

size_t weights_file_size_min(const Config *config);
size_t weights_file_size_with_classifier(const Config *config);

#endif
