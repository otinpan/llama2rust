// @trace-pilot 086689ce75ee5f7bbc16dc377a2bb4cad8598b7b
#ifndef LLAMA_C_TRANSFORMER_H
#define LLAMA_C_TRANSFORMER_H

#include <stdbool.h>
#include <stdint.h>

#include "config.h"
#include "weights.h"

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
} Transformer;

bool run_state_init(RunState *state, const Config *config);
void run_state_free(RunState *state);

bool transformer_init(Transformer *transformer);
bool transformer_load(Transformer *transformer, const char *model_path);
void transformer_free(Transformer *transformer);

bool transformer_forward(Transformer *transformer, int token, int position);

#endif
