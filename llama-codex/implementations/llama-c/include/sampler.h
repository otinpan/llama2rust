// @trace-pilot f919fe249838754017b20d4282f20c828bca2abb
#ifndef LLAMA_C_SAMPLER_H
#define LLAMA_C_SAMPLER_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    int vocab_size;
    float temperature;
    float top_p;
    uint64_t rng_state;
    float *probabilities;
} Sampler;

bool sampler_init(Sampler *sampler, int vocab_size, float temperature, float top_p, uint64_t seed);
void sampler_free(Sampler *sampler);
int sampler_sample(Sampler *sampler, const float *logits);

#endif
