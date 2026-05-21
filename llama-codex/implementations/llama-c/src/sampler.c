// @trace-pilot f919fe249838754017b20d4282f20c828bca2abb
#include "sampler.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

static int sample_argmax(const float *logits, int vocab_size) {
    int i;
    int best = 0;

    for (i = 1; i < vocab_size; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }

    return best;
}

static uint32_t random_u32(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return (uint32_t)((x * 2685821657736338717ULL) >> 32);
}

static float random_f32(uint64_t *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

bool sampler_init(Sampler *sampler, int vocab_size, float temperature, float top_p, uint64_t seed) {
    if (sampler == NULL || vocab_size <= 0) {
        return false;
    }

    *sampler = (Sampler){0};
    sampler->probabilities = calloc((size_t)vocab_size, sizeof(float));
    if (sampler->probabilities == NULL) {
        return false;
    }

    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->top_p = top_p;
    sampler->rng_state = seed == 0 ? 1 : seed;
    return true;
}

void sampler_free(Sampler *sampler) {
    if (sampler == NULL) {
        return;
    }

    free(sampler->probabilities);
    *sampler = (Sampler){0};
}

int sampler_sample(Sampler *sampler, const float *logits) {
    int i;
    float max_logit;
    float sum;
    float coin;
    float cdf;

    if (sampler == NULL || logits == NULL) {
        return -1;
    }

    if (sampler->temperature <= 0.0f) {
        return sample_argmax(logits, sampler->vocab_size);
    }

    max_logit = logits[0] / sampler->temperature;
    for (i = 1; i < sampler->vocab_size; ++i) {
        float scaled = logits[i] / sampler->temperature;
        if (scaled > max_logit) {
            max_logit = scaled;
        }
    }

    sum = 0.0f;
    for (i = 0; i < sampler->vocab_size; ++i) {
        sampler->probabilities[i] = expf(logits[i] / sampler->temperature - max_logit);
        sum += sampler->probabilities[i];
    }

    if (sum <= 0.0f) {
        return sample_argmax(logits, sampler->vocab_size);
    }

    for (i = 0; i < sampler->vocab_size; ++i) {
        sampler->probabilities[i] /= sum;
    }

    coin = random_f32(&sampler->rng_state);
    cdf = 0.0f;
    for (i = 0; i < sampler->vocab_size; ++i) {
        cdf += sampler->probabilities[i];
        if (coin <= cdf) {
            return i;
        }
    }

    return sampler->vocab_size - 1;
}
