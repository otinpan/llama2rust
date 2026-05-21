// @trace-pilot 086689ce75ee5f7bbc16dc377a2bb4cad8598b7b
#include "transformer.h"

#include <math.h>
#include <stddef.h>
// @trace-pilot 95fa4c3a36b2a683d68289bd73932c0ea440c89c
#include <string.h>
#include <stdlib.h>

static void *checked_calloc(size_t count, size_t size) {
    if (count == 0 || size == 0) {
        return NULL;
    }

    if (count > SIZE_MAX / size) {
        return NULL;
    }

    return calloc(count, size);
}

static bool run_state_alloc(float **buffer, size_t count) {
    if (buffer == NULL) {
        return false;
    }

    *buffer = checked_calloc(count, sizeof(float));
    return *buffer != NULL;
}

// @trace-pilot 28990d3c9cbc0614fc7ecfc4c9d3652caba43119
static void matmul(float *out, const float *x, const float *w, int n, int d) {
    int i;
    int j;

    for (i = 0; i < d; ++i) {
        float sum = 0.0f;
        const float *row = w + ((size_t)i * (size_t)n);
        for (j = 0; j < n; ++j) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

// @trace-pilot 95fa4c3a36b2a683d68289bd73932c0ea440c89c
static void softmax_inplace(float *x, int size) {
    int i;
    float max_val;
    float sum;

    if (size <= 0) {
        return;
    }

    max_val = x[0];
    for (i = 1; i < size; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    sum = 0.0f;
    for (i = 0; i < size; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    if (sum == 0.0f) {
        return;
    }

    for (i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

static void rmsnorm(float *out, const float *x, const float *weight, int size) {
    int i;
    float ss = 0.0f;
    float inv_rms;

    for (i = 0; i < size; ++i) {
        ss += x[i] * x[i];
    }

    ss /= (float)size;
    inv_rms = 1.0f / sqrtf(ss + 1e-5f);

    for (i = 0; i < size; ++i) {
        out[i] = weight[i] * (x[i] * inv_rms);
    }
}

static void copy_embedding(float *out, const TransformerWeights *weights, const Config *config, int token) {
    int i;
    const float *embedding = weights->token_embedding_table + ((size_t)token * (size_t)config->dim);

    for (i = 0; i < config->dim; ++i) {
        out[i] = embedding[i];
    }
}

// @trace-pilot 95fa4c3a36b2a683d68289bd73932c0ea440c89c
static float dot(const float *a, const float *b, int size) {
    int i;
    float sum = 0.0f;

    for (i = 0; i < size; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

static void accum(float *dst, const float *src, int size) {
    int i;

    for (i = 0; i < size; ++i) {
        dst[i] += src[i];
    }
}

static void swiglu(float *out, const float *gate, const float *up, int size) {
    int i;

    for (i = 0; i < size; ++i) {
        float x = gate[i];
        float silu = x / (1.0f + expf(-x));
        out[i] = silu * up[i];
    }
}

static void apply_rope(float *vec, int head_size, int position, const float *freq_real, const float *freq_imag) {
    int i;
    const float *real = freq_real + ((size_t)position * (size_t)(head_size / 2));
    const float *imag = freq_imag + ((size_t)position * (size_t)(head_size / 2));

    for (i = 0; i < head_size; i += 2) {
        int j = i / 2;
        float v0 = vec[i];
        float v1 = vec[i + 1];
        float fcr = real[j];
        float fci = imag[j];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
    }
}

static void apply_rope_multi(float *vec, int n_heads, int head_size, int position, const float *freq_real, const float *freq_imag) {
    int h;

    for (h = 0; h < n_heads; ++h) {
        apply_rope(vec + ((size_t)h * (size_t)head_size), head_size, position, freq_real, freq_imag);
    }
}

static void run_state_clear(RunState *state) {
    if (state == NULL) {
        return;
    }

    *state = (RunState){0};
}

bool run_state_init(RunState *state, const Config *config) {
    size_t dim;
    size_t hidden_dim;
    size_t kv_dim;
    size_t seq_len;
    size_t n_layers;

    if (state == NULL || !config_is_valid(config)) {
        return false;
    }

    run_state_clear(state);

    dim = (size_t)config->dim;
    hidden_dim = (size_t)config->hidden_dim;
    kv_dim = (size_t)config_kv_dim(config);
    seq_len = (size_t)config->seq_len;
    n_layers = (size_t)config->n_layers;

    if (!run_state_alloc(&state->x, dim) ||
        !run_state_alloc(&state->xb, dim) ||
        !run_state_alloc(&state->xb2, dim) ||
        !run_state_alloc(&state->hb, hidden_dim) ||
        !run_state_alloc(&state->hb2, hidden_dim) ||
        !run_state_alloc(&state->q, dim) ||
        !run_state_alloc(&state->k, kv_dim) ||
        !run_state_alloc(&state->v, kv_dim) ||
        !run_state_alloc(&state->att, seq_len * (size_t)config->n_heads) ||
        !run_state_alloc(&state->logits, (size_t)config->vocab_size) ||
        !run_state_alloc(&state->key_cache, n_layers * seq_len * kv_dim) ||
        !run_state_alloc(&state->value_cache, n_layers * seq_len * kv_dim)) {
        run_state_free(state);
        return false;
    }

    return true;
}

void run_state_free(RunState *state) {
    if (state == NULL) {
        return;
    }

    free(state->x);
    free(state->xb);
    free(state->xb2);
    free(state->hb);
    free(state->hb2);
    free(state->q);
    free(state->k);
    free(state->v);
    free(state->att);
    free(state->logits);
    free(state->key_cache);
    free(state->value_cache);

    run_state_clear(state);
}

bool transformer_init(Transformer *transformer) {
    if (transformer == NULL) {
        return false;
    }

    *transformer = (Transformer){0};

    if (!config_init(&transformer->config)) {
        return false;
    }

    if (!weights_init(&transformer->weights)) {
        return false;
    }

    run_state_clear(&transformer->state);
    return true;
}

bool transformer_load(Transformer *transformer, const char *model_path) {
    FILE *file;
    Config config;

    if (transformer == NULL || model_path == NULL) {
        return false;
    }

    transformer_free(transformer);
    if (!transformer_init(transformer)) {
        return false;
    }

    file = fopen(model_path, "rb");
    if (file == NULL) {
        return false;
    }

    if (!config_read(file, &config)) {
        fclose(file);
        return false;
    }

    fclose(file);

    if (!weights_map(&transformer->weights, &config, model_path)) {
        return false;
    }

    if (!run_state_init(&transformer->state, &config)) {
        weights_free(&transformer->weights);
        return false;
    }

    transformer->config = config;
    return true;
}

void transformer_free(Transformer *transformer) {
    if (transformer == NULL) {
        return;
    }

    run_state_free(&transformer->state);
    weights_free(&transformer->weights);
    config_init(&transformer->config);
}

bool transformer_forward(Transformer *transformer, int token, int position) {
    const Config *config;
    const TransformerWeights *weights;
    RunState *state;
// @trace-pilot 95fa4c3a36b2a683d68289bd73932c0ea440c89c
    int dim;
    int hidden_dim;
    int head_size;
    int kv_dim;
    int kv_mul;
    int layer;

    if (transformer == NULL || !config_is_valid(&transformer->config)) {
        return false;
    }

    if (token < 0 || token >= transformer->config.vocab_size) {
        return false;
    }

    if (position < 0 || position >= transformer->config.seq_len) {
        return false;
    }
    // @trace-pilot 28990d3c9cbc0614fc7ecfc4c9d3652caba43119
    config = &transformer->config;
    weights = &transformer->weights;
    state = &transformer->state;

    if (weights->token_embedding_table == NULL ||
// @trace-pilot 95fa4c3a36b2a683d68289bd73932c0ea440c89c
        weights->rms_att_weight == NULL ||
        weights->wq == NULL ||
        weights->wk == NULL ||
        weights->wv == NULL ||
        weights->wo == NULL ||
        weights->rms_ffn_weight == NULL ||
        weights->w1 == NULL ||
        weights->w2 == NULL ||
        weights->w3 == NULL ||
        weights->rms_final_weight == NULL ||
// @trace-pilot 95fa4c3a36b2a683d68289bd73932c0ea440c89c
        weights->freq_cis_real == NULL ||
        weights->freq_cis_imag == NULL ||
        weights->wcls == NULL ||
        state->x == NULL ||
        state->xb == NULL ||
// @trace-pilot 95fa4c3a36b2a683d68289bd73932c0ea440c89c
        state->xb2 == NULL ||
        state->hb == NULL ||
        state->hb2 == NULL ||
        state->q == NULL ||
        state->k == NULL ||
        state->v == NULL ||
        state->att == NULL ||
        state->key_cache == NULL ||
        state->value_cache == NULL ||
        state->logits == NULL) {
        return false;
    }

// @trace-pilot 95fa4c3a36b2a683d68289bd73932c0ea440c89c
    dim = config->dim;
    hidden_dim = config->hidden_dim;
    head_size = config_head_size(config);
    kv_dim = config_kv_dim(config);
    kv_mul = config_kv_mul(config);

    copy_embedding(state->x, weights, config, token);

// @trace-pilot 95fa4c3a36b2a683d68289bd73932c0ea440c89c
    for (layer = 0; layer < config->n_layers; ++layer) {
        const float *rms_att = weights->rms_att_weight + ((size_t)layer * (size_t)dim);
        const float *wq = weights->wq + ((size_t)layer * (size_t)dim * (size_t)dim);
        const float *wk = weights->wk + ((size_t)layer * (size_t)dim * (size_t)kv_dim);
        const float *wv = weights->wv + ((size_t)layer * (size_t)dim * (size_t)kv_dim);
        const float *wo = weights->wo + ((size_t)layer * (size_t)dim * (size_t)dim);
        const float *rms_ffn = weights->rms_ffn_weight + ((size_t)layer * (size_t)dim);
        const float *w1 = weights->w1 + ((size_t)layer * (size_t)hidden_dim * (size_t)dim);
        const float *w2 = weights->w2 + ((size_t)layer * (size_t)dim * (size_t)hidden_dim);
        const float *w3 = weights->w3 + ((size_t)layer * (size_t)hidden_dim * (size_t)dim);
        float *key_cache_row;
        float *value_cache_row;
        int h;

        rmsnorm(state->xb, state->x, rms_att, dim);
        matmul(state->q, state->xb, wq, dim, dim);
        matmul(state->k, state->xb, wk, dim, kv_dim);
        matmul(state->v, state->xb, wv, dim, kv_dim);

        apply_rope_multi(state->q, config->n_heads, head_size, position, weights->freq_cis_real, weights->freq_cis_imag);
        apply_rope_multi(state->k, config->n_kv_heads, head_size, position, weights->freq_cis_real, weights->freq_cis_imag);

        key_cache_row = state->key_cache +
                        (((size_t)layer * (size_t)config->seq_len + (size_t)position) * (size_t)kv_dim);
        value_cache_row = state->value_cache +
                          (((size_t)layer * (size_t)config->seq_len + (size_t)position) * (size_t)kv_dim);
        memcpy(key_cache_row, state->k, (size_t)kv_dim * sizeof(float));
        memcpy(value_cache_row, state->v, (size_t)kv_dim * sizeof(float));

        memset(state->xb, 0, (size_t)dim * sizeof(float));
        for (h = 0; h < config->n_heads; ++h) {
            float *att = state->att + ((size_t)h * (size_t)config->seq_len);
            float *xb_head = state->xb + ((size_t)h * (size_t)head_size);
            const float *q_head = state->q + ((size_t)h * (size_t)head_size);
            int kv_head = h / kv_mul;
            int t;

            for (t = 0; t <= position; ++t) {
                const float *k_t = state->key_cache +
                                   (((size_t)layer * (size_t)config->seq_len + (size_t)t) * (size_t)kv_dim) +
                                   ((size_t)kv_head * (size_t)head_size);
                att[t] = dot(q_head, k_t, head_size) / sqrtf((float)head_size);
            }

            softmax_inplace(att, position + 1);

            for (t = 0; t <= position; ++t) {
                const float *v_t = state->value_cache +
                                   (((size_t)layer * (size_t)config->seq_len + (size_t)t) * (size_t)kv_dim) +
                                   ((size_t)kv_head * (size_t)head_size);
                int i;
                for (i = 0; i < head_size; ++i) {
                    xb_head[i] += att[t] * v_t[i];
                }
            }
        }

        matmul(state->xb2, state->xb, wo, dim, dim);
        accum(state->x, state->xb2, dim);

        rmsnorm(state->xb, state->x, rms_ffn, dim);
        matmul(state->hb, state->xb, w1, dim, hidden_dim);
        matmul(state->hb2, state->xb, w3, dim, hidden_dim);
        swiglu(state->hb, state->hb, state->hb2, hidden_dim);
        matmul(state->xb, state->hb, w2, hidden_dim, dim);
        accum(state->x, state->xb, dim);
    }

    rmsnorm(state->xb, state->x, weights->rms_final_weight, dim);
    matmul(state->logits, state->xb, weights->wcls, dim, config->vocab_size);
    return true;
}
