// @trace-pilot abb386e0dd72e969b4685301a151d818cead844d
#include "weights.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static bool checked_mul_size(size_t a, size_t b, size_t *out) {
    if (out == NULL) {
        return false;
    }

    if (a != 0 && b > SIZE_MAX / a) {
        return false;
    }

    *out = a * b;
    return true;
}

static bool checked_add_size(size_t a, size_t b, size_t *out) {
    if (out == NULL) {
        return false;
    }

    if (b > SIZE_MAX - a) {
        return false;
    }

    *out = a + b;
    return true;
}

static bool add_tensor_bytes(size_t *total, size_t count) {
    size_t bytes;
    size_t next_total;

    if (!checked_mul_size(count, sizeof(float), &bytes)) {
        return false;
    }

    if (!checked_add_size(*total, bytes, &next_total)) {
        return false;
    }

    *total = next_total;
    return true;
}

static bool checked_mul3_size(size_t a, size_t b, size_t c, size_t *out) {
    size_t ab;

    if (!checked_mul_size(a, b, &ab)) {
        return false;
    }

    return checked_mul_size(ab, c, out);
}

static bool weights_core_size(const Config *config, size_t *out) {
    size_t total;
    size_t dim;
    size_t hidden_dim;
    size_t n_layers;
    size_t vocab_size;
    size_t seq_len;
    size_t head_size;
    size_t kv_dim;
    size_t count;

    if (!config_is_valid(config) || out == NULL) {
        return false;
    }

    total = sizeof(Config);
    dim = (size_t)config->dim;
    hidden_dim = (size_t)config->hidden_dim;
    n_layers = (size_t)config->n_layers;
    vocab_size = (size_t)config->vocab_size;
    seq_len = (size_t)config->seq_len;
    head_size = (size_t)config_head_size(config);
    kv_dim = (size_t)config_kv_dim(config);

    if (!checked_mul_size(vocab_size, dim, &count) || !add_tensor_bytes(&total, count) ||
        !checked_mul_size(n_layers, dim, &count) || !add_tensor_bytes(&total, count) ||
        !checked_mul3_size(n_layers, dim, dim, &count) || !add_tensor_bytes(&total, count) ||
        !checked_mul3_size(n_layers, dim, kv_dim, &count) || !add_tensor_bytes(&total, count) ||
        !checked_mul3_size(n_layers, dim, kv_dim, &count) || !add_tensor_bytes(&total, count) ||
        !checked_mul3_size(n_layers, dim, dim, &count) || !add_tensor_bytes(&total, count) ||
        !checked_mul_size(n_layers, dim, &count) || !add_tensor_bytes(&total, count) ||
        !checked_mul3_size(n_layers, hidden_dim, dim, &count) || !add_tensor_bytes(&total, count) ||
        !checked_mul3_size(n_layers, dim, hidden_dim, &count) || !add_tensor_bytes(&total, count) ||
        !checked_mul3_size(n_layers, hidden_dim, dim, &count) || !add_tensor_bytes(&total, count) ||
        !add_tensor_bytes(&total, dim) ||
        !checked_mul_size(seq_len, head_size / 2, &count) || !add_tensor_bytes(&total, count) ||
// @trace-pilot abb386e0dd72e969b4685301a151d818cead844d
        !checked_mul_size(seq_len, head_size / 2, &count) || !add_tensor_bytes(&total, count)) {
        return false;
    }

    *out = total;
    return true;
}

static bool map_tensor(const float **tensor, const float **cursor, size_t count) {
    const float *start;

    if (tensor == NULL || cursor == NULL || *cursor == NULL) {
        return false;
    }

    start = *cursor;
    *tensor = start;
    *cursor = start + count;
    return true;
}

static bool weights_assign_pointers(TransformerWeights *weights, const Config *config) {
    const float *cursor;
    size_t dim;
    size_t hidden_dim;
    size_t n_layers;
    size_t vocab_size;
    size_t seq_len;
    size_t head_size;
    size_t kv_dim;
    size_t core_size;
    size_t classifier_bytes;
    size_t count;

    if (weights == NULL || !config_is_valid(config) || weights->mapped_data == NULL) {
        return false;
    }

    dim = (size_t)config->dim;
    hidden_dim = (size_t)config->hidden_dim;
    n_layers = (size_t)config->n_layers;
    vocab_size = (size_t)config->vocab_size;
    seq_len = (size_t)config->seq_len;
    head_size = (size_t)config_head_size(config);
    kv_dim = (size_t)config_kv_dim(config);
    cursor = (const float *)((const unsigned char *)weights->mapped_data + sizeof(Config));

    if (!checked_mul_size(vocab_size, dim, &count) || !map_tensor(&weights->token_embedding_table, &cursor, count) ||
        !checked_mul_size(n_layers, dim, &count) || !map_tensor(&weights->rms_att_weight, &cursor, count) ||
        !checked_mul3_size(n_layers, dim, dim, &count) || !map_tensor(&weights->wq, &cursor, count) ||
        !checked_mul3_size(n_layers, dim, kv_dim, &count) || !map_tensor(&weights->wk, &cursor, count) ||
        !checked_mul3_size(n_layers, dim, kv_dim, &count) || !map_tensor(&weights->wv, &cursor, count) ||
        !checked_mul3_size(n_layers, dim, dim, &count) || !map_tensor(&weights->wo, &cursor, count) ||
        !checked_mul_size(n_layers, dim, &count) || !map_tensor(&weights->rms_ffn_weight, &cursor, count) ||
        !checked_mul3_size(n_layers, hidden_dim, dim, &count) || !map_tensor(&weights->w1, &cursor, count) ||
        !checked_mul3_size(n_layers, dim, hidden_dim, &count) || !map_tensor(&weights->w2, &cursor, count) ||
        !checked_mul3_size(n_layers, hidden_dim, dim, &count) || !map_tensor(&weights->w3, &cursor, count) ||
        !map_tensor(&weights->rms_final_weight, &cursor, dim) ||
        !checked_mul_size(seq_len, head_size / 2, &count) || !map_tensor(&weights->freq_cis_real, &cursor, count) ||
        !checked_mul_size(seq_len, head_size / 2, &count) || !map_tensor(&weights->freq_cis_imag, &cursor, count)) {
        return false;
    }

    if (!weights_core_size(config, &core_size)) {
        return false;
    }

    if (!checked_mul3_size(vocab_size, dim, sizeof(float), &classifier_bytes)) {
        return false;
    }
    if (weights->mapped_size == core_size + classifier_bytes) {
        weights->wcls = cursor;
    } else {
        weights->wcls = weights->token_embedding_table;
    }

    return true;
}

bool weights_init(TransformerWeights *weights) {
    if (weights == NULL) {
        return false;
    }

    *weights = (TransformerWeights){0};
    return true;
}

bool weights_map(TransformerWeights *weights, const Config *config, const char *path) {
    int fd;
    struct stat st;
    void *mapped;
    size_t min_size;
    size_t max_size;

    if (weights == NULL || path == NULL || !config_is_valid(config)) {
        return false;
    }

    weights_free(weights);

    if (!weights_core_size(config, &min_size)) {
        return false;
    }

    max_size = weights_file_size_with_classifier(config);
    if (max_size == 0) {
        return false;
    }

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        return false;
    }

    if (fstat(fd, &st) != 0) {
        close(fd);
        return false;
    }

    if (st.st_size < 0) {
        close(fd);
        return false;
    }

    if ((size_t)st.st_size != min_size && (size_t)st.st_size != max_size) {
        close(fd);
        return false;
    }

    mapped = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        return false;
    }

    *weights = (TransformerWeights){
        .mapped_data = mapped,
        .mapped_size = (size_t)st.st_size,
    };

    if (!weights_assign_pointers(weights, config)) {
        weights_free(weights);
        return false;
    }

    return true;
}

void weights_free(TransformerWeights *weights) {
    if (weights == NULL) {
        return;
    }

    if (weights->mapped_data != NULL && weights->mapped_size != 0) {
        munmap(weights->mapped_data, weights->mapped_size);
    }

    *weights = (TransformerWeights){0};
}

size_t weights_file_size_min(const Config *config) {
    size_t size;

    if (!weights_core_size(config, &size)) {
        return 0;
    }

    return size;
}

size_t weights_file_size_with_classifier(const Config *config) {
    size_t base_size;
    size_t classifier_bytes;
    size_t total_size;

    if (!weights_core_size(config, &base_size)) {
        return 0;
    }

    if (!checked_mul3_size((size_t)config->vocab_size, (size_t)config->dim, sizeof(float), &classifier_bytes)) {
        return 0;
    }

    if (!checked_add_size(base_size, classifier_bytes, &total_size)) {
        return 0;
    }

    return total_size;
}
