// @trace-pilot 0a93042b1acdd0846383b92c73f488584310b601
#include "config.h"

#include <inttypes.h>
#include <stddef.h>

bool config_init(Config *config) {
    if (config == NULL) {
        return false;
    }

    *config = (Config){0};
    return true;
}

bool config_is_valid(const Config *config) {
    if (config == NULL) {
        return false;
    }

    if (config->dim <= 0 ||
        config->hidden_dim <= 0 ||
        config->n_layers <= 0 ||
        config->n_heads <= 0 ||
        config->n_kv_heads <= 0 ||
        config->vocab_size <= 0 ||
        config->seq_len <= 0) {
        return false;
    }

    if (config->dim % config->n_heads != 0) {
        return false;
    }

    if (config->n_heads % config->n_kv_heads != 0) {
        return false;
    }

    return true;
}

bool config_read(FILE *file, Config *config) {
    Config header;

    if (file == NULL || config == NULL) {
        return false;
    }

    if (fread(&header, sizeof(header), 1, file) != 1) {
        return false;
    }

    if (!config_is_valid(&header)) {
        return false;
    }

    *config = header;
    return true;
}

int32_t config_head_size(const Config *config) {
    if (!config_is_valid(config)) {
        return 0;
    }

    return config->dim / config->n_heads;
}

int32_t config_kv_dim(const Config *config) {
    if (!config_is_valid(config)) {
        return 0;
    }

    return (config->dim * config->n_kv_heads) / config->n_heads;
}

int32_t config_kv_mul(const Config *config) {
    if (!config_is_valid(config)) {
        return 0;
    }

    return config->n_heads / config->n_kv_heads;
}

void config_print(const Config *config, FILE *out) {
    FILE *stream = out;

    if (!config_is_valid(config)) {
        return;
    }

    if (stream == NULL) {
        stream = stdout;
    }

    fprintf(stream, "dim=%" PRId32 "\n", config->dim);
    fprintf(stream, "hidden_dim=%" PRId32 "\n", config->hidden_dim);
    fprintf(stream, "n_layers=%" PRId32 "\n", config->n_layers);
    fprintf(stream, "n_heads=%" PRId32 "\n", config->n_heads);
    fprintf(stream, "n_kv_heads=%" PRId32 "\n", config->n_kv_heads);
    fprintf(stream, "vocab_size=%" PRId32 "\n", config->vocab_size);
    fprintf(stream, "seq_len=%" PRId32 "\n", config->seq_len);
    fprintf(stream, "head_size=%" PRId32 "\n", config_head_size(config));
    fprintf(stream, "kv_dim=%" PRId32 "\n", config_kv_dim(config));
    fprintf(stream, "kv_mul=%" PRId32 "\n", config_kv_mul(config));
}
