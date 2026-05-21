// @trace-pilot 28990d3c9cbc0614fc7ecfc4c9d3652caba43119
#include <assert.h>
#include <stdio.h>

#include "config.h"

int main(void) {
    Config config = {
        .dim = 64,
        .hidden_dim = 256,
        .n_layers = 4,
        .n_heads = 8,
        .n_kv_heads = 4,
        .vocab_size = 32000,
        .seq_len = 128,
    };

    assert(config_is_valid(&config));
    assert(config_head_size(&config) == 8);
    assert(config_kv_dim(&config) == 32);
    assert(config_kv_mul(&config) == 2);

    return 0;
}
