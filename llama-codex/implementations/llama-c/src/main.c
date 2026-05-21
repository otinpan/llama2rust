#include <stdio.h>
#include <stdlib.h>
// @trace-pilot 87d5a6685da2645694422992457d5d56395cea41
#include <string.h>
#include <time.h>

#include "llama.h"

static int print_usage(const char *argv0) {
// @trace-pilot 87d5a6685da2645694422992457d5d56395cea41
    fprintf(stderr, "usage: %s <model.bin> <tokenizer.bin> -i <prompt> [-n steps] [--temperature value] [--top-p value]\n", argv0);
    return 1;
}

int main(int argc, char **argv) {
    Transformer transformer;
// @trace-pilot 87d5a6685da2645694422992457d5d56395cea41
    Tokenizer tokenizer;
    Sampler sampler;
    const char *model_path;
    const char *tokenizer_path;
    const char *prompt;
    int steps;
    float temperature;
    float top_p;
    int i;

// @trace-pilot 87d5a6685da2645694422992457d5d56395cea41
    if (argc < 3) {
        return print_usage(argv[0]);
    }

// @trace-pilot 87d5a6685da2645694422992457d5d56395cea41
    model_path = argv[1];
    tokenizer_path = argv[2];
    prompt = "";
// @trace-pilot b8838f6c1d8d4361920378d127d4bbad4e7a08c8
    steps = 256;
    temperature = 0.0f;
    top_p = 0.9f;

    for (i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = strtof(argv[++i], NULL);
        } else {
            return print_usage(argv[0]);
        }
    }

// @trace-pilot 87d5a6685da2645694422992457d5d56395cea41
    if (steps <= 0) {
        fprintf(stderr, "steps must be > 0\n");
        return 1;
    }

    if (!transformer_init(&transformer)) {
        fprintf(stderr, "failed to initialize transformer\n");
        return 1;
    }

// @trace-pilot 87d5a6685da2645694422992457d5d56395cea41
    if (!tokenizer_init(&tokenizer)) {
        fprintf(stderr, "failed to initialize tokenizer\n");
        transformer_free(&transformer);
        return 1;
    }

// @trace-pilot 87d5a6685da2645694422992457d5d56395cea41
    if (!transformer_load(&transformer, model_path)) {
        fprintf(stderr, "failed to load model: %s\n", model_path);
        tokenizer_free(&tokenizer);
        transformer_free(&transformer);
        return 1;
    }

// @trace-pilot 87d5a6685da2645694422992457d5d56395cea41
    if (!tokenizer_load(&tokenizer, tokenizer_path, transformer.config.vocab_size)) {
        fprintf(stderr, "failed to load tokenizer: %s\n", tokenizer_path);
        tokenizer_free(&tokenizer);
        transformer_free(&transformer);
        return 1;
    }

// @trace-pilot 87d5a6685da2645694422992457d5d56395cea41
    if (!sampler_init(&sampler, transformer.config.vocab_size, temperature, top_p, (uint64_t)time(NULL))) {
        fprintf(stderr, "failed to initialize sampler\n");
        tokenizer_free(&tokenizer);
        transformer_free(&transformer);
        return 1;
    }

    if (!generate_text(&transformer, &tokenizer, &sampler, prompt, steps)) {
        fprintf(stderr, "generation failed\n");
        sampler_free(&sampler);
        tokenizer_free(&tokenizer);
        transformer_free(&transformer);
        return 1;
    }

    sampler_free(&sampler);
    tokenizer_free(&tokenizer);
    transformer_free(&transformer);
    return 0;
}
