// @trace-pilot 92fe5e02c071b0c60a204b35d23479e5dfebafbc
#include "generate.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int prompt_token_at(const int *tokens, int count, int index) {
    if (tokens == NULL || count <= 0 || index < 0 || index >= count) {
        return -1;
    }

    return tokens[index];
}

bool generate_text(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, const char *prompt, int steps) {
    int *prompt_tokens;
    int prompt_count;
    int token;
    int next;
    int pos;
    int generated;
    clock_t start;
    clock_t end;

    if (transformer == NULL || tokenizer == NULL || sampler == NULL || prompt == NULL || steps <= 0) {
        return false;
    }

    prompt_tokens = NULL;
    prompt_count = 0;
    if (!tokenizer_encode(tokenizer, prompt, true, false, &prompt_tokens, &prompt_count)) {
        return false;
    }

    if (prompt_count <= 0) {
        free(prompt_tokens);
        return false;
    }

    start = clock();
    token = prompt_token_at(prompt_tokens, prompt_count, 0);
    generated = 0;

    for (pos = 0; pos < steps && pos < transformer->config.seq_len - 1; ++pos) {
        if (!transformer_forward(transformer, token, pos)) {
            free(prompt_tokens);
            return false;
        }

        if (pos < prompt_count - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            const char *piece;

            next = sampler_sample(sampler, transformer->state.logits);
            if (next < 0) {
                free(prompt_tokens);
                return false;
            }

            if (next == 2) {
                break;
            }

            piece = tokenizer_decode(tokenizer, next);
            if (piece != NULL) {
                fputs(piece, stdout);
                fflush(stdout);
            }
            generated += 1;
        }

        token = next;
    }

    end = clock();
    if (generated > 0) {
        double elapsed = (double)(end - start) / (double)CLOCKS_PER_SEC;
        if (elapsed > 0.0) {
            fprintf(stdout, "\n[tok/s: %.2f]\n", generated / elapsed);
        } else {
            fputc('\n', stdout);
        }
    } else {
        fputc('\n', stdout);
    }

    free(prompt_tokens);
    return true;
}
