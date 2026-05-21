// @trace-pilot 30c2d3a2ea3e318716012edde7ade0854dd5fbf2
#ifndef LLAMA_C_TOKENIZER_H
#define LLAMA_C_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    char *text;
    int id;
} TokenIndex;

typedef struct {
    uint32_t max_token_length;
    int vocab_size;
    char **vocab;
    float *scores;
    TokenIndex *sorted_vocab;
    unsigned char byte_pieces[256][2];
} Tokenizer;

bool tokenizer_init(Tokenizer *tokenizer);
bool tokenizer_load(Tokenizer *tokenizer, const char *path, int vocab_size);
void tokenizer_free(Tokenizer *tokenizer);

const char *tokenizer_decode(const Tokenizer *tokenizer, int token);
int tokenizer_lookup(const Tokenizer *tokenizer, const char *text);
// @trace-pilot 4167259149cda1179813b2e1d5e5c0855da2367b
bool tokenizer_encode(const Tokenizer *tokenizer, const char *text, bool bos, bool eos, int **tokens_out, int *count_out);

#endif
