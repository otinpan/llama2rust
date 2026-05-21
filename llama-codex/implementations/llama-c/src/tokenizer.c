// @trace-pilot 30c2d3a2ea3e318716012edde7ade0854dd5fbf2
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int compare_token_index(const void *left, const void *right) {
    const TokenIndex *a = (const TokenIndex *)left;
    const TokenIndex *b = (const TokenIndex *)right;
    return strcmp(a->text, b->text);
}

// @trace-pilot 4167259149cda1179813b2e1d5e5c0855da2367b
static bool push_token(int **tokens, int *count, int *capacity, int token) {
    int new_capacity;
    int *new_tokens;

    if (tokens == NULL || count == NULL || capacity == NULL) {
        return false;
    }

    if (*count >= *capacity) {
        new_capacity = *capacity == 0 ? 16 : (*capacity * 2);
        new_tokens = realloc(*tokens, (size_t)new_capacity * sizeof(int));
        if (new_tokens == NULL) {
            return false;
        }
        *tokens = new_tokens;
        *capacity = new_capacity;
    }

    (*tokens)[*count] = token;
    *count += 1;
    return true;
}

static int utf8_char_len(unsigned char c) {
    if ((c & 0x80u) == 0) {
        return 1;
    }
    if ((c & 0xe0u) == 0xc0u) {
        return 2;
    }
    if ((c & 0xf0u) == 0xe0u) {
        return 3;
    }
    if ((c & 0xf8u) == 0xf0u) {
        return 4;
    }
    return 1;
}

bool tokenizer_init(Tokenizer *tokenizer) {
    int i;

    if (tokenizer == NULL) {
        return false;
    }

    *tokenizer = (Tokenizer){0};
    for (i = 0; i < 256; ++i) {
        tokenizer->byte_pieces[i][0] = (unsigned char)i;
        tokenizer->byte_pieces[i][1] = '\0';
    }

    return true;
}

bool tokenizer_load(Tokenizer *tokenizer, const char *path, int vocab_size) {
    FILE *file;
    int i;

    if (tokenizer == NULL || path == NULL || vocab_size <= 0) {
        return false;
    }

    tokenizer_free(tokenizer);
    if (!tokenizer_init(tokenizer)) {
        return false;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return false;
    }

    if (fread(&tokenizer->max_token_length, sizeof(tokenizer->max_token_length), 1, file) != 1) {
        fclose(file);
        tokenizer_free(tokenizer);
        return false;
    }

    tokenizer->vocab_size = vocab_size;
    tokenizer->vocab = calloc((size_t)vocab_size, sizeof(char *));
    tokenizer->scores = calloc((size_t)vocab_size, sizeof(float));
    tokenizer->sorted_vocab = calloc((size_t)vocab_size, sizeof(TokenIndex));
    if (tokenizer->vocab == NULL || tokenizer->scores == NULL || tokenizer->sorted_vocab == NULL) {
        fclose(file);
        tokenizer_free(tokenizer);
        return false;
    }

    for (i = 0; i < vocab_size; ++i) {
        int32_t length;
        char *piece;

        if (fread(&tokenizer->scores[i], sizeof(float), 1, file) != 1 ||
            fread(&length, sizeof(length), 1, file) != 1 ||
            length < 0) {
            fclose(file);
            tokenizer_free(tokenizer);
            return false;
        }

        piece = calloc((size_t)length + 1, 1);
        if (piece == NULL) {
            fclose(file);
            tokenizer_free(tokenizer);
            return false;
        }

        if (length > 0 && fread(piece, (size_t)length, 1, file) != 1) {
            free(piece);
            fclose(file);
            tokenizer_free(tokenizer);
            return false;
        }

        tokenizer->vocab[i] = piece;
        tokenizer->sorted_vocab[i].text = piece;
        tokenizer->sorted_vocab[i].id = i;
    }

    fclose(file);
    qsort(tokenizer->sorted_vocab, (size_t)vocab_size, sizeof(TokenIndex), compare_token_index);
    return true;
}

void tokenizer_free(Tokenizer *tokenizer) {
    int i;

    if (tokenizer == NULL) {
        return;
    }

    if (tokenizer->vocab != NULL) {
        for (i = 0; i < tokenizer->vocab_size; ++i) {
            free(tokenizer->vocab[i]);
        }
    }

    free(tokenizer->vocab);
    free(tokenizer->scores);
    free(tokenizer->sorted_vocab);
    *tokenizer = (Tokenizer){0};
}

const char *tokenizer_decode(const Tokenizer *tokenizer, int token) {
    if (tokenizer == NULL || tokenizer->vocab == NULL) {
        return NULL;
    }

    if (token < 0 || token >= tokenizer->vocab_size) {
        return NULL;
    }

    return tokenizer->vocab[token];
}

int tokenizer_lookup(const Tokenizer *tokenizer, const char *text) {
    TokenIndex key;
    TokenIndex *found;

    if (tokenizer == NULL || tokenizer->sorted_vocab == NULL || text == NULL) {
        return -1;
    }

    key.text = (char *)text;
    key.id = -1;
    found = bsearch(&key,
                    tokenizer->sorted_vocab,
                    (size_t)tokenizer->vocab_size,
                    sizeof(TokenIndex),
                    compare_token_index);

    return found != NULL ? found->id : -1;
}

// @trace-pilot 4167259149cda1179813b2e1d5e5c0855da2367b
bool tokenizer_encode(const Tokenizer *tokenizer, const char *text, bool bos, bool eos, int **tokens_out, int *count_out) {
    int *tokens;
    int count;
    int capacity;
    const unsigned char *p;

    if (tokenizer == NULL || text == NULL || tokens_out == NULL || count_out == NULL) {
        return false;
    }

    tokens = NULL;
    count = 0;
    capacity = 0;

    if (bos && !push_token(&tokens, &count, &capacity, 1)) {
        free(tokens);
        return false;
    }

    if (text[0] != '\0') {
        int dummy_prefix = tokenizer_lookup(tokenizer, " ");
        if (dummy_prefix >= 0 && !push_token(&tokens, &count, &capacity, dummy_prefix)) {
            free(tokens);
            return false;
        }
    }

    p = (const unsigned char *)text;
    while (*p != '\0') {
        char piece[8] = {0};
        int char_len = utf8_char_len(*p);
        int piece_len = 0;
        int id;
        int i;

        while (piece_len < char_len && p[piece_len] != '\0' && piece_len < (int)sizeof(piece) - 1) {
            piece[piece_len] = (char)p[piece_len];
            piece_len += 1;
        }
        piece[piece_len] = '\0';

        id = tokenizer_lookup(tokenizer, piece);
        if (id >= 0) {
            if (!push_token(&tokens, &count, &capacity, id)) {
                free(tokens);
                return false;
            }
        } else {
            for (i = 0; i < piece_len; ++i) {
                if (!push_token(&tokens, &count, &capacity, (int)p[i] + 3)) {
                    free(tokens);
                    return false;
                }
            }
        }

        p += piece_len;
    }

    for (;;) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_index = -1;
        int i;

        for (i = 0; i < count - 1; ++i) {
            const char *left = tokenizer_decode(tokenizer, tokens[i]);
            const char *right = tokenizer_decode(tokenizer, tokens[i + 1]);
            size_t left_len;
            size_t right_len;
            char *merged;
            int merged_id;

            if (left == NULL || right == NULL) {
                continue;
            }

            left_len = strlen(left);
            right_len = strlen(right);
            merged = malloc(left_len + right_len + 1);
            if (merged == NULL) {
                free(tokens);
                return false;
            }

            memcpy(merged, left, left_len);
            memcpy(merged + left_len, right, right_len + 1);
            merged_id = tokenizer_lookup(tokenizer, merged);
            free(merged);

            if (merged_id >= 0 && tokenizer->scores[merged_id] > best_score) {
                best_score = tokenizer->scores[merged_id];
                best_id = merged_id;
                best_index = i;
            }
        }

        if (best_index < 0) {
            break;
        }

        tokens[best_index] = best_id;
        memmove(&tokens[best_index + 1],
                &tokens[best_index + 2],
                (size_t)(count - best_index - 2) * sizeof(int));
        count -= 1;
    }

    if (eos && !push_token(&tokens, &count, &capacity, 2)) {
        free(tokens);
        return false;
    }

    *tokens_out = tokens;
    *count_out = count;
    return true;
}
