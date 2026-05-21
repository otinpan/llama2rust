// @trace-pilot dbfdb1a3446d84f4b3ce3dc9ccc5afaac39c4bd7
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"

typedef struct {
    float score;
    const char *text;
} TestToken;

static void write_tokenizer_file(const char *path) {
    static const TestToken tokens[] = {
        {0.0f, "<unk>"},
        {0.0f, "<s>"},
        {0.0f, "</s>"},
        {1.0f, "h"},
        {1.0f, "e"},
        {1.0f, "l"},
        {1.0f, "o"},
        {2.0f, " "},
        {3.0f, "he"},
        {3.0f, "ll"},
        {4.0f, "hell"},
        {5.0f, "hello"},
    };
    const uint32_t max_token_length = 8;
    FILE *file = fopen(path, "wb");
    size_t i;

    assert(file != NULL);
    assert(fwrite(&max_token_length, sizeof(max_token_length), 1, file) == 1);

    for (i = 0; i < sizeof(tokens) / sizeof(tokens[0]); ++i) {
        int32_t length = (int32_t)strlen(tokens[i].text);
        assert(fwrite(&tokens[i].score, sizeof(tokens[i].score), 1, file) == 1);
        assert(fwrite(&length, sizeof(length), 1, file) == 1);
        assert(fwrite(tokens[i].text, (size_t)length, 1, file) == 1);
    }

    fclose(file);
}

int main(void) {
    const char *path = "build/test_tokenizer.bin";
    Tokenizer tokenizer;
    int *encoded = NULL;
    int count = 0;

    write_tokenizer_file(path);

    assert(tokenizer_init(&tokenizer));
    assert(tokenizer_load(&tokenizer, path, 12));

    assert(tokenizer.max_token_length == 8);
    assert(tokenizer_lookup(&tokenizer, "hello") == 11);
    assert(tokenizer_lookup(&tokenizer, "missing") == -1);
    assert(strcmp(tokenizer_decode(&tokenizer, 7), " ") == 0);
    assert(strcmp(tokenizer_decode(&tokenizer, 11), "hello") == 0);

    assert(tokenizer_encode(&tokenizer, "hello", true, true, &encoded, &count));
    assert(count == 4);
    assert(encoded[0] == 1);
    assert(encoded[1] == 7);
    assert(encoded[2] == 11);
    assert(encoded[3] == 2);

    free(encoded);
    tokenizer_free(&tokenizer);
    remove(path);
    return 0;
}
