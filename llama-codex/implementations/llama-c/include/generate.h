// @trace-pilot 92fe5e02c071b0c60a204b35d23479e5dfebafbc
#ifndef LLAMA_C_GENERATE_H
#define LLAMA_C_GENERATE_H

#include <stdbool.h>

#include "sampler.h"
#include "tokenizer.h"
#include "transformer.h"

bool generate_text(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, const char *prompt, int steps);

#endif
