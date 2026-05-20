# Binary Format Specification

## 1. model.bin format

### Header

| field | type | description |
|---|---|---|
| dim | int32 | Transformer dimension |
| hidden_dim | int32 | FFN hidden dimension |
| n_layers | int32 | number of layers |
| n_heads | int32 | number of attention heads |
| n_kv_heads | int32 | number of key/value heads |
| vocab_size | int32 | vocabulary size |
| seq_len | int32 | max sequence length |

### Weight layout

After the header, float32 tensors are stored sequentially.

1. token_embedding_table
2. rms_att_weight
3. wq
4. wk
5. wv
6. wo
7. rms_ffn_weight
8. w1
9. w2
10. w3
11. rms_final_weight
12. freq_cis_real
13. freq_cis_imag
14. wcls, optional

## 2. tokenizer.bin format

### Header

| field | type | description |
|---|---|---|
| max_token_length | uint32 | maximum byte length of a token string |

### Vocabulary entries

Repeated `vocab_size` times:

| field | type | description |
|---|---|---|
| score | float32 | token score |
| length | int32 | byte length of token string |
| token | bytes[length] | UTF-8 token bytes |

## 3. Special tokens

| name | meaning |
|---|---|
| BOS | Beginning of sequence |
| EOS | End of sequence |
| UNK | Unknown token |

In llama2.c-style tokenizer, BOS is often token id `1`, EOS is often token id `2`, but exact values should be confirmed from the tokenizer/model configuration.