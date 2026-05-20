# weights.md

# weights モジュール要件仕様書

## 1. 概要

weights モジュールは、Transformer モデルの学習済みパラメータ（Weights）を管理するモジュールである。

本モジュールは、model.bin から各種重みデータを読み込み、Transformer 推論時に利用可能な形でメモリへ配置する。

また、Embedding、Attention、FFN、RMSNorm、Classifier などの全パラメータを統合管理する。

---

## 2. 目的

weights モジュールの目的は以下である。

- 学習済み重みをロードする
- 重み領域を管理する
- Transformer 各層へ重みを提供する
- memory mapping を利用する
- 推論時の高速アクセスを実現する
- parameter offset を管理する

---

## 3. 対象範囲

weights モジュールは、モデル重み管理を対象とする。

以下の処理は対象外とする。

| 処理 | 担当 |
|---|---|
| Transformer 推論 | transformer.md |
| Sampling | sampler.md |
| Tokenizer 処理 | tokenizer.md |
| Generation | generation.md |
| Config 管理 | config.md |

---

## 4. weights モジュールの役割

weights モジュールは以下を担当する。

| 処理 | 内容 |
|---|---|
| model.bin 読み込み | checkpoint load |
| parameter mapping | 重み割当 |
| embedding 管理 | token embedding |
| attention weight 管理 | QKV/WO |
| FFN weight 管理 | W1/W2/W3 |
| RMSNorm 管理 | normalization weight |
| classifier 管理 | logits projection |
| memory mapping | mmap 管理 |

---

## 5. 入力データ

weights モジュールは以下を入力として受け取る。

| 入力項目 | 説明 |
|---|---|
| model.bin | 学習済みモデル |
| Config | モデル構成 |
| binary parameter data | 重みデータ |

---

## 6. 出力データ

weights モジュールは以下を出力する。

| 出力項目 | 説明 |
|---|---|
| TransformerWeights | 全重み情報 |
| parameter pointer | 各 weight 領域 |
| mapped memory | mmap 領域 |

---

## 7. Weights 構造

weights モジュールは以下の重みを保持する。

| 項目 | 説明 |
|---|---|
| token_embedding_table | embedding |
| rms_att_weight | attention RMSNorm |
| rms_ffn_weight | FFN RMSNorm |
| wq | query projection |
| wk | key projection |
| wv | value projection |
| wo | attention output projection |
| w1 | FFN projection |
| w2 | FFN output |
| w3 | FFN gate |
| rms_final_weight | final RMSNorm |
| wcls | classifier weight |

---

## 8. Embedding Weight

### 8.1 概要

Embedding weight は token ID を hidden vector へ変換する。

### 8.2 用途

- token embedding
- input representation
- vocabulary projection

---

## 9. Attention Weights

weights モジュールは Attention 用重みを管理する。

### 管理対象

| 重み | 用途 |
|---|---|
| wq | Query projection |
| wk | Key projection |
| wv | Value projection |
| wo | Attention output |

---

## 10. FFN Weights

weights モジュールは FFN 用重みを管理する。

### 管理対象

| 重み | 用途 |
|---|---|
| w1 | FFN projection |
| w2 | FFN output |
| w3 | SwiGLU gate |

---

## 11. RMSNorm Weights

weights モジュールは RMSNorm 用重みを管理する。

### 管理対象

| 重み | 用途 |
|---|---|
| rms_att_weight | attention normalization |
| rms_ffn_weight | FFN normalization |
| rms_final_weight | final normalization |

---

## 12. Classifier Weight

### 12.1 概要

Classifier weight は hidden state を vocabulary 空間へ変換する。

### 12.2 出力

```text
hidden state
↓
linear projection
↓
logits
```

---

## 13. Shared Weight

weights モジュールは embedding と classifier の shared weight に対応する。

目的:

- parameter 削減
- memory 削減
- embedding tying

---

## 14. model.bin 読み込み

weights モジュールは model.bin をロードする。

読み込み内容:

```text
Config
↓
Weights
↓
Parameter Mapping
```

---

## 15. Memory Mapping

### 15.1 概要

weights モジュールは mmap を利用して model.bin をロードする。

### 15.2 目的

- 高速ロード
- copy 削減
- memory efficiency
- large model support

---

## 16. Parameter Mapping

weights モジュールは binary parameter data を各 weight 領域へ割り当てる。

処理内容:

```text
binary data
↓
offset calculation
↓
pointer assignment
↓
TransformerWeights
```

---

## 17. Parameter Offset 管理

weights モジュールは parameter offset を計算する。

用途:

- memory mapping
- pointer assignment
- layer separation

---

## 18. Layer Weight 管理

weights モジュールは layer 単位で parameter を管理する。

対象:

| 項目 | 内容 |
|---|---|
| attention | layer attention |
| FFN | layer feed forward |
| RMSNorm | layer normalization |

---

## 19. Attention Dimension 管理

weights モジュールは attention dimension を管理する。

利用情報:

| 項目 | 内容 |
|---|---|
| dim | hidden size |
| n_heads | query heads |
| n_kv_heads | kv heads |
| head_size | per-head dimension |

---

## 20. KV Weight 管理

weights モジュールは Key/Value projection weight を管理する。

用途:

- KV Cache
- attention projection
- multi-query attention

---

## 21. Weight Access

Transformer は weights モジュールを通じて parameter を参照する。

利用フロー:

```text
Transformer Layer
↓
Weight Access
↓
MatMul
↓
Activation
```

---

## 22. Vocabulary Projection

weights モジュールは vocabulary projection を管理する。

用途:

- logits generation
- next token prediction
- sampling input

---

## 23. Sequence 関連 Weight

weights モジュールは sequence 長に依存する parameter を管理する。

対象:

| 項目 | 内容 |
|---|---|
| RoPE metadata | rotary position |
| sequence buffer | context support |

---

## 24. 入力と出力

### 24.1 入力

```text
model.bin
Config
binary parameter data
```

### 24.2 出力

```text
TransformerWeights
mapped parameter memory
weight pointers
```

---

## 25. エラー処理

weights モジュールは以下をエラーとして扱う。

| 条件 | 処理 |
|---|---|
| model.bin 読み込み失敗 | 終了 |
| mmap failure | 終了 |
| invalid parameter size | 終了 |
| invalid offset | 終了 |
| memory allocation failure | 終了 |

---

## 26. メモリ管理

weights モジュールは以下を管理する。

| リソース | 内容 |
|---|---|
| mapped memory | mmap 領域 |
| file descriptor | checkpoint file |
| parameter pointers | weight pointer |
| layer metadata | layer info |

終了時にメモリおよび file descriptor を解放する。

---

## 27. 非機能要件

| 項目 | 要件 |
|---|---|
| 高速性 | 高速 parameter access を行うこと |
| メモリ効率 | mmap を利用すること |
| 安定性 | invalid parameter を検出すること |
| 拡張性 | 新 weight 構造へ対応可能であること |
| 保守性 | parameter mapping を独立管理すること |

---

## 28. 他モジュール依存

weights モジュールは以下で利用される。

```text
weights
├── transformer
├── config
├── generation
└── sampler
```

---

## 29. 将来拡張

weights モジュールは将来的に以下へ対応可能とする。

- quantized weights
- FP16 weights
- BF16 weights
- GPU memory mapping
- distributed parameter loading
- tensor parallel weights
- LoRA adapter
- adapter fusion
- multimodal weights
- MoE routing weights
- dynamic weight loading
- sharded checkpoint support