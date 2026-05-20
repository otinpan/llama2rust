# transformer.md

# transformer モジュール要件仕様書

## 1. 概要

transformer モジュールは、大規模言語モデル（LLM）の中核となる Transformer 推論処理を担当するモジュールである。

本モジュールは、入力 token を受け取り、Embedding、Attention、FFN、Residual、RMSNorm などを順次実行し、次 token 予測用 logits を生成する。

また、高速推論のため KV Cache および RoPE を利用する。

---

## 2. 目的

transformer モジュールの目的は以下である。

- Token embedding を生成する
- Self Attention を計算する
- Feed Forward Network を計算する
- RoPE を適用する
- Residual 接続を適用する
- RMSNorm を適用する
- logits を生成する
- KV Cache を管理する

---

## 3. 対象範囲

transformer モジュールは、Transformer 推論処理を対象とする。

以下の処理は対象外とする。

| 処理 | 担当 |
|---|---|
| Sampling | sampler.md |
| Tokenizer 処理 | tokenizer.md |
| Text Generation | generate.md |
| Chat Generation | chat.md |
| Config 管理 | config.md |

---

## 4. transformer モジュールの役割

transformer モジュールは以下を担当する。

| 処理 | 内容 |
|---|---|
| Embedding | token embedding |
| Attention | self attention |
| RoPE | rotary positional embedding |
| RMSNorm | normalization |
| FFN | feed forward network |
| Residual | residual connection |
| KV Cache | key/value cache |
| logits 出力 | next token prediction |

---

## 5. 入力データ

transformer モジュールは以下を入力として受け取る。

| 入力項目 | 説明 |
|---|---|
| token | 現在 token |
| position | sequence position |
| Config | モデル構成 |
| Weights | 学習済み重み |
| KV Cache | 過去 token 状態 |

---

## 6. 出力データ

transformer モジュールは以下を出力する。

| 出力項目 | 説明 |
|---|---|
| logits | 次 token 候補 |
| hidden state | 中間活性 |
| attention scores | attention weight |
| KV Cache | 更新済み cache |

---

## 7. Transformer 構成

transformer モジュールは以下の構造を持つ。

```text
Input Token
↓
Embedding
↓
Transformer Layer × N
    ├── RMSNorm
    ├── Self Attention
    ├── Residual
    ├── RMSNorm
    ├── FFN
    └── Residual
↓
Final RMSNorm
↓
Linear Projection
↓
Logits
```

---

## 8. Embedding 処理

### 8.1 概要

入力 token を embedding vector へ変換する。

### 8.2 入力

| 入力 | 説明 |
|---|---|
| token id | vocabulary index |

### 8.3 出力

| 出力 | 説明 |
|---|---|
| embedding vector | hidden representation |

---

## 9. Transformer Layer

各 layer は以下で構成される。

| 構成 | 内容 |
|---|---|
| RMSNorm | attention 前正規化 |
| Attention | self attention |
| Residual | skip connection |
| RMSNorm | FFN 前正規化 |
| FFN | feed forward network |
| Residual | skip connection |

---

## 10. Self Attention

### 10.1 概要

現在 token と過去 token 間の関連性を計算する。

### 10.2 Attention 構成

| 要素 | 内容 |
|---|---|
| Query | 現在 token |
| Key | 過去 token |
| Value | context information |

---

### 10.3 Attention 処理

```text
Input
↓
QKV Projection
↓
QK Similarity
↓
Softmax
↓
Weighted Sum
↓
Attention Output
```

---

## 11. Multi-Head Attention

transformer モジュールは Multi-Head Attention を利用する。

目的:

- 異なる特徴抽出
- context 理解向上
- attention parallelization

---

## 12. Multi-Query Attention

transformer モジュールは n_kv_heads を利用した Multi-Query Attention に対応する。

特徴:

- KV cache 削減
- メモリ効率向上
- 推論高速化

---

## 13. RoPE

### 13.1 概要

RoPE（Rotary Positional Embedding）は位置情報を Query/Key に埋め込む。

### 13.2 目的

- 長文位置情報保持
- relative position 表現
- sequence ordering

---

### 13.3 適用対象

| 対象 | 内容 |
|---|---|
| Query | 回転適用 |
| Key | 回転適用 |

---

## 14. RMSNorm

### 14.1 概要

RMSNorm は hidden state を正規化する。

### 14.2 目的

- 学習安定化
- 数値安定性向上
- activation scaling

---

## 15. FFN

### 15.1 概要

FFN は各 token に対する非線形変換を行う。

### 15.2 構成

| 要素 | 内容 |
|---|---|
| W1 | projection |
| W2 | output projection |
| W3 | gating projection |
| SwiGLU | activation |

---

## 16. SwiGLU

transformer モジュールは SwiGLU activation を利用する。

特徴:

- smooth activation
- gating mechanism
- 表現力向上

---

## 17. Residual Connection

Residual Connection は layer 入出力を加算する。

目的:

- gradient flow 維持
- 深層安定化
- 情報保持

---

## 18. KV Cache

### 18.1 概要

KV Cache は過去 token の Key/Value を保存する。

### 18.2 目的

- 再計算削減
- 推論高速化
- 長文効率化

---

### 18.3 保存内容

| 保存対象 | 内容 |
|---|---|
| Key Cache | 過去 key |
| Value Cache | 過去 value |

---

## 19. logits 出力

Transformer 最終出力は vocabulary 空間へ projection される。

出力:

```text
hidden state
↓
linear projection
↓
logits
```

---

## 20. Sequence Position 管理

transformer モジュールは token position を管理する。

用途:

- RoPE
- Attention
- KV Cache indexing

---

## 21. Attention Buffer

Attention score は buffer に格納される。

用途:

- softmax
- weighted attention
- temporary storage

---

## 22. RunState 管理

transformer モジュールは推論時状態を保持する。

管理対象:

| 項目 | 内容 |
|---|---|
| hidden state | current activation |
| query | attention query |
| key | attention key |
| value | attention value |
| logits | output logits |
| attention buffer | attention score |
| KV cache | cached state |

---

## 23. Memory Mapping

transformer モジュールは model.bin を mmap によりロードする。

目的:

- 高速ロード
- コピー削減
- メモリ効率化

---

## 24. 並列化

transformer モジュールは OpenMP により並列化可能である。

対象:

| 処理 | 内容 |
|---|---|
| matmul | matrix multiplication |
| attention | head parallelization |

---

## 25. 数値安定性

transformer モジュールは数値安定性を考慮する。

対策:

- RMSNorm
- softmax stabilization
- max subtraction
- float scaling

---

## 26. 入力と出力

### 26.1 入力

```text
Token ID
Position
Weights
Config
KV Cache
```

### 26.2 出力

```text
Logits
Updated KV Cache
Hidden State
Attention Output
```

---

## 27. エラー処理

transformer モジュールは以下をエラーとして扱う。

| 条件 | 処理 |
|---|---|
| weight load failure | 終了 |
| invalid dimension | 終了 |
| sequence overflow | 終了 |
| memory allocation failure | 終了 |
| invalid token | 終了 |

---

## 28. 非機能要件

| 項目 | 要件 |
|---|---|
| 高速性 | token 単位高速推論を行うこと |
| メモリ効率 | KV Cache を利用すること |
| 安定性 | 数値 overflow を抑制すること |
| 拡張性 | 新 attention 構造へ対応可能であること |
| 保守性 | layer 構造を独立管理すること |

---

## 29. 他モジュール依存

transformer モジュールは以下へ依存する。

```text
transformer
├── config
├── weights
├── tokenizer
├── sampler
└── generation
```

---

## 30. 将来拡張

transformer モジュールは将来的に以下へ対応可能とする。

- FlashAttention
- grouped-query attention
- sliding window attention
- speculative decoding
- tensor parallelism
- pipeline parallelism
- quantized inference
- GPU backend
- mixed precision
- distributed inference
- multimodal transformer
- MoE architecture
