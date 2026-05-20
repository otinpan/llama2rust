# config.md

# config モジュール要件仕様書

## 1. 概要

config モジュールは、Transformer モデル全体の構成情報を管理するモジュールである。

本モジュールは、モデル構造に必要なハイパーパラメータを保持し、Transformer、Weights、Tokenizer、Sampler など各モジュールへ構成情報を提供する。

config はモデルの設計図として機能し、推論処理全体の基盤となる。

---

## 2. 目的

config モジュールの目的は以下である。

- モデル構造を定義する
- Transformer の構成情報を保持する
- 推論時の各種サイズ情報を管理する
- 各モジュールへ設定情報を提供する
- モデルファイルから構成情報を読み込む
- 推論時の整合性を保証する

---

## 3. 対象範囲

config モジュールは、モデル構成情報の管理を対象とする。

以下の処理は対象外とする。

| 処理 | 担当 |
|---|---|
| 重みロード | weights.md |
| 推論処理 | transformer.md |
| Sampling | sampler.md |
| Tokenizer 処理 | tokenizer.md |
| テキスト生成 | generation.md |

---

## 4. config モジュールの役割

config モジュールは以下を管理する。

| 管理対象 | 内容 |
|---|---|
| モデル次元 | hidden size |
| FFN サイズ | feed forward dimension |
| レイヤ数 | Transformer layer count |
| Attention Head 数 | query head 数 |
| KV Head 数 | key/value head 数 |
| Vocabulary サイズ | token 種類数 |
| Sequence Length | 最大 context 長 |

---

## 5. Config データ構造

config モジュールは、Transformer モデル構成を保持する。

### 管理項目

| 項目 | 説明 |
|---|---|
| dim | Transformer hidden dimension |
| hidden_dim | FFN hidden dimension |
| n_layers | Transformer layer 数 |
| n_heads | query attention head 数 |
| n_kv_heads | key/value attention head 数 |
| vocab_size | vocabulary size |
| seq_len | 最大 sequence 長 |

---

## 6. 入力データ

config モジュールは以下を入力として受け取る。

| 入力 | 説明 |
|---|---|
| model.bin | 学習済みモデルファイル |
| binary header | Config 情報 |

---

## 7. 出力データ

config モジュールは以下を出力する。

| 出力 | 説明 |
|---|---|
| Config Object | モデル構成情報 |
| Parameter Metadata | 各種サイズ情報 |

---

## 8. モデル構成管理

config モジュールは、Transformer 推論に必要なサイズ情報を管理する。

### 主な利用先

| モジュール | 利用内容 |
|---|---|
| transformer | attention/ffn サイズ |
| weights | weight offset 計算 |
| tokenizer | vocabulary サイズ |
| sampler | logits サイズ |
| generation | sequence 長管理 |

---

## 9. hidden dimension

### 9.1 dim

dim は Transformer の基本埋め込み次元を表す。

用途:

- embedding size
- attention size
- residual size
- hidden state size

---

### 9.2 hidden_dim

hidden_dim は FFN 内部次元を表す。

用途:

- SwiGLU
- feed-forward network
- intermediate activation

---

## 10. Attention 構成

### 10.1 n_heads

n_heads は query attention head 数を表す。

用途:

- multi-head attention
- query 分割
- attention parallelization

---

### 10.2 n_kv_heads

n_kv_heads は key/value head 数を表す。

用途:

- multi-query attention
- grouped-query attention
- KV cache 圧縮

---

## 11. Vocabulary 管理

### 11.1 vocab_size

vocab_size は tokenizer が扱う token 数を表す。

用途:

- embedding table サイズ
- logits サイズ
- sampling 範囲

---

## 12. Sequence 管理

### 12.1 seq_len

seq_len は最大 sequence 長を表す。

用途:

- KV cache サイズ
- attention buffer サイズ
- context window 制限

---

## 13. モデルファイル読み込み

config モジュールは model.bin のヘッダから構成情報を読み込む。

読み込み対象:

```text
binary header
↓
Config
↓
Transformer initialization
```

---

## 14. サイズ計算

config モジュールは以下のサイズ計算に利用される。

| 計算対象 | 内容 |
|---|---|
| head_size | dim / n_heads |
| kv_dim | dim × n_kv_heads / n_heads |
| attention size | n_heads × seq_len |
| cache size | layer × seq_len × kv_dim |

---

## 15. 推論時利用

config モジュールは推論中に以下へ利用される。

| 処理 | 用途 |
|---|---|
| Attention | head 数管理 |
| RoPE | head dimension 管理 |
| FFN | hidden size 管理 |
| logits | vocabulary サイズ |
| sampling | token 範囲 |

---

## 16. Sequence 制限

config モジュールは最大 sequence 長を制御する。

| 制限対象 | 内容 |
|---|---|
| Prompt Length | 最大入力長 |
| Generation Length | 最大生成長 |
| KV Cache | 最大保存長 |

---

## 17. KV Cache 関係

config モジュールは KV Cache サイズ決定に利用される。

使用項目:

| 項目 | 用途 |
|---|---|
| n_layers | layer 数 |
| seq_len | token 長 |
| n_kv_heads | KV head 数 |
| dim | hidden size |

---

## 18. 入力と出力

### 18.1 入力

```text
model.bin
binary header
model metadata
```

### 18.2 出力

```text
Config Object
model dimensions
attention metadata
sequence metadata
```

---

## 19. エラー処理

config モジュールは以下をエラーとして扱う。

| 条件 | 処理 |
|---|---|
| model.bin 読み込み失敗 | 終了 |
| Config サイズ不正 | 終了 |
| vocab_size 不正 | 終了 |
| seq_len 不正 | 終了 |
| layer 数不正 | 終了 |

---

## 20. 非機能要件

| 項目 | 要件 |
|---|---|
| 一貫性 | 全モジュールで共通設定を利用すること |
| 安全性 | 不正 Config を検出すること |
| 効率性 | サイズ計算を高速化すること |
| 保守性 | モデル構成を独立管理すること |
| 拡張性 | 新しいモデル構造へ対応可能であること |

---

## 21. 他モジュール依存

config モジュールは以下で利用される。

```text
config
├── transformer
├── weights
├── tokenizer
├── sampler
└── generation
```

---

## 22. 将来拡張

config モジュールは将来的に以下へ対応可能とする。

- dynamic context length
- rotary scaling
- grouped-query attention
- mixture of experts
- multimodal config
- quantization metadata
- distributed inference metadata
- flash attention metadata
- tensor parallel metadata
- speculative decoding metadata