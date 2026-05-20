# system.md
# システム要件仕様書

## 1. システム概要

本システムは、Transformer ベースの大規模言語モデル（LLM）を C 言語で推論実行する軽量推論システムである。

主な目的は以下である。

- 学習済み Transformer モデルのロード
- トークン列入力から次トークン予測を行う
- テキスト生成（generate モード）
- 対話生成（chat モード）
- Tokenizer による文字列⇔トークン変換
- Sampler による確率的トークンサンプリング

本システムは単一バイナリで動作し、CLI から利用可能である。

---

# 2. システム構成

```text
system.md
├── main.md
├── generation.md
│   ├── generate.md
│   └── chat.md
├── config.md
├── sampler.md
├── transformer.md
├── tokenizer.md
└── weights.md

binary_format.md
```

---

# 3. システムアーキテクチャ

## 3.1 全体フロー

```text
ユーザー入力
    ↓
Tokenizer.encode()
    ↓
Token ID列
    ↓
Transformer.forward()
    ↓
Logits
    ↓
Sampler.sample()
    ↓
Next Token
    ↓
Tokenizer.decode()
    ↓
文字列出力
```

---

# 4. モジュール一覧

| モジュール | 役割 |
|---|---|
| main | CLIエントリポイント |
| transformer | Transformer推論 |
| tokenizer | BPEトークナイズ |
| sampler | 次トークンサンプリング |
| generation/generate | テキスト生成 |
| generation/chat | チャット生成 |
| config | モデル設定 |
| weights | 学習済み重み管理 |

---

# 5. システム入力データ

## 5.1 モデル入力

Transformer モデルは以下を入力として受け取る。

| 入力 | 型 | 説明 |
|---|---|---|
| token | int | 現在トークンID |
| pos | int | シーケンス位置 |

forward関数:

```c
float* forward(Transformer* transformer, int token, int pos)
```

---

## 5.2 テキスト入力

CLI入力:

```bash
./run model.bin -i "Hello"
```

入力文字列は Tokenizer により Token ID 列へ変換される。

---

## 5.3 モデルファイル入力

| ファイル | 説明 |
|---|---|
| model.bin | 学習済みモデル |
| tokenizer.bin | Tokenizer辞書 |

---

# 6. システム出力データ

## 6.1 モデル出力

Transformer の出力は logits ベクトルである。

| 出力 | 型 | 説明 |
|---|---|---|
| logits | float* | 語彙全体の確率スコア |

出力サイズ:

```text
(vocab_size)
```

---

## 6.2 Sampler出力

Sampler は logits から次トークンを生成する。

| 出力 | 型 | 説明 |
|---|---|---|
| next_token | int | 次トークンID |

---

## 6.3 最終出力

Tokenizer.decode() により文字列へ変換される。

例:

```text
Input:
"Hello"

Output:
" world"
```

---

# 7. Transformer構造

## 7.1 Config

モデル構成定義。

```c
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;
```

---

## 7.2 Weights

学習済みパラメータ群。

主な重み:

| 重み | 説明 |
|---|---|
| token_embedding_table | Token埋め込み |
| wq | Query projection |
| wk | Key projection |
| wv | Value projection |
| wo | Attention output |
| w1/w2/w3 | FFN |
| rms_att_weight | Attention RMSNorm |
| rms_ffn_weight | FFN RMSNorm |

---

## 7.3 RunState

推論時の状態管理。

| バッファ | 説明 |
|---|---|
| x | 現在活性 |
| q | Query |
| k | Key |
| v | Value |
| att | Attention score |
| logits | 出力logits |
| key_cache | KV Cache |
| value_cache | KV Cache |

---

# 8. 推論処理

## 8.1 Forward処理

Transformer.forward() は以下を実施する。

### 処理フロー

```text
Embedding
  ↓
RMSNorm
  ↓
QKV Projection
  ↓
RoPE
  ↓
Self Attention
  ↓
Residual
  ↓
FFN
  ↓
Residual
  ↓
Final RMSNorm
  ↓
Logits Projection
```

---

# 9. Tokenizer

## 9.1 エンコード

入力文字列を Token ID 列へ変換。

```text
文字列
 ↓
UTF-8分解
 ↓
BPE Merge
 ↓
Token列
```

---

## 9.2 デコード

Token ID から文字列へ復元。

---

# 10. Sampling

## 10.1 Sampling方式

| 方式 | 説明 |
|---|---|
| Greedy | 最大確率選択 |
| Temperature | 温度調整 |
| Top-p | Nucleus Sampling |

---

# 11. Generationモード

## 11.1 generate

単方向テキスト生成。

入力:

```text
Prompt
```

出力:

```text
Generated Text
```

---

# 12. Chatモード

## 12.1 chat

Llama2 Chat形式を利用。

テンプレート:

```text
[INST] user prompt [/INST]
```

system prompt対応:

```text
<<SYS>>
system prompt
<</SYS>>
```

---

# 13. メモリ管理

## 13.1 mmap

モデル重みは mmap によりロードされる。

利点:

- 高速ロード
- コピー削減
- メモリ効率向上

---

# 14. 並列化

OpenMP により matmul を並列化。

```c
#pragma omp parallel for
```

---

# 15. エラーハンドリング

| エラー | 処理 |
|---|---|
| モデル読込失敗 | exit |
| malloc失敗 | exit |
| tokenizer読込失敗 | exit |

---

# 16. CLI仕様

## 16.1 実行形式

```bash
run <checkpoint> [options]
```

---

## 16.2 オプション

| オプション | 説明 |
|---|---|
| -t | temperature |
| -p | top-p |
| -s | random seed |
| -n | steps |
| -i | input prompt |
| -z | tokenizer path |
| -m | mode |
| -y | system prompt |

---

# 17. 性能要件

| 項目 | 内容 |
|---|---|
| 推論方式 | CPU |
| 重みロード | mmap |
| Attention | KV Cache対応 |
| 並列化 | OpenMP |

---

# 18. 今後の拡張

- GPU対応
- FlashAttention
- Batch推論
- Quantization
- Streaming生成
- Speculative Decoding
