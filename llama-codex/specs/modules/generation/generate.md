# generate.md
# generate モジュール要件仕様書

## 1. 概要

generate モジュールは、入力プロンプトを基に自然言語テキストを逐次生成するためのモジュールである。

本モジュールは、Tokenizer によって変換された入力トークン列を Transformer に入力し、Sampler を利用して次トークンを決定しながら文章生成を行う。

生成されたトークンは decode 処理によって文字列へ変換され、標準出力へ逐次出力される。

---

## 2. 目的

generate モジュールの目的は以下である。

- 入力プロンプトをトークン列へ変換する
- Transformer により次トークンを予測する
- Sampler により次トークンを選択する
- 自然言語テキストを逐次生成する
- 生成結果をリアルタイム出力する
- 推論速度を計測する

---

## 3. 対象範囲

generate モジュールは、単方向テキスト生成を対象とする。

以下の機能は対象外とする。

| 機能 | 担当 |
|---|---|
| チャット形式生成 | chat.md |
| モデル構築 | transformer.md |
| Tokenizer 処理 | tokenizer.md |
| Sampling ロジック | sampler.md |
| モデル設定 | config.md |

---

## 4. 入力データ

generate モジュールは以下を入力として受け取る。

| 入力項目 | 説明 |
|---|---|
| Transformer | 推論モデル |
| Tokenizer | トークン変換器 |
| Sampler | サンプリング器 |
| Prompt | 入力文字列 |
| Steps | 最大生成ステップ数 |

---

## 5. 出力データ

generate モジュールは以下を出力する。

| 出力項目 | 説明 |
|---|---|
| Generated Text | 生成テキスト |
| Token/sec | 推論速度 |
| Error Message | エラー発生時のメッセージ |

---

## 6. generate モジュールの役割

本モジュールは、以下の処理を統括する。

| 処理 | 内容 |
|---|---|
| Prompt Encoding | 入力文字列を Token ID 化 |
| Forward 推論 | 次トークン logits 計算 |
| Sampling | 次トークン選択 |
| Decode | Token を文字列へ変換 |
| Output | 標準出力へ表示 |
| Loop Control | 最大 step まで反復 |

---

## 7. 処理フロー

generate モジュールの処理フローを以下に示す。

1. Prompt を受け取る
2. Prompt を Tokenizer で encode する
3. 最初の token を Transformer に入力する
4. logits を取得する
5. Sampler で次トークンを決定する
6. decode により文字列へ変換する
7. 出力する
8. 次トークンを入力として再度推論する
9. 終了条件まで繰り返す

---

## 8. Prompt 処理

### 8.1 Prompt 入力

generate モジュールは Prompt 文字列を受け取る。

Prompt 未指定時は空文字列を使用する。

---

### 8.2 Prompt Encoding

Tokenizer を使用して Prompt を Token ID 列へ変換する。

変換対象:

```text
文字列
↓
UTF-8
↓
BPE Token
↓
Token ID列
```

---

## 9. 推論処理

generate モジュールは、1 token ごとに Transformer.forward を実行する。

### 推論入力

| 入力 | 説明 |
|---|---|
| token | 現在トークン |
| position | シーケンス位置 |

### 推論出力

| 出力 | 説明 |
|---|---|
| logits | 次トークン候補スコア |

---

## 10. Sampling 処理

Sampler は logits を入力として次トークンを選択する。

対応する sampling 方式:

| 方式 | 説明 |
|---|---|
| Greedy | 最大確率 token を選択 |
| Temperature Sampling | 温度付き sampling |
| Top-p Sampling | nucleus sampling |

---

## 11. 出力処理

選択された token は decode により文字列へ変換される。

生成された文字列は逐次標準出力へ表示される。

---

## 12. 生成ループ

generate モジュールは以下条件まで生成を継続する。

| 条件 | 説明 |
|---|---|
| 最大 step 到達 | steps に達した場合 |
| BOS token 検出 | sequence delimiter |
| エラー発生 | 推論継続不能 |

---

## 13. Position 管理

generate モジュールは、各 token に対して sequence position を管理する。

| position | 説明 |
|---|---|
| 0 | 最初の token |
| n | n 番目 token |

position は RoPE および KV Cache に利用される。

---

## 14. KV Cache 利用

generate モジュールは Transformer 内部の KV Cache を利用する。

目的:

- 過去 token の再計算削減
- 推論高速化
- Attention 計算効率化

---

## 15. 推論速度計測

generate モジュールは token/sec を計測する。

計測内容:

| 指標 | 説明 |
|---|---|
| tok/s | 1秒あたり生成 token 数 |

---

## 16. エラー処理

generate モジュールは以下をエラーとして扱う。

| 条件 | 処理 |
|---|---|
| Prompt encode 失敗 | 終了 |
| Token 数異常 | エラー出力 |
| 推論失敗 | 終了 |
| decode 失敗 | 安全出力 |
| メモリ不足 | 終了 |

---

## 17. メモリ管理

generate モジュールは以下を管理する。

| リソース | 内容 |
|---|---|
| Prompt Token Buffer | encode 結果 |
| logits | 推論出力 |
| decode buffer | 出力文字列 |
| timing state | 性能計測 |

推論終了後に必要なバッファを解放する。

---

## 18. 入力と出力

### 18.1 入力

```text
Prompt
Transformer
Tokenizer
Sampler
Generation Parameters
```

### 18.2 出力

```text
Generated Text
Token/sec
Error Logs
```

---

## 19. 非機能要件

| 項目 | 要件 |
|---|---|
| リアルタイム性 | token 単位で逐次出力すること |
| 効率性 | KV Cache を利用すること |
| 安定性 | 不正 token 出力を抑制すること |
| 拡張性 | 新 sampling 手法を追加可能であること |
| 保守性 | encode / sample / decode を分離すること |

---

## 20. 他モジュール依存

generate モジュールは以下へ依存する。

```text
generate
├── transformer
├── tokenizer
├── sampler
└── config
```

---

## 21. 将来拡張

generate モジュールは将来的に以下へ対応可能とする。

- Streaming API
- speculative decoding
- beam search
- repetition penalty
- stop sequence
- batch generation
- async generation
- GPU inference
- FlashAttention
- quantized inference