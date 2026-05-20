# chat.md

# chat モジュール要件仕様書

## 1. 概要

chat モジュールは、ユーザーと Assistant の対話形式による自然言語生成を実現するモジュールである。

本モジュールは、ユーザー入力および System Prompt を Llama 系 Chat Template に変換し、Transformer を用いて応答生成を行う。

生成された応答は逐次 decode され、リアルタイムに出力される。

---

## 2. 目的

chat モジュールの目的は以下である。

- 対話形式の入力を受け付ける
- System Prompt を適用する
- Chat Template を生成する
- ユーザー入力を token 化する
- Assistant 応答を生成する
- マルチターン会話を管理する
- リアルタイム応答を出力する

---

## 3. 対象範囲

chat モジュールは、対話型生成を対象とする。

以下の処理は対象外とする。

| 処理 | 担当 |
|---|---|
| 通常テキスト生成 | generate.md |
| Transformer 推論 | transformer.md |
| Tokenizer 処理 | tokenizer.md |
| Sampling 処理 | sampler.md |
| モデル設定 | config.md |

---

## 4. 入力データ

chat モジュールは以下を入力として受け取る。

| 入力項目 | 説明 |
|---|---|
| Transformer | 推論モデル |
| Tokenizer | トークン変換器 |
| Sampler | サンプリング器 |
| User Prompt | ユーザー入力 |
| System Prompt | システム指示 |
| Steps | 最大生成ステップ数 |

---

## 5. 出力データ

chat モジュールは以下を出力する。

| 出力項目 | 説明 |
|---|---|
| Assistant Response | Assistant 応答 |
| Generated Tokens | 生成 token |
| Error Message | エラー情報 |

---

## 6. chat モジュールの役割

本モジュールは以下の処理を統括する。

| 処理 | 内容 |
|---|---|
| User Input | ユーザー入力受付 |
| System Prompt 管理 | システム指示管理 |
| Chat Template 生成 | Llama Chat Format 化 |
| Prompt Encoding | Token 化 |
| Forward 推論 | logits 計算 |
| Sampling | 次 token 選択 |
| Decode | token を文字列化 |
| Output | Assistant 応答出力 |
| Turn Management | 会話ターン管理 |

---

## 7. Chat Template

chat モジュールは、Llama 系 Chat Template を利用する。

### 7.1 通常形式

```text
[INST] user prompt [/INST]
```

---

### 7.2 System Prompt 付き形式

```text
[INST]
<<SYS>>
system prompt
<</SYS>>

user prompt
[/INST]
```

---

## 8. 処理フロー

chat モジュールの処理フローを以下に示す。

1. System Prompt を取得する
2. User Prompt を取得する
3. Chat Template を生成する
4. Prompt を encode する
5. Transformer に token を入力する
6. logits を取得する
7. Sampler で次 token を決定する
8. decode して文字列化する
9. Assistant 応答を出力する
10. EOS token まで繰り返す
11. 次の User Turn を待機する

---

## 9. User Input 管理

chat モジュールは、ユーザー入力を逐次受け付ける。

入力方法:

| 方法 | 内容 |
|---|---|
| CLI Prompt | 標準入力 |
| 初期引数 | command line prompt |

---

## 10. System Prompt 管理

chat モジュールは、会話全体に適用される System Prompt を管理する。

用途:

- Assistant の人格設定
- 出力スタイル制御
- 応答方針制御
- 制約条件付与

---

## 11. Prompt Encoding

Chat Template 化された Prompt は Tokenizer により token 化される。

変換フロー:

```text
Chat Prompt
↓
UTF-8
↓
BPE Tokenization
↓
Token ID列
```

---

## 12. 推論処理

chat モジュールは token 単位で Transformer.forward を実行する。

### 推論入力

| 入力 | 説明 |
|---|---|
| token | 現在 token |
| position | sequence position |

### 推論出力

| 出力 | 説明 |
|---|---|
| logits | 次 token 候補 |

---

## 13. Sampling 処理

Sampler は logits を用いて次 token を選択する。

対応方式:

| Sampling | 説明 |
|---|---|
| Greedy | 最大確率選択 |
| Temperature | 温度付き sampling |
| Top-p | nucleus sampling |

---

## 14. Assistant 応答生成

Assistant 応答は逐次生成される。

生成された token は decode 後に標準出力へ表示される。

出力形式:

```text
Assistant: response text...
```

---

## 15. Turn 管理

chat モジュールは User Turn と Assistant Turn を管理する。

| Turn | 内容 |
|---|---|
| User Turn | Prompt 入力 |
| Assistant Turn | 応答生成 |

---

## 16. EOS 管理

EOS token は Assistant 応答終了を示す。

| Token | 意味 |
|---|---|
| EOS (=2) | Assistant 応答終了 |

EOS 検出後、chat モジュールは User Turn に戻る。

---

## 17. Position 管理

chat モジュールは sequence position を管理する。

用途:

- RoPE
- Attention
- KV Cache
- sequence tracking

---

## 18. KV Cache 利用

chat モジュールは KV Cache を利用する。

目的:

- Attention 再計算削減
- 応答生成高速化
- 長文対話効率化

---

## 19. リアルタイム出力

chat モジュールは token 単位で応答を逐次出力する。

目的:

- Streaming 応答
- 応答遅延低減
- 対話体験向上

---

## 20. エラー処理

chat モジュールは以下をエラーとして扱う。

| 条件 | 処理 |
|---|---|
| Prompt encode 失敗 | 終了 |
| 推論失敗 | 終了 |
| decode 失敗 | 安全出力 |
| Token 異常 | エラー出力 |
| メモリ不足 | 終了 |

---

## 21. メモリ管理

chat モジュールは以下を管理する。

| リソース | 内容 |
|---|---|
| Prompt Buffer | 入力文字列 |
| Token Buffer | encode 結果 |
| logits | 推論出力 |
| decode buffer | 出力文字列 |

推論終了後に必要なバッファを解放する。

---

## 22. 入力と出力

### 22.1 入力

```text
User Prompt
System Prompt
Transformer
Tokenizer
Sampler
Generation Parameters
```

### 22.2 出力

```text
Assistant Response
Generated Tokens
Error Logs
```

---

## 23. 非機能要件

| 項目 | 要件 |
|---|---|
| リアルタイム性 | token 単位で逐次出力すること |
| 応答性 | User Turn へ即座に復帰可能であること |
| 効率性 | KV Cache を利用すること |
| 安定性 | 不正 token を安全処理すること |
| 拡張性 | 新 Chat Template を追加可能であること |
| 保守性 | Turn 管理を独立実装すること |

---

## 24. 他モジュール依存

chat モジュールは以下へ依存する。

```text
chat
├── transformer
├── tokenizer
├── sampler
└── config
```

---

## 25. 将来拡張

chat モジュールは将来的に以下へ対応可能とする。

- conversation history
- memory management
- function calling
- tool use
- streaming API
- multi-user session
- websocket chat
- repetition penalty
- stop sequence
- role-based prompts
- multimodal input
- long context support