# tokenizer.md

# tokenizer モジュール要件仕様書

## 1. 概要

tokenizer モジュールは、自然言語文字列と Transformer が扱う token ID 列との相互変換を行うモジュールである。

本モジュールは、入力文字列を Byte Pair Encoding（BPE）に基づき token 化し、逆に token ID を自然言語文字列へ復元する。

また、UTF-8 処理、byte fallback、特殊 token 管理も担当する。

---

## 2. 目的

tokenizer モジュールの目的は以下である。

- 文字列を token ID 列へ変換する
- token ID を文字列へ復元する
- UTF-8 文字列を処理する
- BPE merge を実行する
- vocabulary を管理する
- 特殊 token を処理する
- byte fallback encoding を提供する

---

## 3. 対象範囲

tokenizer モジュールは、文字列と token の変換処理を対象とする。

以下の処理は対象外とする。

| 処理 | 担当 |
|---|---|
| Transformer 推論 | transformer.md |
| Sampling | sampler.md |
| Text Generation | generate.md |
| Chat Generation | chat.md |
| モデル構成管理 | config.md |

---

## 4. tokenizer モジュールの役割

tokenizer モジュールは以下を担当する。

| 処理 | 内容 |
|---|---|
| Encode | text → token |
| Decode | token → text |
| UTF-8 Parsing | マルチバイト文字処理 |
| Vocabulary 管理 | token 辞書管理 |
| BPE Merge | token 結合 |
| byte fallback | 未知文字処理 |
| Special Token 管理 | BOS/EOS 等 |

---

## 5. 入力データ

tokenizer モジュールは以下を入力として受け取る。

| 入力項目 | 説明 |
|---|---|
| Text | 入力文字列 |
| Token IDs | token 列 |
| tokenizer.bin | vocabulary file |
| vocab_size | vocabulary サイズ |

---

## 6. 出力データ

tokenizer モジュールは以下を出力する。

| 出力項目 | 説明 |
|---|---|
| Token IDs | encode 結果 |
| Decoded Text | decode 結果 |
| Vocabulary Metadata | token 情報 |

---

## 7. Tokenizer 構造

tokenizer モジュールは以下を管理する。

| 項目 | 説明 |
|---|---|
| vocabulary | token 文字列 |
| vocabulary scores | merge score |
| sorted vocabulary | 高速検索用 |
| max token length | 最大 token 長 |
| byte pieces | byte fallback 用 |

---

## 8. Vocabulary 管理

Tokenizer は vocabulary を保持する。

用途:

- token lookup
- encode
- decode
- BPE merge

---

## 9. Encode 処理

### 9.1 概要

Encode は文字列を token ID 列へ変換する。

### 9.2 処理フロー

```text
Input Text
↓
UTF-8 Parsing
↓
Initial Tokenization
↓
BPE Merge
↓
Token IDs
```

---

## 10. UTF-8 処理

Tokenizer は UTF-8 をサポートする。

対応内容:

| 項目 | 内容 |
|---|---|
| ASCII | 1 byte |
| UTF-8 multi-byte | 最大 4 byte |
| continuation byte | UTF-8 continuation |
| Unicode handling | UTF-8 sequence |

---

## 11. Initial Tokenization

入力文字列はまず UTF-8 単位で token 化される。

目的:

- BPE merge の初期状態生成
- Unicode 安全処理
- byte fallback 対応

---

## 12. BPE Merge

### 12.1 概要

BPE（Byte Pair Encoding）は隣接 token を merge する。

### 12.2 目的

- token 数削減
- language compression
- subword representation

---

### 12.3 処理

```text
Initial Tokens
↓
Best Pair Search
↓
Highest Score Merge
↓
Repeat
↓
Final Tokens
```

---

## 13. Vocabulary Score

各 token は merge score を持つ。

用途:

- merge 優先順位
- token ranking
- BPE optimization

---

## 14. byte fallback

### 14.1 概要

未知 token は byte 単位へ分解される。

### 14.2 目的

- unknown token 回避
- 任意文字列対応
- UTF-8 安全処理

---

### 14.3 動作

```text
Unknown UTF-8
↓
Byte Split
↓
Byte Token
```

---

## 15. Decode 処理

### 15.1 概要

Decode は token ID を文字列へ変換する。

### 15.2 処理フロー

```text
Token IDs
↓
Vocabulary Lookup
↓
Byte Restore
↓
UTF-8 Text
```

---

## 16. 特殊 Token

Tokenizer は特殊 token を管理する。

| Token | 説明 |
|---|---|
| BOS | begin of sequence |
| EOS | end of sequence |
| UNK | unknown token |

---

## 17. BOS Token

BOS token は sequence 開始を表す。

用途:

- generation start
- prompt initialization
- sequence boundary

---

## 18. EOS Token

EOS token は sequence 終了を表す。

用途:

- generation stop
- assistant response end
- sequence boundary

---

## 19. Dummy Prefix

Tokenizer は dummy prefix を利用する。

目的:

- sentencepiece 互換
- whitespace handling
- token consistency

---

## 20. Vocabulary Lookup

Tokenizer は token lookup を実行する。

用途:

- encode
- decode
- merge search

---

## 21. Sorted Vocabulary

Tokenizer は高速検索用 vocabulary を保持する。

用途:

- binary search
- token lookup
- merge candidate search

---

## 22. Decode Safety

Tokenizer は decode 時に unsafe byte を抑制する。

対象:

| 対象 | 内容 |
|---|---|
| control character | 非表示制御文字 |
| invalid byte | 不正 UTF-8 |
| non-printable | 非表示文字 |

---

## 23. Prompt Encoding

Tokenizer は Prompt を token 化する。

利用箇所:

| モジュール | 用途 |
|---|---|
| generate | text generation |
| chat | conversation prompt |

---

## 24. Assistant Response Decoding

Tokenizer は生成 token を decode する。

利用箇所:

| モジュール | 用途 |
|---|---|
| generate | generated text |
| chat | assistant response |

---

## 25. tokenizer.bin

Tokenizer は tokenizer.bin を利用する。

格納内容:

| 内容 | 説明 |
|---|---|
| vocabulary | token list |
| merge scores | BPE scores |
| token length | max token size |

---

## 26. 入力と出力

### 26.1 入力

```text
Input Text
Token IDs
tokenizer.bin
Vocabulary
```

### 26.2 出力

```text
Encoded Tokens
Decoded Text
Vocabulary Metadata
```

---

## 27. エラー処理

tokenizer モジュールは以下をエラーとして扱う。

| 条件 | 処理 |
|---|---|
| tokenizer.bin 読み込み失敗 | 終了 |
| invalid UTF-8 | byte fallback |
| unknown token | byte encoding |
| vocabulary overflow | 終了 |
| decode failure | safe output |

---

## 28. メモリ管理

tokenizer モジュールは以下を管理する。

| リソース | 内容 |
|---|---|
| vocabulary | token 文字列 |
| score buffer | merge score |
| sorted vocabulary | lookup table |
| temporary buffer | merge 作業領域 |

終了時にバッファを解放する。

---

## 29. 非機能要件

| 項目 | 要件 |
|---|---|
| Unicode 対応 | UTF-8 を正しく処理すること |
| 安全性 | invalid byte を安全処理すること |
| 効率性 | 高速 lookup を実現すること |
| 拡張性 | 新 tokenizer へ対応可能であること |
| 保守性 | encode/decode を分離管理すること |

---

## 30. 他モジュール依存

tokenizer モジュールは以下で利用される。

```text
tokenizer
├── generate
├── chat
├── transformer
└── sampler
```

---

## 31. 将来拡張

tokenizer モジュールは将来的に以下へ対応可能とする。

- sentencepiece tokenizer
- unigram tokenizer
- wordpiece tokenizer
- multilingual tokenizer
- streaming tokenizer
- tokenizer parallelization
- dynamic vocabulary
- multimodal tokenizer
- byte-level tokenizer
- regex tokenizer
- normalization pipeline