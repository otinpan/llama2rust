# sampler.md

# sampler モジュール要件仕様書

## 1. 概要

sampler モジュールは、Transformer が出力した logits から次トークンを選択するためのモジュールである。

本モジュールは、確率分布に基づき自然言語生成の多様性と安定性を制御する役割を持つ。

Sampler は temperature、top-p などの生成パラメータを利用し、最終的な next token を決定する。

---

## 2. 目的

sampler モジュールの目的は以下である。

- logits を確率分布へ変換する
- 次 token を選択する
- 出力のランダム性を制御する
- 低確率 token を抑制する
- 再現可能な sampling を提供する
- 複数 sampling 手法を提供する

---

## 3. 対象範囲

sampler モジュールは、次 token 選択処理を対象とする。

以下の処理は対象外とする。

| 処理 | 担当 |
|---|---|
| Transformer 推論 | transformer.md |
| Tokenizer 処理 | tokenizer.md |
| Text Generation | generate.md |
| Chat Generation | chat.md |
| モデル設定 | config.md |

---

## 4. sampler モジュールの役割

sampler モジュールは以下を担当する。

| 処理 | 内容 |
|---|---|
| logits 正規化 | softmax 計算 |
| temperature 適用 | 分布調整 |
| token sampling | 次 token 選択 |
| top-p filtering | 候補 token 制限 |
| random generation | 確率的生成 |
| deterministic generation | Greedy 出力 |

---

## 5. 入力データ

sampler モジュールは以下を入力として受け取る。

| 入力項目 | 説明 |
|---|---|
| logits | Transformer 出力 |
| temperature | ランダム性制御 |
| top-p | nucleus sampling 閾値 |
| random seed | 乱数状態 |
| vocab_size | vocabulary サイズ |

---

## 6. 出力データ

sampler モジュールは以下を出力する。

| 出力項目 | 説明 |
|---|---|
| next token | 次 token ID |
| probability distribution | token 確率分布 |

---

## 7. Sampler 構造

sampler モジュールは sampling 状態を保持する。

### 管理項目

| 項目 | 説明 |
|---|---|
| vocab_size | token 数 |
| temperature | 出力ランダム性 |
| top-p | nucleus threshold |
| rng_state | 乱数状態 |
| probability buffer | sampling 用バッファ |

---

## 8. Sampling 処理フロー

sampler モジュールは以下の手順で動作する。

1. logits を受け取る
2. temperature を適用する
3. softmax により確率化する
4. sampling 候補を決定する
5. 確率に従い token を選択する
6. next token を返却する

---

## 9. Temperature Sampling

### 9.1 概要

temperature は logits の分布を調整する。

### 9.2 効果

| temperature | 効果 |
|---|---|
| 0 | deterministic |
| 0〜1 | 安定生成 |
| 1 | 標準 |
| >1 | 多様性増加 |

---

## 10. Greedy Sampling

temperature が 0 の場合、最大確率 token を選択する。

特徴:

- deterministic
- 再現性あり
- 高速
- 多様性低下

---

## 11. Softmax 処理

Sampler は logits を softmax により確率分布へ変換する。

処理内容:

```text
logits
↓
exp normalization
↓
probability distribution
```

---

## 12. Top-p Sampling

### 12.1 概要

Top-p Sampling は累積確率が p を超える token 集合のみを候補として sampling を行う。

別名:

- nucleus sampling

---

### 12.2 目的

- 低確率 token 抑制
- 暴走生成防止
- 自然な文章生成

---

### 12.3 動作

```text
probabilities
↓
sort descending
↓
累積確率計算
↓
top-p 到達まで保持
↓
候補集合から sampling
```

---

## 13. Random Sampling

Sampler は乱数を利用して確率的 sampling を行う。

用途:

- 多様な生成
- 創造的生成
- 非 deterministic 応答

---

## 14. RNG 管理

sampler モジュールは内部 RNG 状態を保持する。

用途:

- sampling randomness
- reproducibility
- deterministic replay

---

## 15. 確率分布管理

sampler モジュールは token ごとの確率分布を管理する。

用途:

- token selection
- top-p filtering
- probability ranking

---

## 16. vocab_size 利用

vocab_size は sampling 範囲を制御する。

利用箇所:

| 処理 | 用途 |
|---|---|
| softmax | 配列サイズ |
| sampling | token 範囲 |
| logits | 出力次元 |

---

## 17. Top-p 候補管理

Sampler は top-p 用候補配列を管理する。

用途:

- probability sort
- cumulative probability
- token filtering

---

## 18. 推論時利用

sampler モジュールは generate/chat モジュールから利用される。

利用フロー:

```text
Transformer
↓
logits
↓
Sampler
↓
next token
↓
Tokenizer.decode
```

---

## 19. 入力と出力

### 19.1 入力

```text
logits
temperature
top-p
vocab_size
rng_state
```

### 19.2 出力

```text
next token
probability distribution
sampling result
```

---

## 20. エラー処理

sampler モジュールは以下をエラーとして扱う。

| 条件 | 処理 |
|---|---|
| vocab_size 不正 | 終了 |
| logits 異常 | 終了 |
| probability overflow | clamp |
| top-p 不正 | default 値使用 |
| temperature 不正 | 補正 |

---

## 21. メモリ管理

sampler モジュールは以下を管理する。

| リソース | 内容 |
|---|---|
| probability buffer | softmax 結果 |
| top-p buffer | candidate token |
| RNG state | random state |

推論終了後にバッファを解放する。

---

## 22. 非機能要件

| 項目 | 要件 |
|---|---|
| 安定性 | 数値オーバーフローを回避すること |
| 効率性 | 高速 sampling を実現すること |
| 再現性 | seed による deterministic 実行を可能にすること |
| 拡張性 | 新 sampling 手法を追加可能であること |
| 保守性 | sampling 処理を独立管理すること |

---

## 23. 他モジュール依存

sampler モジュールは以下で利用される。

```text
sampler
├── generate
├── chat
└── transformer
```

---

## 24. 将来拡張

sampler モジュールは将来的に以下へ対応可能とする。

- beam search
- repetition penalty
- frequency penalty
- presence penalty
- contrastive decoding
- speculative decoding
- constrained decoding
- bad word filtering
- grammar constrained decoding
- adaptive sampling
- entropy based sampling