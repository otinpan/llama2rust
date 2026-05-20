# Transformer 要件仕様書

## 1. 概要

### 1.1 システム名
Transformer モデル

### 1.2 目的
自然言語処理において、入力文を別の表現へ変換するためのニューラルネットワークモデルを提供する。

主な用途:
- 機械翻訳
- テキスト生成
- 要約
- 質問応答
- コード生成

### 1.3 特徴
- Self-Attention による文脈理解
- Encoder / Decoder 構造
- 並列計算可能
- 長距離依存関係の学習
- Multi-Head Attention による多視点特徴抽出

```
Usage:   {program} <checkpoint> [options]
Example: {program} model.bin -n 256 -i "Once upon a time"
Options:
-t <float>  temperature in [0, inf), default 1.0
-p <float>  top-p sampling value in [0, 1], default 0.9
-s <int>    random seed, default 0
-n <int>    number of steps to run for, default 256; 0 = max_seq_len
-i <string> input prompt
-z <string> optional path to custom tokenize
-m <string> mode: generate|chat, default generate
-y <string> optional system prompt in chat mode
```

---

# 2. システム構成

## 2.1 全体構成

Transformer は以下で構成される。

```text
入力文
  ↓
Embedding
  ↓
Positional Encoding
  ↓
Encoder Stack
  ↓
Decoder Stack
  ↓
Linear Layer
  ↓
Softmax
  ↓
出力単語
```

学習済みモデル
* wq,wk,wv,wo: attentionの学習済み線形行列
* w1,w2,w3: FFNの学習済み行列。SwiGLU用
* rms_att_weight,rms_ffn_weight,rms_final_weight: 正規化学習済みスケール
* token_embedding_table: トークン埋め込み
* wcls: logitsを出す分類器

---

# 3. Encoder 要件

## 3.1 Encoder Stack

### 要件
- Encoder を複数層積み重ねる
- 各 Encoder は同一構造を持つ
- 重みは共有しない

### 標準構成
- Encoder 層数: 6

---

## 3.2 Encoder 内部構成

各 Encoder は以下で構成される。

```text
入力
 ↓
Self-Attention
 ↓
Add & LayerNorm
 ↓
Feed Forward Network
 ↓
Add & LayerNorm
 ↓
出力
```

---

## 3.3 Self-Attention

### 目的
入力文中の他単語との関連性を計算する。

### 入力
- Embedding Vector

### 出力
- Attention を反映した特徴ベクトル

### 処理

#### Step1: Q/K/V生成

各単語ベクトル x に対して:

```math
Q = x@W_Q
K = x@W_K
V = x@W_V
```

### 要件
- Query Vector を生成する
- Key Vector を生成する
- Value Vector を生成する

---

#### Step2: Attention Score 計算

```math
score = QK^T
```

### 要件
- Query と Key の内積を計算する

---

#### Step3: Scaling

```math
scaled = \frac{QK^T}{\sqrt{d_k}}
```

### 要件
- 勾配安定化のためスケーリングを行う

---

#### Step4: Softmax

```math
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
```

### 要件
- 確率分布へ正規化する
- 総和が1になること

---

#### Step5: Weighted Sum

### 要件
- 各 Value Vector に Attention Weight を適用
- 加重和を出力する

---

# 4. Multi-Head Attention 要件

## 4.1 概要

### 目的
異なる特徴空間を同時に学習する。

---

## 4.2 Head 構成

### 要件
- 複数 Head を持つ
- 各 Head は独立した Q/K/V 重みを持つ

標準:
- Head 数: 8

---

## 4.3 Multi-Head 処理

```text
Head1 Attention
Head2 Attention
...
Head8 Attention
   ↓
Concat
   ↓
WO による線形変換
```

### 要件
- 各 Head の出力を結合する
- 結合後に線形変換する

---

# 5. Feed Forward Network 要件

## 5.1 概要

### 目的
各位置の特徴変換を行う。

### 要件
- 各トークン位置へ独立適用
- 並列実行可能

---

# 6. Positional Encoding 要件

## 6.1 目的

Transformer は単語順序を直接扱えないため、
位置情報を付加する。

---

## 6.2 要件

- 各 Embedding に位置ベクトルを加算
- 位置ごとに異なる値を持つ
- 長文にも対応可能であること

---

## 6.3 数学的定義

```math
PE(pos,2i)=sin(pos/10000^{2i/d_model})
```

```math
PE(pos,2i+1)=cos(pos/10000^{2i/d_model})
```

---

# 7. Decoder 要件

## 7.1 Decoder Stack

### 構成

```text
Masked Self-Attention
 ↓
Encoder-Decoder Attention
 ↓
Feed Forward Network
```

---

## 7.2 Masked Self-Attention

### 目的
未来単語を参照しないよう制御する。

### 要件
- 未来位置を Mask する
- softmax 前に `-inf` を適用

---

## 7.3 Encoder-Decoder Attention

### 目的
入力文の relevant 部分へ注目する。

### 要件
- Query: Decoder 出力
- Key/Value: Encoder 出力

---

# 8. 出力層要件

## 8.1 Linear Layer

### 目的
Decoder 出力を Vocabulary 空間へ写像する。

### 要件
- vocab_size 次元へ変換

---

## 8.2 Softmax

### 要件
- 出力確率を生成
- 最大確率単語を次単語とする

---

# 9. Embedding 要件

## 9.1 概要

### 要件
- 各単語を固定次元ベクトルへ変換
- 埋め込み次元は d_model

標準:
- d_model = 512

---

# 10. 並列化要件

## 10.1 並列実行

### 要件
- Self-Attention は行列演算化する
- Feed Forward はトークン単位で並列可能

---

# 11. ハイパーパラメータ

| 項目 | 標準値 |
|---|---|
| Encoder Layers | 6 |
| Decoder Layers | 6 |
| Embedding Dimension | 512 |
| Attention Head 数 | 8 |
| Q/K/V Dimension | 64 |

---

# 12. 入出力仕様

## 12.1 入力

### 形式
- Token Sequence

例:

```text
["I", "am", "student"]
```

---

## 12.2 出力

### 形式
- 次単語確率分布

例:

```text
P("I") = 0.7
P("You") = 0.1
...
```

---

# 13. 用語定義

| 用語 | 意味 |
|---|---|
| Query | 注目したい特徴 |
| Key | 比較対象特徴 |
| Value | 実際に集約する情報 |
| Attention | 単語間関連度 |
| Head | 独立 Attention 空間 |
| Embedding | 単語ベクトル |
| Logits | Softmax 前スコア |

---

