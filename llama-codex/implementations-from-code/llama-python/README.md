# llama-python

`llama2.c` を段階的に Python へ写すための最小実装です。

現在入っているもの:

- `model.bin` の header/config 読み込み
- weight offset の切り出し
- `RunState`
- `forward(token, pos)` の最小実装
- tokenizer
- sampler
- generate ループ
- chat ループ
- 簡単な CLI

前提:

- `python3`
- `numpy`

実行例:

```bash
python3 -m llama_runner.cli /path/to/model.bin --token 1 --pos 0
python3 -m llama_runner.cli /path/to/model.bin --tokenizer /path/to/tokenizer.bin --prompt "Hello" --steps 32
python3 -m llama_runner.cli /path/to/model.bin --tokenizer /path/to/tokenizer.bin --mode chat --prompt "Hello" --system-prompt "You are concise." --steps 64
```

次に足す対象:

- C 実装との数値比較テスト
- 複数ターン会話の履歴管理
