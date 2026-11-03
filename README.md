# quantum_tools

thrifty-qiskit (量子回路の呼び出し回数を減らす道具)
causal-qiskit (量子因果推論の数値検証環境) のコード一式．

## 構成:

```
quantum_tools/
  thrifty_qiskit/   回路呼び出しの削減 (16 モジュール)
  causal_qiskit/    量子因果推論 L2/L3 (4 モジュール)
  experiments/      ベンチマークと図の生成 (7 スクリプト)
  tests/            pytest (20 ファイル, 314 件)
  archive/          旧版 (変更せずに保存)
  conftest.py       pytest がこのディレクトリを import の起点にするための空ファイル
```

causal_qiskit は thrifty_qiskit の `gram_resampling` (EvalCache) と `shadows` (基底変換行列) を使う．逆向きの依存はありません．

## 実行の仕方:

すべて quantum_tools/ の直下で実行する．

```
cd ./quantum_tools
~/venv/bin/python3 -m pytest -q tests
~/venv/bin/python3 -m experiments.l2_benchmark
```

スクリプトは `python3 experiments/l2_benchmark.py` ではなく `python3 -m experiments.l2_benchmark` の形で呼んでください．前者では thrifty_qiskit と causal_qiskit が見つからないことになります．なお、ログと JSON は実行したディレクトリ (quantum_tools_vN/ の直下) に出ます．

