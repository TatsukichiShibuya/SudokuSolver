# SudokuSolver

## How to use
1. solve.pyを実行する
2. "train or solve? -> "に対して、モデルが学習し終わっているならsolve、まだならtrainを入力する
3. (train)imageの画像が90倍に拡張され、traindataにjoblibで保存される -> epochを選択し学習する
4. (solve)青い四角の枠にときたい問題を入れる(可能な限り水平にし、枠に被らない程度に大きくする)

### image
* 大きさは32×32にしておく
* データを作りたいときはsudokuSolverの50行目あたりでnumを保存すれば良い
