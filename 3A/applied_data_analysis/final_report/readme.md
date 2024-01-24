# 期末レポート
## problem1: 課題1
- 各問いとファイル(`config/`と`result/`内)の対応は以下
  - 1_1: (1)
  - 2_1: (2)(ア)
  - 2_2: (2)(イ)
  - 2_3: (2)(ウ)
  - 2_4: (2)(エ)
  - 2_5: (2)(オ)
  - 2_6: (2)(カ)
  - 3_0: (3)ベースライン
  - 3_1: (3)(ア)
  - 3_2: (3)(イ)
  - 3_3: (3)(ウ)
  - 3_4: (3)(エ)
  - 3_5: (3)(オ)
- `config/`内のファイルは各問のパラメータ設定、`result/`内のファイルは各問の結果を示す

### 実行方法
ここで
```
$ poetry run python final_report/problem1/main.py final_report/problem1/config/config(hoge).yaml
```
とすると、`config/`内の`config(hoge).yaml`を読み込んで実行する。
結果は`result/`内にhogeに対応したファイル名で保存される。

## problem2: 課題2
- `data/`内のファイルは問題文に記載されているデータセット