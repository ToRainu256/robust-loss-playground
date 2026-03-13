# robust-loss-playground: 実装設計書（コードエージェント用）

## 0. この文書の目的

本書は、`robust-loss-playground` を実装するコードエージェントに渡すための設計仕様書である。目的は、単なる loss 関数集ではなく、**研究者がそのまま実験・論文・OSS に使いたくなる robust loss ライブラリ**を構築することである。

本ライブラリは次を満たさねばならない。

* robust loss を **統一 API** で扱えること
* 研究用途で重要な **loss 値・influence function・weight function** を取得できること
* **Mac の CPU 環境で開発・検証可能** でありつつ、**GPU マシン上でそのまま動作**すること
* 将来的な **custom CUDA kernel / fused op / torch.compile** への発展を阻害しないこと
* 数式仕様が曖昧でなく、**docs と unit test が仕様書として機能**すること

本書は「望ましい実装」ではなく、原則として **従うべき仕様** を記述する。

---

## 1. プロジェクトのビジョン

### 1.1 何を作るか

`robust-loss-playground` は、PyTorch と NumPy を基盤とした研究指向の robust loss ライブラリである。

単なる `HuberLoss` や `CharbonnierLoss` の実装集ではなく、以下を統一的に扱う。

* residual に対する loss 関数 (\rho(r))
* influence function (\psi(r) = \partial \rho(r)/\partial r)
* IRLS 的 weight function (w(r))
* 可視化（loss / influence / weight curve）
* toy 実験（外れ値を含む回帰など）

### 1.2 誰のためのライブラリか

主な利用者は以下である。

* CV / 3D vision / rendering / 3DGS の研究者・開発者
* M-estimation や robust optimization に関心のある機械学習研究者
* loss 関数の数式・微分・外れ値耐性を比較したい人

### 1.3 設計思想

このライブラリは以下の思想に基づく。

1. **residual-first** であること
   prediction/target よりも residual を第一級オブジェクトとして扱う。

2. **M-estimation の言葉で統一**すること
   `rho`, `influence`, `weight` を主 API とする。

3. **数式仕様を固定**すること
   各 loss の定義を docs に明記し、unit test と一致させる。

4. **CPU で開発し GPU に自然移行**できること
   device / dtype を壊す実装を避ける。

5. **まず Python/PyTorch で正しく実装し、その後高速化**すること
   初期段階では custom CUDA を前提としない。

---

## 2. スコープ

### 2.1 初期バージョンのスコープ内

初期版（v0.1 系）では以下を実装対象とする。

* PyTorch 実装
* NumPy 参照実装
* 基底クラス
* reduction (`none`, `mean`, `sum`)
* 以下の loss

  * L2
  * L1
  * Huber
  * Charbonnier
  * Cauchy
  * Tukey biweight
* plotting utility
* toy regression example
* unit test
* 数式ドキュメント

### 2.2 初期バージョンでは後回し

以下は設計上は考慮するが、v0.1 では無理に実装しない。

* JAX backend
* custom `torch.autograd.Function`
* C++/CUDA extension
* fused kernel
* mixed precision 最適化
* distributed training 向け最適化
* Barron general loss
* mixture loss

ただし API は将来拡張を阻害してはならない。

---

## 3. 数学的統一仕様

### 3.1 基本変数

* residual: (r)
* scale: (s > 0)
* normalized residual: (u = r / s)

### 3.2 統一規約

本ライブラリでは、loss を次の規約で定義する。

[
\rho(r; s) = s^2 \phi\left(\frac{r}{s}\right)
]

ここで (\phi) は標準化 residual (u) 上で定義された基底関数である。

このとき influence function は

[
\psi(r; s) = \frac{\partial \rho(r; s)}{\partial r} = s \phi'(u)
]

weight function は

[
w(r; s) = \frac{\psi(r; s)}{r}
]

とする。ただし (r=0) では極限値を採用する。

### 3.3 この規約を採用する理由

* scale の意味が明確になる
* 単位・次元の整合性が取りやすい
* Huber / Charbonnier / Cauchy / Barron を同一視点で記述できる
* 研究用ドキュメントとして美しい

### 3.4 実装上の注意

* `rho()` は **要素ごとの loss 値**を返す
* `forward()` は `rho()` に reduction を適用した結果を返す
* `influence()` は解析的に与える
* `weight()` は数値安定化込みで与える

---

## 4. API 仕様

### 4.1 基本方針

API は class ベースを主とし、必要なら後で functional wrapper を足す。まずは class API を安定させる。

利用例は次を目標とする。

```python
import torch
from robust_loss.torch import Charbonnier

r = torch.randn(1024)
loss_fn = Charbonnier(scale=1.0, eps=1e-3, reduction="mean")
loss = loss_fn(r)
psi = loss_fn.influence(r)
w = loss_fn.weight(r)
```

### 4.2 中核 API

各 loss クラスは次を持たねばならない。

* `forward(residual)`
* `rho(residual)`
* `influence(residual)`
* `weight(residual, eps=...)`

### 4.3 reduction

`reduction` は以下のみを許容する。

* `"none"`
* `"mean"`
* `"sum"`

`forward()` は `rho()` に reduction をかけたものとする。

### 4.4 入力について

基本入力は residual tensor / ndarray とする。`pred, target` を直接受ける API は初期版には不要である。

理由は以下である。

* robust loss の理論対象は residual である
* rendering / reprojection / photometric error への流用が容易である
* API が最小かつ明確になる

---

## 5. ディレクトリ構成

以下の構成で実装せよ。

```text
robust-loss-playground/
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ .gitignore
├─ src/
│  └─ robust_loss/
│     ├─ __init__.py
│     ├─ types.py
│     ├─ utils.py
│     ├─ plotting.py
│     ├─ registry.py
│     ├─ torch/
│     │  ├─ __init__.py
│     │  ├─ base.py
│     │  ├─ l2.py
│     │  ├─ l1.py
│     │  ├─ huber.py
│     │  ├─ charbonnier.py
│     │  ├─ cauchy.py
│     │  └─ tukey.py
│     └─ numpy/
│        ├─ __init__.py
│        ├─ base.py
│        ├─ l2.py
│        ├─ l1.py
│        ├─ huber.py
│        ├─ charbonnier.py
│        ├─ cauchy.py
│        └─ tukey.py
├─ tests/
│  ├─ test_numpy_value.py
│  ├─ test_torch_value.py
│  ├─ test_torch_numpy_consistency.py
│  ├─ test_influence_weight.py
│  ├─ test_reduction.py
│  └─ test_device_dtype.py
├─ examples/
│  ├─ plot_loss_curves.py
│  ├─ plot_influence_functions.py
│  ├─ plot_weight_functions.py
│  └─ toy_regression_outliers.py
└─ docs/
   ├─ formulas.md
   └─ design.md
```

---

## 6. PyTorch 側の基底クラス仕様

### 6.1 要件

`src/robust_loss/torch/base.py` に PyTorch 用基底クラスを実装せよ。以下を満たすこと。

* `torch.nn.Module` を継承すること
* `scale` と `reduction` を保持すること
* `forward()` は `rho()` + reduction のみを行うこと
* `rho()` と `influence()` は具象クラスで実装すること
* `weight()` は共通実装を与えること
* device / dtype を壊さないこと

### 6.2 推奨シグネチャ

```python
class BaseRobustLoss(nn.Module):
    def __init__(self, scale: float = 1.0, reduction: str = "mean") -> None: ...
    def forward(self, residual: Tensor) -> Tensor: ...
    def rho(self, residual: Tensor) -> Tensor: ...
    def influence(self, residual: Tensor) -> Tensor: ...
    def weight(self, residual: Tensor, eps: float = 1e-12) -> Tensor: ...
    def normalized_residual(self, residual: Tensor) -> Tensor: ...
```

### 6.3 振る舞い仕様

* `scale <= 0` は `ValueError`
* 不正な `reduction` は `ValueError`
* `normalized_residual(residual)` は `residual / self.scale`
* `weight()` は (r=0) 近傍で安全に計算する

### 6.4 `weight()` の扱い

`weight()` は

[
w(r) = \psi(r) / r
]

を数値安定化付きで実装すること。

実装方針は次である。

* `abs(r) > eps` では `psi / r`
* `abs(r) <= eps` では `r -> 0` の極限値を返す

デフォルトでは `_weight_limit_at_zero()` を持ち、必要なら具象クラスで override できるようにせよ。

---

## 7. NumPy 側の基底クラス仕様

### 7.1 役割

NumPy 実装は高速化のためではなく、**参照実装** である。以下を目的とする。

* 数式仕様の明確化
* PyTorch 実装との一致確認
* plotting
* docs と notebook での軽量利用

### 7.2 方針

* PyTorch 実装とできるだけ同じメソッド名を使う
* `__call__()` は `rho()` + reduction を返す
* shape semantics は PyTorch と一致させる

---

## 8. 各 loss の数学仕様

この節の定義を docs と test の基準とせよ。式の取り違えは許容しない。

### 8.1 L2

標準化関数を

[
\phi(u) = \frac{1}{2}u^2
]

とする。したがって

[
\rho(r;s)=\frac{1}{2}r^2
]

である。influence は

[
\psi(r)=r
]

である。

### 8.2 L1

[
\phi(u)=|u|
]

したがって

[
\rho(r;s)=s|r/s|=s|u|\times s = s^2|u|/s
]

ここは規約上やや扱いが微妙である。実装では **L1 は特別扱いで (\rho(r)=|r|)** としてよい。L1 は scale 統一規約の例外として明示すること。

influence はサブグラディエントとして

[
\psi(r)=\operatorname{sign}(r),\quad \psi(0)=0
]

とする。

### 8.3 Huber

標準化 residual (u=r/s) とし、しきい値 (\delta>0) に対し

[
\phi(u)=
\begin{cases}
\frac{1}{2}u^2 & |u| \le \delta \
\delta(|u|-\frac{1}{2}\delta) & |u| > \delta
\end{cases}
]

とする。したがって

[
\rho(r;s)=s^2\phi(r/s)
]

である。

### 8.4 Charbonnier

[
\phi(u)=\sqrt{u^2+\varepsilon^2}-\varepsilon
]

とする。これにより (\phi(0)=0) となる。

したがって

[
\rho(r;s)=s^2\left(\sqrt{(r/s)^2+\varepsilon^2}-\varepsilon\right)
]

influence は

[
\psi(r;s)=s\cdot \frac{u}{\sqrt{u^2+\varepsilon^2}}
]

である。

### 8.5 Cauchy

[
\phi(u)=\frac{1}{2}\log(1+u^2)
]

とする。したがって

[
\rho(r;s)=\frac{s^2}{2}\log(1+(r/s)^2)
]

influence は

[
\psi(r;s)=\frac{r}{1+(r/s)^2}
]

である。

### 8.6 Tukey biweight

しきい値 (c>0) に対し、標準化 residual (u=r/s) を用いて

[
\phi(u)=
\begin{cases}
\frac{c^2}{6}\left[1 - \left(1 - (u/c)^2\right)^3\right] & |u| \le c \
\frac{c^2}{6} & |u| > c
\end{cases}
]

とする。

influence は

[
\psi(r;s)=s\cdot u\left(1-(u/c)^2\right)^2 \cdot \mathbf{1}(|u|\le c)
]

に対応する形で実装せよ。境界の扱いは docs と test で固定すること。

---

## 9. device / dtype 仕様

### 9.1 最重要原則

本ライブラリは **Mac CPU 上で開発**し、**GPU マシン上でそのまま実行**する前提で作る。したがって device / dtype を壊す実装は禁止する。

### 9.2 禁止事項

入力と無関係に次のようなテンソルを裸で生成してはならない。

```python
torch.tensor(1e-6)
```

これは CPU / default dtype に固定され、device mismatch や dtype mismatch の原因になる。

### 9.3 許可される実装

定数は原則として以下のいずれかで扱う。

* Python float のまま演算に使う
* 必要なら `torch.as_tensor(value, dtype=residual.dtype, device=residual.device)` を使う

### 9.4 必須テスト

最低限、以下が通ること。

* `float32` で動作
* `float64` で動作
* CPU 上で動作
* `tensor.to(device)` 後も動作

GPU 専用 CI は必須ではないが、コードは CUDA 対応でなければならない。

---

## 10. 実装の品質要件

### 10.1 コードスタイル

* Python 3.10+
* 型注釈を付ける
* `ruff` で lint を通す
* `mypy` で重大な型問題を潰す
* docstring は簡潔でよいが、数式の意味が分かるようにする

### 10.2 実装原則

* for-loop より vectorized 実装を優先
* shape-preserving に実装
* reduction は共通化
* 数学仕様は docs と test に固定
* 近似や特殊ケース分岐は明記

### 10.3 将来の高速化を見据えた書き方

* `rho()` / `influence()` の内部表現を明確に分ける
* custom CUDA 化を想定して、具象 loss ごとのコア演算を見通し良く保つ
* Python 側で過剰に抽象化しすぎない
* ただし premature optimization はしない

---

## 11. テスト仕様

### 11.1 テストの位置づけ

このライブラリでは unit test は単なる回帰確認ではなく、**数式仕様の一部**である。コードが test を通っても docs と矛盾するなら実装は誤りである。

### 11.2 必須テスト群

#### (A) 値テスト

各 loss について、小さな residual 配列に対し既知の値を確認する。

例:

```python
r = [-2.0, -1.0, 0.0, 1.0, 2.0]
```

#### (B) PyTorch と NumPy の一致

同一パラメータ・同一入力に対して、PyTorch 実装と NumPy 実装が十分近いことを確認する。

#### (C) influence と autograd の一致

PyTorch 実装では

```python
grad = autograd(rho.sum(), r)
```

と `influence(r)` が一致しなければならない。

#### (D) weight の極限挙動

`r -> 0` 近傍で不安定にならず、期待する極限値を返すことを確認する。

#### (E) reduction

`none`, `mean`, `sum` が正しく動くこと。

#### (F) dtype / device

`float32`, `float64` での整合性を確認する。

### 11.3 autograd 一致テストの例

各 smooth loss について以下に相当する test を書け。

```python
r = torch.randn(16, dtype=torch.float64, requires_grad=True)
rho = loss_fn.rho(r)
grad = torch.autograd.grad(rho.sum(), r)[0]
psi = loss_fn.influence(r)
assert torch.allclose(grad, psi, atol=1e-8, rtol=1e-6)
```

L1 のような非滑らかな loss は零点を避けたテスト入力を使え。

---

## 12. plotting 仕様

### 12.1 目的

このライブラリは playground である以上、可視化は必須である。特に研究者は shape や influence を見たがるため、プロット API は重要である。

### 12.2 必須関数

少なくとも以下を実装せよ。

* `plot_rho(losses, xlim=(-5, 5), num=1000)`
* `plot_influence(losses, xlim=(-5, 5), num=1000)`
* `plot_weight(losses, xlim=(-5, 5), num=1000)`

### 12.3 方針

* backend は NumPy でもよい
* matplotlib を用いる
* 関数は Figure / Axes を返す形が望ましい
* README の図としてそのまま使える品質にする

---

## 13. example 仕様

### 13.1 例の位置づけ

examples はデモではなく、ライブラリの価値を最短で伝える資産である。

### 13.2 必須 example

#### 1. `plot_loss_curves.py`

L2 / Huber / Charbonnier / Cauchy / Tukey の (\rho(r)) を比較する。

#### 2. `plot_influence_functions.py`

各 loss の influence function を比較する。robustness の直感が最も出る。

#### 3. `plot_weight_functions.py`

IRLS 的 weight を比較する。研究者向けには非常に価値が高い。

#### 4. `toy_regression_outliers.py`

外れ値を含む一次元回帰データを作り、L2 と robust loss で fitting 結果を比較する。README の看板として使えるようにせよ。

---

## 14. README 仕様

README は単なる導入ではなく、利用者が 2 分で価値を理解できる資料でなければならない。

### 14.1 必須内容

* ライブラリの目的
* 主要機能
* インストール方法
* 最小使用例
* `rho / influence / weight` が取れること
* プロット例
* toy regression 例
* 数式ドキュメントへの導線

### 14.2 強調すべき点

README では次を前面に出せ。

* residual-first API
* research-oriented design
* PyTorch + NumPy
* `rho`, `influence`, `weight`
* extensible to future custom kernels

---

## 15. パッケージング仕様

### 15.1 `pyproject.toml`

最低限、以下を含めよ。

* `setuptools` ベース
* `numpy`, `torch`, `matplotlib`
* dev dependency として `pytest`, `ruff`, `mypy`

### 15.2 import 体験

次が成立するように export を設計せよ。

```python
from robust_loss.torch import L2, L1, Huber, Charbonnier, Cauchy, Tukey
```

---

## 16. 実装順序

コードエージェントは以下の順で作業せよ。勝手に順序を崩さないこと。

### Phase 1: 骨格

1. `pyproject.toml`
2. ディレクトリ構成
3. PyTorch 基底クラス
4. NumPy 基底クラス
5. 共通 utility

### Phase 2: 基本 loss

6. L2
7. L1
8. Charbonnier
9. Huber
10. Cauchy
11. Tukey

### Phase 3: テスト

12. 値テスト
13. torch/numpy 一致テスト
14. influence/autograd テスト
15. reduction テスト
16. dtype/device テスト

### Phase 4: 可視化と examples

17. plotting utility
18. `plot_loss_curves.py`
19. `plot_influence_functions.py`
20. `plot_weight_functions.py`
21. `toy_regression_outliers.py`

### Phase 5: ドキュメント

22. `README.md`
23. `docs/formulas.md`
24. `docs/design.md`

---

## 17. 受け入れ条件

以下を満たしたら初期版を受け入れてよい。

1. `pytest` が通る
2. CPU 上で examples が動く
3. `rho`, `influence`, `weight` が全 loss で一貫した API を持つ
4. docs の式と実装が一致する
5. NumPy と PyTorch 実装が概ね一致する
6. コードが device / dtype を壊さない
7. README を見て研究者が用途を即理解できる

---

## 18. 禁止事項

以下は禁止する。

* 先に custom CUDA を書こうとすること
* `torch` と `numpy` の API を無意味にズラすこと
* reduction を各 loss に重複実装すること
* docs と異なる数式を採用すること
* device / dtype を暗黙に固定すること
* 使われない抽象化を増やしすぎること
* 何となく functional API を先に増やすこと

---

## 19. 将来拡張の指針

初期版完成後、次の順で拡張可能な設計にしておけ。

1. Barron general loss
2. mixture loss
3. `from_prediction(pred, target)` convenience API
4. JAX backend
5. benchmark suite
6. torch.compile 最適化
7. custom autograd / C++ / CUDA kernel

ただし初期実装では、この将来拡張のために現在の見通しを悪くしてはならない。

---

## 20. エージェントへの最終要求

実装者は、単に「動く robust loss 集」を作るのではなく、**研究者が使いたくなる最小で美しい robust loss 基盤** を作ること。

そのために最重要なのは次の 5 点である。

1. 数式が明確であること
2. `rho`, `influence`, `weight` が揃っていること
3. PyTorch と NumPy の二層構造があること
4. CPU 開発から GPU 実行へ自然に移行できること
5. README / examples が研究用途を直感的に示すこと

この 5 点を満たせない実装は、本設計書の目的を満たしていない。
