# RetrievalEngine

[**在线交互展示**](https://sheatheli.github.io/RetrievalEngine/) · [**工程证据与验证索引**](docs/verification-map.md) · [![Smoke CI](https://github.com/SheatheLi/RetrievalEngine/actions/workflows/smoke.yml/badge.svg)](https://github.com/SheatheLi/RetrievalEngine/actions/workflows/smoke.yml)

> 将向量召回、量化残差界和精确Top-K认证结合的可执行检索研究系统。项目同时审查**排名正确性、候选计算量、CPU运行时间和API调用粒度**，避免用单个“加速比”替代全部性能结论。

**工程状态：** Python检索与推荐算法实验、模型工件、回归测试及代理数据基准。未证明面向线上推荐业务、FAISS/HNSW或大规模GPU部署的通用性能优势。

## 问题与设计

量化或近似分数可以减少计算，但直接根据近似值裁剪候选，可能错过真正应该进入Top-K的结果。完全精算又带来额外计算与Python循环成本。

项目把问题分为以下环节：

```text
Embedding/索引输入
      ↓
候选召回（KMeans / IVF-like）
      ↓
量化分数与残差误差界
      ↓
不能进入Top-K的候选被安全剪枝
      ↓
剩余候选精确评分（向量化）
      ↓
Exact Top-K 对照 + 延迟/计算次数度量
```

- **正确性：** 通过上下界判定是否允许裁剪，对照原始精度Exact排序。
- **效率：** 区分精确评分次数与实际CPU耗时，观察Python逐项操作和NumPy向量化的差别。
- **对照：** 区分逐查询API与批量查询API，避免把不同调用契约直接排名。

## 快速运行

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_retrieval.py tests/test_query_contract.py tests/test_ivf_fidelity.py
python benchmarks/compare.py
```

量化证书路径及其CPU比较：

```bash
python -m pytest -q tests/test_certified_quantized.py
python benchmarks/certified_quantized_runtime.py
```

复现实验应控制BLAS线程数（`OPENBLAS_NUM_THREADS`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS`均为1）；Windows PowerShell需使用`$env:OPENBLAS_NUM_THREADS='1'`等语法。对应脚本还需要scikit-learn及其依赖。

## 关键源码

| 文件 | 职责 |
|---|---|
| [`src/retrieval.py`](src/retrieval.py) · [`src/recommender.py`](src/recommender.py) | 检索、表示与排序评估 |
| [`src/certified_quantized.py`](src/certified_quantized.py) | 量化残差证书、上下界与安全剪枝 |
| [`benchmarks/certified_quantized_runtime.py`](benchmarks/certified_quantized_runtime.py) | Python旧路径、向量化与密集精算对照 |
| [`benchmarks/sklearn_exact_baseline.py`](benchmarks/sklearn_exact_baseline.py) | scikit-learn外部精确API对照 |
| [`models/`](models/) · [`tests/`](tests/) · [`evidence/`](evidence/) | 示例工件、测试与实验收据 |

## 已记录结果

在**Digits/PCA24代理数据、1400索引向量、397次查询、K=10、7轮计时、单线程BLAS约束**下：

| 指标 | 旧Python路径 | 向量化路径 | 密集NumPy精算 |
|---|---:|---:|---:|
| 397次查询总耗时中位数 | 47.60 ms | **19.70 ms** | 38.21 ms |
| 与密集Exact的Top-10一致性 | 397/397 | **397/397** | 参照值 |

结果来源：[`evidence/certified_quantized_vectorization_benchmark.json`](evidence/certified_quantized_vectorization_benchmark.json)。**这是整个397次查询批次的计时，不是单次请求延迟或在线服务SLA。**另外，外部scikit-learn批量API在对应实验中显著快于项目逐查询执行路径，详见[`evidence/sklearn_external_api_comparison_20261112.json`](evidence/sklearn_external_api_comparison_20261112.json)。

早期流传但不能对齐原始七轮数据的`44.79/19.02 ms`**不并入本README的性能结果**。项目也没有证明真实用户相关性或FAISS竞争优势。

## 交互展示

- [独立技术展示与复现实验说明](docs/README.md)（发布后可通过GitHub Pages交互阅读）。
- [性能图（SVG）](docs/assets/figures/RetrievalEngine.svg)与[外部API对照图](docs/assets/figures/RetrievalEngine_外部基线.svg)。

**授权状态：** 根目录暂无LICENSE，公开许可证需单独决定。
