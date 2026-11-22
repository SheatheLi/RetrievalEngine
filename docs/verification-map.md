# 验证路线｜Exact Top-K、量化界与性能对照

这个仓库把**检索正确性、候选精算数量、CPU时间和调用粒度**分开验证，不用一个加速比替代全部结论。

## 必须分别核对的证据

| 命题 | 复核位置 | 解释边界 |
|---|---|---|
| Exact Top-K一致性 | `src/certified_quantized.py`、`tests/test_certified_quantized.py` | 只针对报告指定的向量与查询条件 |
| 保存的模型与IVF候选处理 | `tests/test_ivf_fidelity.py`、`src/exact_diagnostic.py` | 依赖LightGBM读取保存的ranker；不等于FAISS原生路径已通过 |
| 向量化CPU成本 | `benchmarks/certified_quantized_runtime.py`、`evidence/certified_quantized_vectorization_benchmark.json` | 397条查询整个批次、7轮、中位时间；不是单次请求延迟 |
| 外部库比较 | `benchmarks/sklearn_exact_baseline.py`、`evidence/sklearn_external_api_comparison_20261112.json` | 逐查询与批量API不得在未对齐契约时直接比较 |

在Digits/PCA24代理数据上，已记录的397次查询批次中，旧Python路径47.60ms、向量化路径19.70ms；397/397个Top-10结果一致。以上不是线上推荐相关性或真实服务SLA。

## CI修复与复核

首次公开部署的Smoke在**收集测试**时失败：`tests/test_ivf_fidelity.py → src/exact_diagnostic.py`导入`lightgbm`但基础依赖未安装。本次把`lightgbm>=4.6,<5`加入基础`requirements.txt`，让所选Smoke测试具备必要依赖；这仍需等待GitHub Actions重新执行，**不能仅因文件已修改就宣称CI通过**。

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_retrieval.py tests/test_query_contract.py tests/test_ivf_fidelity.py tests/test_evaluation_protocol.py
```
