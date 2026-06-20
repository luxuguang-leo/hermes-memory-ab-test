# Hermes Memory A/B Test

MRAgent（ICML 2026）对比测试：主动图记忆 vs 轻量实体索引 vs 扁平检索。

## 测试的三个策略

| 策略 | 做法 | LLM 调用 |
|:---|---|:---:|
| **MRAgent** | Cue-Tag-Content 图 + LLM 多步遍历 | 3 次 |
| **Flat RAG** | 关键词匹配 + 单次回答 | 2 次 |
| **改进方案** | jieba 实体索引 + 两段检索 | 1 次 |

## 结果概要

| 指标 | 改进方案 | MRAgent | Flat RAG |
|:---|:---:|:---:|:---:|
| Correct+Partial | **80%** | 30% | 40% |
| 平均耗时 | **11s** | 33s | 4s |
| 稳定运行 | **10/10** | 5/10 | 5/10 |

详细结果见 `results/results.json`。

## 目录

```
test_data/      10 个测试用例（多跳/时间/跨会话）
strategies/     三种策略的 Python 实现
results/        完整运行结果
article/        文章正文
skill/          Hermes Skill 封装
references/     论文笔记
```

## 运行

需 Ollama + Gemma4 12B（或其他兼容模型）：

```bash
cd strategies
python3 ab_test_v3.py          # MRAgent vs Flat RAG
python3 ../skill/run_improved.py  # 改进方案
```

## 论文

- MRAgent: [Memory is Reconstructed, Not Retrieved](https://arxiv.org/abs/2606.06036) — ICML 2026
- LoCoMo: [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753) — ACL 2024
- LongMemEval: [Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813) — ICLR 2025

## 协议

MIT
