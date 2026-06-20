# MRAgent: Memory is Reconstructed, Not Retrieved — Graph Memory for LLM Agents

**arXiv:2606.06036 | ICML 2026**
作者: Shuo Ji, Yibo Li, Bryan Hooi (NUS)
代码: https://github.com/Ji-shuo/MRAgent

---

## 核心思想

从 **被动检索 (passive retrieve-then-reason)** 转向 **主动记忆重建 (active memory reconstruction)**。

**当前问题**：所有现有记忆系统都是"一次性检索"。查询来了 → 算相似度 → 取 top-k → 送入 LLM。检索路径在一开始就固定了，LLM 不能根据中间发现调整搜索方向。

**核心创新**：把 LLM 推理嵌入到记忆访问过程中，让 Agent 在检索时就能根据已找到的证据决定下一步往哪儿搜，同时避免组合爆炸。

---

## 框架设计

### 1. Cue-Tag-Content 图（Associative Memory Graph）

```
Cue (线索)     Tag (标签)        Content (内容)
  "Nate"  ───→  "gaming"   ───→  参加周末比赛
  "July"  ───→  "event"    ───→  Caroline 去度假
  "Jonna" ───→  "screenplay"───→  投了剧本
         ───→  "rejection" ───→  被拒了
```

- **Cue**：细粒度关键词（实体、属性等）
- **Tag**：语义桥接标签（创新点！Link Cue → Content 的中间层）
- **Content**：三层记忆内容
  - **Episodic Layer**（具体事件、时间线）
  - **Semantic Layer**（抽象知识、个人属性、偏好）
  - **Abstraction Layer**（话题级摘要）

### 2. Active Reconstruction 过程

```
Step 1:  查询提取 Cue → "Jonna"
Step 2:  LLM 选择 Tags → ["screenplay", "rejection"]
Step 3:  展开 Tag→Content → 发现投了剧本、被拒了
Step 4:  根据中间证据推理 → 需要时间线索 "July"
Step 5:  用新线索继续查 → 完整回答
```

关键操作：
- **Forward traversal**: Cue→Tag, (Cue,Tag)→Content
- **Reverse traversal**: Content→(Cue,Tag) backpropagation
- **LLM action selection**: 选哪些路径展开
- **LLM routing**: 从候选集中选相关、剪无关

### 3. 理论贡献

**定理 4.1**: 主动检索严格强于被动检索。给定相同预算 T，主动策略的表达力严格包含被动策略。

证明用了 Binary-Tree Needle-in-a-Haystack 任务族：主动策略零误差，被动策略除非预算指数级增长否则有不可约误差。

---

## 实验结果

### LoCoMo 基准

| 方法 | 多跳 F1 | 时间推理 F1 | 总评分 (J) |
|:---|:---:|:---:|:---:|
| RAG | 34.89 | 43.52 | 61.30 |
| Mem0 (最强基线) | 45.17 | 58.19 | 68.31 |
| **MRAgent** | **56.72** | **69.82** | **88.32** |

### LongMemEval 基准

| 方法 | 多会话 | 单会话 | 时间推理 | 偏好 | 总体 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **MRAgent** | **68.42** | **92.85** | **68.42** | **66.67** | **72.95** |
| MRAgent* | 86.46 | 92.85 | 85.71 | 78.57 | 86.76 |

### Token 成本（越低越好）

| 方法 | Token 消耗 | 运行时间 |
|:---|:---:|:---:|
| A-Mem | 632k | 1,122s |
| MemoryOS | 273k | 3,135s |
| LangMem | 3,268k | 1,209s |
| Mem0 | 245k | 533s |
| **MRAgent** | **118k** | **586s** |

**Token 最少**，因为按需展开，不需要一次性强塞。

### 消融实验关键发现

1. **多步推理 > 记忆结构本身**（有推理比无推理高出 30%+）
2. **Tag 提供有效语义引导**（Cue-Tag-Content > Cue-Episode）
3. **Episodic + Semantic 互补**（去掉 Semantic 层明显下降）

---

## 局限

- 多步遍历时延迟比单步检索高
- 记忆图只增不减，长期运行存储膨胀
- 没有遗忘/更新机制
- 当前实现记忆构造策略较简单

---

## 启发

1. **session_search 可改进**：先搜标签确定方向，再展开内容
2. **迭代检索**：找到线索A → 推断B → 用B继续
3. **Hermes memory 可借鉴**：当前是扁平关键词匹配，可以加标签层
4. **与之前"AI工程层级图"对应**：Context Engineering + Logic Engineering
