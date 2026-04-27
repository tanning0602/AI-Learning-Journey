# 04-27 学习笔记和已完成任务

这份文档对应第一天目标：先理解 RAG 的主流程，再通过一个能运行的小项目把概念串起来。

## 一、你今天要掌握的基础概念

### 1. Token

大模型不会直接理解自然语言文本，而是先把文本切成 token。英文里 token 可能接近单词、词根或符号；中文里 token 可能是字、词或子词。token 数量会影响上下文窗口、成本和速度。

在本项目中，`src/rag_doctor/embedder.py` 用一个简单 tokenizer 同时处理英文单词、数字和中文字符，目的是让你看清楚文本如何进入检索流程。

### 2. Embedding

Embedding 是把文本变成向量。语义相近的文本，向量距离通常也更近。真实项目里常用 bge、text-embedding、sentence-transformers 等模型。

本项目第一版用 `HashingEmbedder` 做无依赖 baseline。它不是最强效果，但足够展示完整流程：文本 -> token -> 向量 -> 相似度。

### 3. Attention 和上下文窗口

Transformer 的 attention 机制让模型在生成时关注上下文中的不同位置。上下文窗口是一次能放进模型的最大 token 数。RAG 的意义之一，就是从大量文档里找出少量最相关内容，再放进上下文窗口。

### 4. RAG

RAG 是 Retrieval-Augmented Generation，核心流程是：

1. 加载文档。
2. 切成 chunk。
3. 把 chunk 变成 embedding。
4. 建向量索引。
5. 用户提问。
6. 检索最相关 chunk。
7. 把检索结果交给模型生成答案。
8. 返回答案和引用。

本项目的 `qa.py` 先做抽取式答案，不调用大模型。这是故意的：你可以先学习检索和引用，不被 prompt 和模型不稳定性干扰。

### 5. Chunking

chunk 太大，检索结果会混入无关内容；chunk 太小，答案需要的上下文可能被切碎。overlap 可以缓解边界信息丢失。

本项目默认：

- `chunk_size=700`
- `overlap=120`

你可以通过 CLI 改参数：

```powershell
python -m rag_doctor.cli index examples/sample_docs --chunk-size 500 --overlap 80
```

### 6. Vector Search

向量检索就是把问题向量和文档 chunk 向量做相似度计算，返回最相近的 top-k。小规模数据可以直接内存搜索，大规模项目会用 FAISS、Milvus、Qdrant、Chroma 等。

本项目第一版用内存索引和 cosine similarity，代码在 `retriever.py`。

### 7. RAG Evaluation

RAG 不能只看“答案像不像对”。至少要看：

- Retrieval score：检索分数是否足够高。
- Context relevance：检索片段是否和问题相关。
- Faithfulness：答案是否能被引用片段支持。
- Citation quality：答案有没有明确引用来源。
- Term coverage：答案是否覆盖预期关键词。

本项目的 `evaluate.py` 会生成一个 HTML 报告，帮你观察这些指标。

## 二、04-27 已完成任务

- 已创建 `rag-doctor` 项目。
- 已实现 `.md` 和 `.txt` 文档加载。
- 已实现文档 chunking。
- 已实现无依赖 embedding baseline。
- 已实现本地向量索引和 top-k 检索。
- 已实现带引用的抽取式问答。
- 已实现问题集评估和 HTML 报告。
- 已补充示例文档和示例问题。
- 已补充 README、架构图、Quickstart 和 Roadmap。

## 三、你应该怎么学习这个项目

建议按这个顺序读代码：

1. `load_documents.py`：看文档如何进入系统。
2. `chunker.py`：看文本为什么要切块。
3. `embedder.py`：看 token 如何变成向量。
4. `retriever.py`：看相似度搜索如何返回 top-k。
5. `qa.py`：看答案如何绑定引用。
6. `evaluate.py`：看怎么评价结果。
7. `cli.py`：看工程入口如何组织。

## 四、今天必须跑通的命令

```powershell
$env:PYTHONPATH="src"
python -m rag_doctor.cli index examples/sample_docs
python -m rag_doctor.cli ask "RAG Doctor 可以帮助定位什么问题？"
python -m rag_doctor.cli eval examples/sample_questions.json --output reports/report.html
```

跑完后，你应该看到：

- `.rag-doctor/index.json`
- `reports/report.html`

## 五、明天继续做什么

第二天重点不是再堆功能，而是把项目做得更像可展示产品：

- 加 PDF 支持。
- 接真实 embedding 模型。
- 加 FAISS 或 Chroma。
- 做一个 Streamlit/Gradio 页面。
- 增加更多中文示例数据。
- 写一篇技术文章发布。
