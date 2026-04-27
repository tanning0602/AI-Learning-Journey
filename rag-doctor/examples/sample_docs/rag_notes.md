# RAG Doctor Notes

RAG Doctor evaluates retrieval augmented generation pipelines. It checks whether the retriever found relevant chunks, whether the answer is grounded in citations, and whether the result has hallucination risk.

The first version uses a simple hashing embedder. This makes the project easy to run without external services. Later versions can replace the embedder with bge, sentence-transformers, OpenAI embeddings, FAISS, or Chroma.

Good RAG systems need reliable document loading, careful chunking, useful embeddings, strong retrieval, citation-aware answers, and repeatable evaluation.

RAG Doctor 可以帮助定位 RAG 应用为什么答错，例如检索没有命中、上下文太弱、答案没有引用、或者存在幻觉风险。
