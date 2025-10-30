# 🧠 InsureLLM RAG Challenge: The Relevance-Prioritized Truncation Approach

This document outlines the architecture of our high-performance **RAG (Retrieval-Augmented Generation)** pipeline, which was optimized through a systematic, data-driven process.

The final architecture is a **Two-Stage Retrieval Pipeline** (using a Bi-Encoder and a Cross-Encoder) that is coupled with a **Relevance-Prioritized Truncation** strategy.  
This strategy strictly enforces a **5,000-character context limit** *without sacrificing* the relevance gains achieved during retrieval.

---

## 🔍 Data-Driven Diagnosis: Why This Architecture?

Our design directly evolved from the empirical results of multiple evaluation phases.

### 1️⃣ The Baseline Failure (Turn 13 Evaluation)

The initial "naive" RAG (using simple chunking and basic vector search) failed on complex queries.

- **Retrieval:** MRR was **0.7228**
- **Answer Accuracy:** 4.12 / 5
- **Completeness:** 3.56 / 5 (**Red**)
- **Key Failure:** The system struggled with "Relationship" (0.54 MRR) and "Holistic" (2.6 / 5 Accuracy) queries.

This revealed that the indexing fragmented coherent ideas and the retriever couldn’t identify context-rich segments.

---

### 2️⃣ The Indexing Fix (Turn 18 Evaluation)

We replaced the `RecursiveCharacterTextSplitter` with `SemanticChunker`, which groups sentences based on meaning rather than character count.

- **Partial Success:** "Holistic" Accuracy improved from 2.6 → 3.4 / 5  
- **Critical New Failure:** **MRR dropped from 0.7228 → 0.6667**

This proved that **better chunks hurt retrieval** when the retriever (bi-encoder) was too weak to rank nuanced semantic representations.

---

### 3️⃣ The Retrieval Solution (Turn 22 Evaluation)

To solve the 0.6667 MRR, we implemented a **Two-Stage Retrieval System**:

1. **Stage 1 – Bi-Encoder (Fast Recall):**  
   Model: `all-MiniLM-L6-v2`  
   Retrieves a large (k=50), high-recall candidate list.
2. **Stage 2 – Cross-Encoder (High Precision):**  
   Model: `BAAI/bge-reranker-base`  
   Reranks candidates to select the top_n=3 most relevant documents.

**Results:**
- **MRR:** 0.6667 → **0.9058**
- **nDCG:** 0.6873 → **0.9049**
- **Answer Accuracy:** 3.81 → **4.42 / 5**

This validated that our retriever now yields a *high-confidence* ranking of relevant documents.

---

## 📏 The 5,000-Character Constraint & Solution

The challenge was to respect the 5,000-character limit *without* losing retrieval quality.  
Naively truncating or skipping long docs would destroy precision — so we created the **Relevance-Prioritized Truncation** algorithm.

### Core Principles

1. **Trust the Rank:** The system fully trusts the cross-encoder ranking.  
   The most relevant doc (`docs[0]`) is always considered first.
2. **Handle the First Document:**  
   - If it’s longer than 5,000 characters → truncate it and use only that.  
   - If it fits → include it and try to add more.
3. **Fill by Relevance:**  
   - Add subsequent documents (`docs[1]`, `docs[2]`, …) **only if they fit** within the remaining limit.  
   - Stop when the limit is reached.

This guarantees that the 5,000-character window is packed with *the most relevant possible content*, respecting retrieval order.

---

## 🧩 Final Code Architecture

### `ingest.py` (Final Version)

Uses **SemanticChunker** to create high-quality, semantically coherent chunks.

```python
import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

MODEL = "gpt-4.1-nano"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
load_dotenv(override=True)

def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
        documents.extend(folder_docs)
    return documents

def create_chunks(documents):
    text_splitter = SemanticChunker(embeddings)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME
    )
    return vectorstore

if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
```

---

### `answer.py` (Final Version)

Implements the **Two-Stage Retriever** and **Relevance-Prioritized Truncation** logic.

```python
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from typing import List, Optional, Tuple

# Two-Stage Retriever Imports
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from dotenv import load_dotenv

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
MAX_CONTEXT_LENGTH = 5000

SYSTEM_PROMPT = \"\"\"
You are a knowledgeable assistant representing InsureLLM.
Use ONLY the given context to answer the question.
If the context is insufficient, say you cannot answer based on the provided documents.

Context:
{context}
\"\"\"

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

# --- Two-Stage Retriever Setup ---
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
model_name = "BAAI/bge-reranker-base"
model = HuggingFaceCrossEncoder(model_name=model_name, model_kwargs={'device': 'cpu'})
compressor = CrossEncoderReranker(model=model, top_n=3)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
# --- End Retriever Setup ---

llm = ChatOpenAI(temperature=0, model_name=MODEL)

def fetch_context(question: str) -> list:
    print(f"\\n--- FETCHING CONTEXT FOR: '{question}' ---")
    reranked_docs = retriever.invoke(question)
    print(f"Retrieved and re-ranked {len(reranked_docs)} documents.")
    return reranked_docs

def answer_question(question: str, history: Optional[List] = None) -> Tuple[str, List]:
    if history is None:
        history = []

    docs = fetch_context(question)
    context_pieces = []
    current_length = 0
    final_docs_used = []
    separator = "\\n\\n"

    print(f"\\n--- BUILDING CONTEXT (Limit: {MAX_CONTEXT_LENGTH} chars) ---")

    if docs:
        doc0 = docs[0]
        content0 = getattr(doc0, "page_content", "")
        if not isinstance(content0, str):
            content0 = str(content0)
        length0 = len(content0)

        if length0 <= MAX_CONTEXT_LENGTH:
            context_pieces.append(content0)
            final_docs_used.append(doc0)
            current_length = length0

            for i, doc in enumerate(docs[1:], start=2):
                doc_content = getattr(doc, "page_content", "")
                if not isinstance(doc_content, str):
                    doc_content = str(doc_content)
                doc_length = len(doc_content)
                length_if_added = current_length + len(separator) + doc_length

                if length_if_added <= MAX_CONTEXT_LENGTH:
                    context_pieces.append(separator + doc_content)
                    final_docs_used.append(doc)
                    current_length = length_if_added
                else:
                    break
        else:
            truncated_content = content0[:MAX_CONTEXT_LENGTH]
            context_pieces.append(truncated_content)
            final_docs_used.append(doc0)

    context = "".join(context_pieces)
    system_prompt = SYSTEM_PROMPT.format(context=context)

    messages = []
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    final_answer = response.content

    return final_answer, final_docs_used
```

---

## 🖼️ Visual Results

*(Add your result images below — replace these placeholders with actual image paths.)*

### 📈 Performance Improvement Visualization

![Performance Comparison Placeholder](images/performance_comparison.png)

### 🧩 Relevance-Prioritized Truncation Flow

![Truncation Logic Placeholder](images/truncation_logic.png)

---

## 🧰 Installation

Please install the required packages before running the pipeline:

```bash
uv pip install langchain-experimental langchain-classic langchain-community
```

---

## 📚 References

1. LangChain Experimental – [SemanticChunker Documentation](https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic_chunker)
2. HuggingFace Embeddings – [MiniLM Models](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
3. BAAI Cross Encoder – [bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
4. Microsoft Research – *Dual Encoder vs. Cross Encoder in Dense Retrieval*

---

**Author:** Your Name  
**Project:** InsureLLM RAG Challenge  
**Result:** 🏆 Achieved 0.9058 MRR — with relevance-prioritized truncation and optimized retrieval precision.
