from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document


from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

MAX_CONTEXT_LENGTH = 5000

# RETRIEVAL_K = 3

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.

Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

# retriever = vectorstore.as_retriever()

# Stage 1 - Bi-Encoder
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

# Stage 2 - Cross-Encoder
model_name = "BAAI/bge-reranker-base"
model_kwargs = {'device': 'cpu'}  
model = HuggingFaceCrossEncoder(model_name=model_name, model_kwargs=model_kwargs)
compressor = CrossEncoderReranker(model=model, top_n=3)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

llm = ChatOpenAI(temperature=0, model_name=MODEL)

def fetch_context(question: str) -> list:
    """
    Retrieve relevant context documents for a question.
    (This now uses the two-stage re-ranking retriever)
    """
    # return retriever.invoke(question, k=RETRIEVAL_K)
    return retriever.invoke(question)


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    docs = fetch_context(question)
    # Step 2: Build the context string, respecting the character limit
    context_pieces = []
    current_length = 0
    final_docs_used = [] # Keep track of docs actually used
    separator = "\n\n"
    separator_length = len(separator)

    print(f"BUILDING CONTEXT (Limit: {MAX_CONTEXT_LENGTH} chars)")
    for i, doc in enumerate(docs):
        doc_content = doc.page_content
        doc_length = len(doc_content)
        
        # Calculate length if this doc is added (plus separator if not the first doc)
        length_if_added = current_length + (separator_length if context_pieces else 0) + doc_length
        
        print(f"Considering Doc {i+1}/{len(docs)} (Length: {doc_length} chars). Current context length: {current_length}. Length if added: {length_if_added}")

        if length_if_added <= MAX_CONTEXT_LENGTH:
            context_pieces.append(doc_content)
            final_docs_used.append(doc) # Add doc to the list of used docs
            current_length = length_if_added
            print(f"Added Doc {i+1}. New context length: {current_length}")
        else:
            print(f"Skipping Doc {i+1} and subsequent docs (exceeds limit).")
            break # Stop adding documents if the limit is exceeded

    # Join the selected pieces to form the final context string
    context = separator.join(context_pieces)
    print(f"FINAL CONTEXT BUILT (Length: {len(context)} chars)")

    # context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, final_docs_used