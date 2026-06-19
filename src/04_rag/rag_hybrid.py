from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r"D:\genai-learning\.env")
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ✅ STEP 1: Load and chunk
print("📚 Loading SAP runbook...")
loader = TextLoader(r"D:\genai-learning\src\04_rag\sap_runbook.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["## ", "### ", "\n\n", "\n", " "]
)
chunks = splitter.split_documents(docs)
print(f"✅ Split into {len(chunks)} chunks")

# ✅ STEP 2: Vector retriever
print("\n🔢 Building vector retriever...")
embeddings = FakeEmbeddings(size=384)
vectorstore = Chroma.from_documents(chunks, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Vector retriever ready")

# ✅ STEP 3: BM25 keyword retriever
print("\n🔍 Building BM25 keyword retriever...")
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3
print("✅ BM25 retriever ready")

# ✅ STEP 4: Manual Hybrid — no EnsembleRetriever needed!
def hybrid_retrieve(query):
    """Combine vector + BM25 results — best of both worlds"""
    vector_docs = vector_retriever.invoke(query)
    bm25_docs   = bm25_retriever.invoke(query)
    # Deduplicate by content
    seen, combined = set(), []
    for doc in vector_docs + bm25_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)
    return combined[:4]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("\n🔀 Hybrid retriever ready (Vector + BM25 combined)!")

# ✅ STEP 5: Build chains
prompt = ChatPromptTemplate.from_template("""
You are a SAP HCM DevOps expert for SCI environment.
Answer using ONLY the context below.
If not in context, say "Not found in runbook."

Context:
{context}

Question: {question}
""")

# Hybrid chain
hybrid_chain = (
    {
        "context": RunnableLambda(hybrid_retrieve) | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Vector only chain
vector_chain = (
    {
        "context": RunnableLambda(lambda q: format_docs(vector_retriever.invoke(q))),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ✅ STEP 6: Compare both
print("\n" + "="*55)
print("🧪 COMPARING: Vector Only vs Hybrid Search")
print("="*55)

questions = [
    "What is the fix for GTID replication error?",
    "How do I check openstack loadbalancer member weights?",
    "What is the NATS TLS subject for log shipping?",
    "How do I fix ImagePullBackOff?"
]

for q in questions:
    print(f"\n❓ {q}")
    print(f"\n🔵 Vector only:")
    print(f"   {vector_chain.invoke(q)[:150]}")
    print(f"\n🟢 Hybrid (Vector + BM25):")
    print(f"   {hybrid_chain.invoke(q)[:150]}")
    print("-"*55)