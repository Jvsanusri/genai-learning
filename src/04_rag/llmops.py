from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import time

load_dotenv(dotenv_path=r"D:\genai-learning\.env")

print("🔍 LLMOps Configuration:")
print(f"   Tracing  : {os.getenv('LANGCHAIN_TRACING_V2', 'false')}")
print(f"   Project  : {os.getenv('LANGCHAIN_PROJECT', 'default')}")
print(f"   API Key  : {'✅ Set' if os.getenv('LANGCHAIN_API_KEY') else '❌ Missing'}")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

loader = TextLoader(r"D:\genai-learning\src\04_rag\sap_runbook.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,
    separators=["## ", "### ", "\n\n", "\n", " "]
)
chunks = splitter.split_documents(docs)
embeddings = FakeEmbeddings(size=384)
vectorstore = Chroma.from_documents(chunks, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

def hybrid_retrieve(query):
    v = vector_retriever.invoke(query)
    b = bm25_retriever.invoke(query)
    seen, combined = set(), []
    for doc in v + b:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)
    return combined[:4]

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

prompt = ChatPromptTemplate.from_template("""
You are a SAP HCM DevOps expert for SCI environment.
Answer using ONLY the context. If not found say "Not in runbook."

Context: {context}
Question: {question}
""")

rag_chain = (
    {
        "context": RunnableLambda(hybrid_retrieve) | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

metrics = {"total_calls": 0, "total_latency": 0, "errors": 0, "by_query": []}

def tracked_query(query: str, run_name: str = None) -> dict:
    metrics["total_calls"] += 1
    start = time.time()
    try:
        answer = rag_chain.invoke(
            query,
            config={
                "run_name": run_name or f"query_{metrics['total_calls']}",
                "tags": ["sap-hcm", "rag", "production"]
            }
        )
        latency = round(time.time() - start, 2)
        metrics["total_latency"] += latency
        input_tokens = len(query.split()) + 200
        output_tokens = len(answer.split())
        result = {
            "query": query[:50],
            "answer": answer[:100],
            "latency_s": latency,
            "input_tokens_est": input_tokens,
            "output_tokens_est": output_tokens,
            "status": "success"
        }
        metrics["by_query"].append(result)
        return result
    except Exception as e:
        metrics["errors"] += 1
        return {"query": query[:50], "answer": str(e),
                "latency_s": 0, "status": "error"}

print("\n" + "="*55)
print("🔍 RUNNING MONITORED QUERIES")
print("="*55)
print("All calls traced to LangSmith!\n")

queries = [
    ("How do I fix a CrashLoopBackOff pod?",     "crashloop-fix"),
    ("What steps fix MySQL write to replica?",    "mysql-fix"),
    ("Where does Vector ship logs?",              "vector-logs"),
    ("How do I resolve ArgoCD OutOfSync?",        "argocd-fix"),
    ("What is the weather in Dallas?",            "off-topic-test"),
]

for query, run_name in queries:
    print(f"❓ {query}")
    result = tracked_query(query, run_name)
    print(f"   ✅ {result['answer'][:70]}...")
    print(f"   ⏱️  Latency : {result['latency_s']}s")
    print(f"   Status   : {result['status']}\n")

total = metrics["total_calls"]
avg_latency = metrics["total_latency"] / total if total > 0 else 0
total_tokens = sum(
    r.get("input_tokens_est", 0) + r.get("output_tokens_est", 0)
    for r in metrics["by_query"]
)
cost = total_tokens * 0.00000059

print("="*55)
print("📊 LLMOPS METRICS DASHBOARD")
print("="*55)
print(f"Total queries   : {total}")
print(f"Errors          : {metrics['errors']}")
print(f"Avg latency     : {avg_latency:.2f}s")
print(f"Total tokens    : ~{total_tokens}")
print(f"Estimated cost  : ${cost:.6f}")
print(f"Error rate      : {metrics['errors']/total*100:.0f}%")

print(f"\n⏱️  LATENCY PER QUERY:")
for r in metrics["by_query"]:
    bar = "█" * int(r["latency_s"] * 3)
    status = "✅" if r["status"] == "success" else "❌"
    print(f"  {status} {r['query'][:40]:<40} {r['latency_s']}s {bar}")

if os.getenv("LANGCHAIN_API_KEY"):
    print(f"\n🔗 View traces at: https://smith.langchain.com")
    print(f"   Project: {os.getenv('LANGCHAIN_PROJECT')}")