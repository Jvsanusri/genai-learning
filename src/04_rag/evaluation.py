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
import json
import os

load_dotenv(dotenv_path=r"D:\genai-learning\.env")
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ✅ Build RAG system (same as before)
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
    v_docs = vector_retriever.invoke(query)
    b_docs = bm25_retriever.invoke(query)
    seen, combined = set(), []
    for doc in v_docs + b_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)
    return combined[:4]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_template("""
You are a SAP HCM DevOps expert.
Answer using ONLY the context below.
If not in context, say "Not found in runbook."

Context: {context}
Question: {question}
""")

rag_chain = (
    {
        "context": RunnableLambda(hybrid_retrieve) | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ════════════════════════════════════
# TEST SET — questions with known answers
# This is your ground truth!
# ════════════════════════════════════
TEST_SET = [
    {
        "question": "How do I fix a CrashLoopBackOff pod?",
        "expected_keywords": ["kubectl logs", "vault", "restart", "secret"],
        "expected_contains": "kubectl"
    },
    {
        "question": "What steps fix MySQL write traffic going to replica?",
        "expected_keywords": ["openstack", "weight", "loadbalancer", "terraform"],
        "expected_contains": "weight"
    },
    {
        "question": "How do I resolve ArgoCD OutOfSync issue?",
        "expected_keywords": ["argocd app sync", "crd", "diff"],
        "expected_contains": "sync"
    },
    {
        "question": "Where does Vector ship logs to?",
        "expected_keywords": ["nats", "hcm.logs", "splunk"],
        "expected_contains": "hcm.logs"
    },
    {
        "question": "How do I fix GTID replication error?",
        "expected_keywords": ["stop slave", "reset", "start slave", "gtid"],
        "expected_contains": "slave"
    },
    {
        "question": "What is the Dynatrace environment URL?",
        "expected_keywords": ["apm.btp", "c41fce75", "sci-prod"],
        "expected_contains": "apm.btp"
    },
    {
        "question": "How do I fix ImagePullBackOff?",
        "expected_keywords": ["kubectl describe", "registry", "imagepullsecret"],
        "expected_contains": "describe"
    },
    {
        "question": "What is the prevention for CrashLoopBackOff?",
        "expected_keywords": ["ttl", "renewal", "cron", "vault"],
        "expected_contains": "renewal"
    },
    {
        "question": "How do I check OpenStack load balancer member weights?",
        "expected_keywords": ["openstack loadbalancer member list", "pool-id"],
        "expected_contains": "openstack"
    },
    {
        "question": "What is the weather in Dallas?",
        "expected_keywords": ["not found"],
        "expected_contains": "not found"
    }
]

# ════════════════════════════════════
# EVALUATION METHOD 1: Reference-based
# Check if answer contains expected keywords
# ════════════════════════════════════
def evaluate_reference_based(question, answer, expected_keywords, expected_contains):
    answer_lower = answer.lower()
    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    keyword_score = keyword_hits / len(expected_keywords)
    contains_score = 1.0 if expected_contains.lower() in answer_lower else 0.0
    final_score = (keyword_score * 0.6) + (contains_score * 0.4)
    return {
        "keyword_score": round(keyword_score, 2),
        "contains_score": contains_score,
        "final_score": round(final_score, 2),
        "keyword_hits": f"{keyword_hits}/{len(expected_keywords)}"
    }

# ════════════════════════════════════
# EVALUATION METHOD 2: LLM-as-judge
# Use AI to score AI answers!
# ════════════════════════════════════
judge_prompt = ChatPromptTemplate.from_template("""
You are an expert SAP DevOps engineer evaluating AI answers.
Score the answer from 1-5 on these criteria:

Question: {question}
AI Answer: {answer}

Score EACH criterion from 1-5:
- Accuracy: Is the answer factually correct for SAP/Kubernetes?
- Completeness: Does it cover the main steps needed?
- Actionability: Can an engineer follow these steps immediately?

Reply in EXACTLY this JSON format:
{{"accuracy": <1-5>, "completeness": <1-5>, "actionability": <1-5>, "reason": "<one sentence>"}}
""")

def evaluate_llm_judge(question, answer):
    try:
        r = llm.invoke(judge_prompt.format_messages(
            question=question,
            answer=answer[:300]
        ))
        # Parse JSON from response
        content = r.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        scores = json.loads(content)
        avg = (scores["accuracy"] + scores["completeness"] + scores["actionability"]) / 3
        scores["average"] = round(avg, 2)
        return scores
    except Exception as e:
        return {"accuracy": 0, "completeness": 0, "actionability": 0, "average": 0, "reason": str(e)}

# ════════════════════════════════════
# RUN FULL EVALUATION
# ════════════════════════════════════
print("🧪 SAP HCM RAG SYSTEM — EVALUATION REPORT")
print("="*55)
print(f"Test set size: {len(TEST_SET)} questions")
print(f"Evaluation methods: Reference-based + LLM-as-judge")
print("="*55)

results = []
total_ref_score = 0
total_llm_score = 0

for i, test in enumerate(TEST_SET, 1):
    question = test["question"]
    print(f"\n[{i}/{len(TEST_SET)}] ❓ {question[:60]}")

    # Get AI answer
    answer = rag_chain.invoke(question)
    print(f"   ✅ Answer: {answer[:80]}...")

    # Method 1: Reference-based
    ref_eval = evaluate_reference_based(
        question, answer,
        test["expected_keywords"],
        test["expected_contains"]
    )
    print(f"   📊 Reference: {ref_eval['keyword_hits']} keywords | "
          f"Score: {ref_eval['final_score']:.0%}")

    # Method 2: LLM judge (only for first 5 to save API calls)
    llm_eval = {"average": 0, "reason": "skipped"}
    if i <= 5:
        llm_eval = evaluate_llm_judge(question, answer)
        print(f"   🤖 LLM Judge: {llm_eval['average']}/5 — {llm_eval.get('reason', '')[:60]}")

    results.append({
        "question": question,
        "answer": answer[:100],
        "ref_score": ref_eval["final_score"],
        "llm_score": llm_eval["average"],
        "keyword_hits": ref_eval["keyword_hits"]
    })

    total_ref_score += ref_eval["final_score"]
    total_llm_score += llm_eval["average"]

# ════════════════════════════════════
# FINAL REPORT
# ════════════════════════════════════
avg_ref = total_ref_score / len(TEST_SET)
avg_llm = total_llm_score / 5  # only 5 were judged

print(f"\n{'='*55}")
print(f"📊 EVALUATION SUMMARY")
print(f"{'='*55}")
print(f"Reference-based avg score : {avg_ref:.0%}")
print(f"LLM judge avg score (1-5) : {avg_llm:.1f}/5")
print(f"Production ready?          : {'✅ YES' if avg_ref >= 0.7 else '❌ NO — needs improvement'}")
print(f"\n📋 Per-question breakdown:")
for r in results:
    status = "✅" if r["ref_score"] >= 0.5 else "❌"
    print(f"  {status} [{r['ref_score']:.0%}] {r['question'][:50]}")

print(f"\n💡 Improvement areas:")
failed = [r for r in results if r["ref_score"] < 0.5]
if failed:
    for f in failed:
        print(f"  → '{f['question'][:50]}' — add more content to runbook")
else:
    print("  → All questions passed! Ready for production.")