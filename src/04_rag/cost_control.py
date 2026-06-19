from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import time
import hashlib
import json
from pathlib import Path

load_dotenv(dotenv_path=r"D:\genai-learning\.env")

# ════════════════════════════════════
# COST ENGINEERING STRATEGY 1:
# MODEL ROUTING
# Cheap model for simple, expensive for complex
# ════════════════════════════════════

# Fast cheap model — for simple questions
llm_fast = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",   # Small = cheap + fast
    max_tokens=200
)

# Powerful model — for complex analysis
llm_powerful = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile", # Large = accurate
    max_tokens=500
)

def route_model(query: str):
    """Route to cheap or powerful model based on complexity"""
    simple_keywords = [
        "what is", "where is", "who is", "show me",
        "list", "url", "name", "contact", "when"
    ]
    complex_keywords = [
        "how do i fix", "troubleshoot", "debug", "analyze",
        "why is", "investigate", "diagnose", "resolve"
    ]

    query_lower = query.lower()

    if any(kw in query_lower for kw in complex_keywords):
        print(f"   🔴 Routing to POWERFUL model (complex query)")
        return llm_powerful, "powerful"
    elif any(kw in query_lower for kw in simple_keywords):
        print(f"   🟢 Routing to FAST model (simple query)")
        return llm_fast, "fast"
    else:
        print(f"   🟡 Routing to FAST model (default)")
        return llm_fast, "fast"

# ════════════════════════════════════
# COST ENGINEERING STRATEGY 2:
# RESPONSE CACHING
# Same question = don't call LLM again!
# ════════════════════════════════════

CACHE_FILE = Path(r"D:\genai-learning\src\04_rag\query_cache.json")

def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}

def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))

def get_cache_key(query: str) -> str:
    """Create a hash key for the query"""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()

def cached_query(query: str, system_prompt: str) -> dict:
    """
    Query with caching — saves LLM calls for repeated questions!
    Returns: {answer, source, tokens_saved}
    """
    cache = load_cache()
    cache_key = get_cache_key(query)

    # ✅ Cache HIT — return cached answer
    if cache_key in cache:
        cached = cache[cache_key]
        print(f"   💚 CACHE HIT! Saved ~{cached['token_estimate']} tokens")
        return {
            "answer": cached["answer"],
            "source": "cache",
            "tokens_saved": cached["token_estimate"],
            "model_used": "none (cached)"
        }

    # ❌ Cache MISS — call LLM
    print(f"   🔴 CACHE MISS — calling LLM...")
    llm, model_type = route_model(query)

    start = time.time()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    response = llm.invoke(messages)
    elapsed = time.time() - start

    answer = response.content
    token_estimate = len(query.split()) + len(answer.split())

    # Save to cache
    cache[cache_key] = {
        "query": query,
        "answer": answer,
        "token_estimate": token_estimate,
        "model_used": model_type,
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_cache(cache)

    return {
        "answer": answer,
        "source": "llm",
        "tokens_saved": 0,
        "model_used": model_type,
        "latency": round(elapsed, 2)
    }

# ════════════════════════════════════
# COST ENGINEERING STRATEGY 3:
# TOKEN BUDGETING
# Limit tokens to control costs
# ════════════════════════════════════

def count_tokens_estimate(text: str) -> int:
    """Rough token estimate: ~0.75 words per token"""
    return int(len(text.split()) / 0.75)

def budget_aware_query(query: str, max_tokens: int = 300) -> str:
    """Query with token budget enforcement"""
    estimated_input = count_tokens_estimate(query)
    print(f"   📊 Estimated input tokens: ~{estimated_input}")

    if estimated_input > max_tokens:
        # Truncate query to fit budget
        words = query.split()
        truncated = " ".join(words[:int(max_tokens * 0.75)])
        print(f"   ✂️  Query truncated to fit budget")
        query = truncated

    return query

# ════════════════════════════════════
# DEMONSTRATION
# ════════════════════════════════════

SYSTEM_PROMPT = """You are a SAP HCM DevOps assistant.
Answer concisely in 2-3 sentences max."""

print("💰 SAP HCM AI — COST ENGINEERING DEMO")
print("="*55)

# Test queries — mix of simple and complex, with repeats
test_queries = [
    "What is the ArgoCD UI URL for SCI dev?",        # simple
    "How do I fix a CrashLoopBackOff pod?",           # complex
    "Who is the on-call engineer?",                    # simple
    "How do I troubleshoot MySQL replica lag?",        # complex
    "What is the ArgoCD UI URL for SCI dev?",         # REPEAT → cache hit!
    "Who is the on-call engineer?",                    # REPEAT → cache hit!
    "How do I fix a CrashLoopBackOff pod?",           # REPEAT → cache hit!
]

total_tokens_used = 0
total_tokens_saved = 0
llm_calls = 0
cache_hits = 0

print("\n🔀 STRATEGY 1: Model Routing")
print("-"*55)
for query in test_queries[:4]:
    print(f"\n❓ {query}")
    llm, model = route_model(query)
    print(f"   → Using: {model} model")

print("\n💾 STRATEGY 2: Response Caching")
print("-"*55)
for i, query in enumerate(test_queries, 1):
    print(f"\n[{i}] ❓ {query[:50]}")
    result = cached_query(query, SYSTEM_PROMPT)
    print(f"   ✅ {result['answer'][:80]}...")
    print(f"   📊 Source: {result['source']} | "
          f"Model: {result['model_used']}")

    if result['source'] == 'cache':
        cache_hits += 1
        total_tokens_saved += result['tokens_saved']
    else:
        llm_calls += 1
        total_tokens_used += result.get('tokens_saved', 100)

print("\n📊 STRATEGY 3: Token Budgeting")
print("-"*55)
long_query = """Please provide me with an extremely detailed 
explanation of everything related to how I should troubleshoot 
the SAP HCM BizX application when it experiences issues with 
the Kubernetes pod named bizx-car in the bizx namespace on the 
SAP Sovereign Cloud infrastructure managed environment."""
budgeted = budget_aware_query(long_query, max_tokens=50)
print(f"   Original length: {len(long_query.split())} words")
print(f"   Budgeted length: {len(budgeted.split())} words")

print(f"\n{'='*55}")
print(f"💰 COST ENGINEERING SUMMARY")
print(f"{'='*55}")
print(f"Total queries     : {len(test_queries)}")
print(f"LLM calls made    : {llm_calls}")
print(f"Cache hits        : {cache_hits}")
print(f"Tokens saved      : ~{total_tokens_saved}")
cache_rate = (cache_hits/len(test_queries))*100
print(f"Cache hit rate    : {cache_rate:.0f}%")
print(f"Cost reduction    : ~{cache_rate:.0f}% on repeated queries")
print(f"\n✅ Run again to see MORE cache hits!")
print(f"   Cache saved at: {CACHE_FILE}")