from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
import json
from pathlib import Path

load_dotenv(dotenv_path=r"D:\genai-learning\.env")
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ════════════════════════════════════════
# PART 1: SHORT-TERM MEMORY (in-session)
# ════════════════════════════════════════
print("\n" + "="*55)
print("🧠 PART 1: SHORT-TERM MEMORY (lost on exit)")
print("="*55)

# Memory lives only in this list
short_term = [
    SystemMessage(content="You are a SAP HCM DevOps assistant.")
]

def chat_short(message):
    short_term.append(HumanMessage(content=message))
    r = llm.invoke(short_term)
    short_term.append(AIMessage(content=r.content))
    return r.content

# Test memory across messages
print("\n💬 Message 1:")
print(chat_short("My name is Anu. I manage SAP SuccessFactors on SCI Prod."))

print("\n💬 Message 2 (does it remember?):")
print(chat_short("What is my name and what do I manage?"))

print("\n💬 Message 3 (deeper memory):")
print(chat_short("We have 60 microservices and use ArgoCD for deployment."))

print("\n💬 Message 4 (remembers everything?):")
print(chat_short("Summarize what you know about my work in one sentence."))

print(f"\n📊 Messages in short-term memory: {len(short_term)}")
print("⚠️  This memory DISAPPEARS when program ends!\n")

# ════════════════════════════════════════
# PART 2: LONG-TERM MEMORY (filesystem)
# ════════════════════════════════════════
print("="*55)
print("💾 PART 2: LONG-TERM MEMORY (saved to file!)")
print("="*55)

# Save memory to a JSON file
MEMORY_FILE = Path(r"D:\genai-learning\src\02_langchain\long_term_memory.json")

def load_memory():
    """Load memory from file"""
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    # Default structure if file doesn't exist
    return {
        "engineer": {},
        "team_preferences": [],
        "incident_history": [],
        "known_facts": []
    }

def save_memory(memory):
    """Save memory to file"""
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))
    print(f"   💾 Saved to: {MEMORY_FILE}")

def remember(category, fact):
    """Add a fact to long-term memory"""
    memory = load_memory()
    if category == "fact":
        if fact not in memory["known_facts"]:
            memory["known_facts"].append(fact)
    elif category == "preference":
        if fact not in memory["team_preferences"]:
            memory["team_preferences"].append(fact)
    elif category == "incident":
        memory["incident_history"].append(fact)
    save_memory(memory)
    print(f"   ✅ Remembered: {fact}")

def recall_all():
    """Get all memories as context string"""
    memory = load_memory()
    context_parts = []
    if memory["known_facts"]:
        context_parts.append("Known facts: " + "; ".join(memory["known_facts"]))
    if memory["team_preferences"]:
        context_parts.append("Team preferences: " + "; ".join(memory["team_preferences"]))
    if memory["incident_history"]:
        recent = memory["incident_history"][-3:]  # Last 3 incidents
        context_parts.append("Recent incidents: " + "; ".join(recent))
    return " | ".join(context_parts)

def chat_with_long_term(message):
    """Chat using long-term memory as context"""
    context = recall_all()
    messages = [
        SystemMessage(content=f"""You are a SAP HCM DevOps assistant.
        Use this context about the engineer: {context}"""),
        HumanMessage(content=message)
    ]
    r = llm.invoke(messages)
    return r.content

# ✅ Save important facts to long-term memory
print("\n📝 Saving facts to long-term memory...")
remember("fact", "Engineer name: Anuradha, works at Aarna Technologies on SAP SCI")
remember("fact", "Managing SAP SuccessFactors HCM with 60+ microservices on Gardener K8s")
remember("fact", "Production environments: SCI Dev and SCI Prod both active")
remember("preference", "Team prefers JIRA for incident tracking over email")
remember("preference", "On-call engineer is Prabhakar Aluri")
remember("preference", "Slack channel for alerts: #sci-devops-alerts")
remember("incident", "2026-06-19: Pod bizx-car CrashLoopBackOff - Vault token expired")
remember("incident", "2026-06-18: MySQL replica receiving write traffic - LB misconfiguration")

# ✅ Now chat using long-term memory
print("\n💬 Chatting with LONG-TERM memory...")
print("\nQ: What do you know about my team?")
print(chat_with_long_term("What do you know about my team and environment?"))

print("\nQ: Who should I contact for incidents?")
print(chat_with_long_term("If I have a production incident, who should I contact?"))

print("\nQ: What incidents happened recently?")
print(chat_with_long_term("What recent incidents have occurred in my environment?"))

# Show what's saved
memory = load_memory()
print(f"\n📊 LONG-TERM MEMORY CONTENTS:")
print(f"   Known facts    : {len(memory['known_facts'])}")
print(f"   Preferences    : {len(memory['team_preferences'])}")
print(f"   Incident history: {len(memory['incident_history'])}")
print(f"\n✅ This memory PERSISTS — even after program restarts!")
print(f"   Saved at: {MEMORY_FILE}")