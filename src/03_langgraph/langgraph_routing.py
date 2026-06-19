from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r"D:\genai-learning\.env")
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ✅ STATE
class IncidentState(TypedDict):
    incident: str
    severity: str
    result: str

# ✅ NODE 1 — Classify
def classify(state: IncidentState) -> IncidentState:
    print("\n🔍 Classifying severity...")
    r = llm.invoke([HumanMessage(
        content=f"Reply ONLY with one word — high, medium, or low: {state['incident']}"
    )])
    state['severity'] = r.content.strip().lower().replace(".", "")
    print(f"   → Severity: {state['severity']}")
    return state

# ✅ ROUTER FUNCTION — this decides which node runs next
def route_by_severity(state: IncidentState) -> str:
    severity = state['severity']
    if 'high' in severity:
        return 'escalate'     # → go to escalate node
    elif 'medium' in severity:
        return 'auto_fix'     # → go to auto_fix node
    else:
        return 'just_log'     # → go to just_log node

# ✅ NODE 2a — High severity path
def escalate(state: IncidentState) -> IncidentState:
    print("🚨 HIGH → Paging on-call engineer!")
    r = llm.invoke([HumanMessage(
        content=f"Give immediate 1-sentence fix for: {state['incident']}"
    )])
    state['result'] = (
        f"ESCALATED via PagerDuty | "
        f"JIRA P1 ticket created | "
        f"Fix: {r.content[:100]}"
    )
    return state

# ✅ NODE 2b — Medium severity path
def auto_fix(state: IncidentState) -> IncidentState:
    print("🔧 MEDIUM → Running auto-remediation...")
    r = llm.invoke([HumanMessage(
        content=f"Give a 1-sentence automated fix script for: {state['incident']}"
    )])
    state['result'] = (
        f"AUTO-REMEDIATION applied | "
        f"JIRA P2 ticket created | "
        f"Script: {r.content[:100]}"
    )
    return state

# ✅ NODE 2c — Low severity path
def just_log(state: IncidentState) -> IncidentState:
    print("📝 LOW → Logging to dashboard...")
    state['result'] = (
        f"LOGGED to Splunk dashboard | "
        f"JIRA P3 ticket created | "
        f"Monitor for 24hrs"
    )
    return state

# ✅ BUILD THE GRAPH
workflow = StateGraph(IncidentState)

# Add all nodes
workflow.add_node("classify", classify)
workflow.add_node("escalate", escalate)
workflow.add_node("auto_fix", auto_fix)
workflow.add_node("just_log", just_log)

# Entry point
workflow.set_entry_point("classify")

# ✅ CONDITIONAL EDGE — the magic!
workflow.add_conditional_edges(
    "classify",          # From this node...
    route_by_severity,   # ...call this router function...
    {                    # ...map result to node name
        "escalate": "escalate",
        "auto_fix": "auto_fix",
        "just_log": "just_log"
    }
)

# All paths end here
workflow.add_edge("escalate", END)
workflow.add_edge("auto_fix", END)
workflow.add_edge("just_log", END)

app = workflow.compile()

# ✅ TEST — 3 different incidents, 3 different paths
incidents = [
    {
        "name": "PROD - Vault token expired",
        "text": """Production pod bizx-car CrashLoopBackOff.
        Vault token expired. 500 engineers blocked.
        Revenue impact. Immediate fix needed."""
    },
    {
        "name": "DEV - ArgoCD OutOfSync",
        "text": """ArgoCD app hxm-platform OutOfSync in dev-prod.
        Missing CRD blocking deployment.
        Dev team cannot deploy new features."""
    },
    {
        "name": "STAGE - Minor CSS issue",
        "text": """Minor CSS alignment issue on BizX login page.
        Affects only visual appearance.
        No functional impact, no data loss."""
    }
]

for incident in incidents:
    print(f"\n{'='*55}")
    print(f"📌 {incident['name']}")
    print(f"{'='*55}")

    result = app.invoke({
        "incident": incident['text'],
        "severity": "",
        "result": ""
    })

    print(f"\n📊 OUTCOME:")
    print(f"   Severity : {result['severity']}")
    print(f"   Action   : {result['result']}")