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

# ✅ STATE — data that flows through every node
# Think of it as a clipboard passed between team members
class IncidentState(TypedDict):
    incident: str        # Original incident text
    severity: str        # filled by Node 1
    runbook_advice: str  # filled by Node 2
    action_taken: str    # filled by Node 3
    resolved: bool       # filled by Node 3

# ✅ NODE 1 — Classify the incident
def classify_incident(state: IncidentState) -> IncidentState:
    print("\n🔍 NODE 1: Classifying incident...")
    response = llm.invoke([HumanMessage(
        content=f"""Classify this SAP incident severity as 
        ONLY one word: high, medium, or low.
        Incident: {state['incident']}"""
    )])
    state['severity'] = response.content.strip().lower().replace(".", "").replace(",", "")
    print(f"   → Severity: {state['severity']}")
    return state

# ✅ NODE 2 — Look up runbook advice
def lookup_runbook(state: IncidentState) -> IncidentState:
    print("\n📚 NODE 2: Looking up runbook...")
    response = llm.invoke([HumanMessage(
        content=f"""You are a SAP DevOps expert.
        Give a 2-sentence fix for this incident:
        {state['incident']}
        Severity: {state['severity']}"""
    )])
    state['runbook_advice'] = response.content
    print(f"   → Advice: {state['runbook_advice'][:80]}...")
    return state

# ✅ NODE 3 — Take action based on severity
def take_action(state: IncidentState) -> IncidentState:
    print("\n⚙️  NODE 3: Taking action...")
    if state['severity'] == 'high':
        state['action_taken'] = (
            f"🚨 ESCALATED to on-call engineer via PagerDuty.\n"
            f"   Runbook advice shared: {state['runbook_advice'][:100]}"
        )
    elif state['severity'] == 'medium':
        state['action_taken'] = (
            f"🔧 AUTO-REMEDIATION triggered.\n"
            f"   Steps applied: {state['runbook_advice'][:100]}"
        )
    else:
        state['action_taken'] = (
            f"📝 LOGGED to monitoring dashboard.\n"
            f"   Note: {state['runbook_advice'][:100]}"
        )
    state['resolved'] = True
    print(f"   → Action: {state['action_taken'][:80]}...")
    return state

# ✅ BUILD THE GRAPH
workflow = StateGraph(IncidentState)

# Add nodes
workflow.add_node("classify", classify_incident)
workflow.add_node("lookup_runbook", lookup_runbook)
workflow.add_node("take_action", take_action)

# Add edges — defines the flow
workflow.set_entry_point("classify")           # Start here
workflow.add_edge("classify", "lookup_runbook") # Then here
workflow.add_edge("lookup_runbook", "take_action") # Then here
workflow.add_edge("take_action", END)           # Then finish

# Compile the graph
app = workflow.compile()

# ✅ TEST with 2 real SAP incidents
incidents = [
    """Pod bizx-car in bizx namespace CrashLoopBackOff.
    Vault token expired. 500 engineers cannot login to 
    SAP SuccessFactors HCM. Production environment.""",

    """ArgoCD app hxm-platform OutOfSync in dev-prod shoot.
    Missing CRD causing deployment failure.
    Dev environment only, no production impact."""
]

for incident in incidents:
    print(f"\n{'='*55}")
    print(f"🚨 NEW INCIDENT: {incident[:60]}...")
    print(f"{'='*55}")

    result = app.invoke({
        "incident": incident,
        "severity": "",
        "runbook_advice": "",
        "action_taken": "",
        "resolved": False
    })

    print(f"\n📊 FINAL RESULT:")
    print(f"Severity:  {result['severity']}")
    print(f"Action:    {result['action_taken']}")
    print(f"Resolved:  {result['resolved']}")