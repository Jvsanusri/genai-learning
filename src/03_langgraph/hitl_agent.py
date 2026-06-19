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
class HITLState(TypedDict):
    incident: str
    proposed_action: str
    human_approved: bool
    final_result: str

# ✅ NODE 1 — AI analyzes and proposes action
def analyze_and_propose(state: HITLState) -> HITLState:
    print("\n🤖 AI is analyzing the incident...")
    r = llm.invoke([HumanMessage(
        content=f"""You are a SAP DevOps expert.
        Analyze this incident and propose ONE specific 
        action to fix it. Be very concise — one sentence.
        Incident: {state['incident']}"""
    )])
    state['proposed_action'] = r.content.strip()
    print(f"\n🤖 AI proposes: {state['proposed_action']}")
    return state

# ✅ NODE 2 — Human reviews and approves/rejects
def human_approval(state: HITLState) -> HITLState:
    print("\n" + "="*55)
    print("⏸️  PAUSED — WAITING FOR HUMAN APPROVAL")
    print("="*55)
    print(f"📌 Incident : {state['incident'][:80]}...")
    print(f"🤖 Proposed : {state['proposed_action']}")
    print("="*55)

    while True:
        answer = input("\n✋ Do you approve this action? (yes/no): ")
        if answer.lower() in ['yes', 'no', 'y', 'n']:
            break
        print("Please type yes or no")

    state['human_approved'] = answer.lower() in ['yes', 'y']
    return state

# ✅ NODE 3a — Execute if approved
def execute_action(state: HITLState) -> HITLState:
    print("\n✅ APPROVED — Executing action...")
    r = llm.invoke([HumanMessage(
        content=f"""The engineer approved this fix: {state['proposed_action']}
        For incident: {state['incident']}
        Write a 2-sentence execution confirmation 
        as if you just ran the fix."""
    )])
    state['final_result'] = (
        f"✅ ACTION EXECUTED\n"
        f"   Fix: {state['proposed_action']}\n"
        f"   Status: {r.content[:150]}"
    )
    return state

# ✅ NODE 3b — Escalate if rejected
def escalate_action(state: HITLState) -> HITLState:
    print("\n❌ REJECTED — Escalating to senior engineer...")
    state['final_result'] = (
        f"❌ ACTION REJECTED BY ENGINEER\n"
        f"   Proposed fix: {state['proposed_action']}\n"
        f"   Status: Escalated to Prabhakar (Senior SRE)\n"
        f"   JIRA P1 ticket created for manual review"
    )
    return state

# ✅ ROUTER — based on human decision
def route_approval(state: HITLState) -> str:
    if state['human_approved']:
        return 'execute'
    return 'escalate'

# ✅ BUILD GRAPH
workflow = StateGraph(HITLState)

workflow.add_node("analyze", analyze_and_propose)
workflow.add_node("human_approval", human_approval)
workflow.add_node("execute", execute_action)
workflow.add_node("escalate", escalate_action)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "human_approval")

# ✅ Conditional routing based on human decision
workflow.add_conditional_edges(
    "human_approval",
    route_approval,
    {
        "execute": "execute",
        "escalate": "escalate"
    }
)
workflow.add_edge("execute", END)
workflow.add_edge("escalate", END)

app = workflow.compile()

# ✅ TEST
print("🚨 SAP HCM HITL INCIDENT RESPONSE SYSTEM")
print("="*55)

result = app.invoke({
    "incident": """Production pod bizx-car in bizx namespace 
    is CrashLoopBackOff. Vault token expired 2 hours ago. 
    500 SAP SuccessFactors engineers cannot login.""",
    "proposed_action": "",
    "human_approved": False,
    "final_result": ""
})

print(f"\n{'='*55}")
print(f"📊 FINAL RESULT:")
print(f"{'='*55}")
print(result['final_result'])