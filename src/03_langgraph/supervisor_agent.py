from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r"D:\genai-learning\.env")
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ✅ STATE
class SupervisorState(TypedDict):
    incident: str
    agents_to_call: List[str]
    infra_findings: str
    db_findings: str
    monitoring_findings: str
    final_report: str

# ════════════════════════════════════
# SUPERVISOR — decides which agents
# ════════════════════════════════════
def supervisor_router(state: SupervisorState) -> SupervisorState:
    print("\n🔴 SUPERVISOR: Reading incident and planning...")
    r = llm.invoke([
        SystemMessage(content="""You are a senior SRE supervisor.
        Read the incident and decide which specialist agents to call.

        Available agents:
        - infra: for Kubernetes, pods, ArgoCD, deployments
        - database: for MySQL, Vault tokens, secrets, OpenStack LB
        - monitoring: for Dynatrace, Splunk, Vector logs, alerts

        Reply with ONLY a comma-separated list.
        Example: infra,database  or  database  or  infra,database,monitoring
        Choose only what is relevant!"""),
        HumanMessage(content=f"Incident: {state['incident']}")
    ])
    agents = [a.strip() for a in r.content.lower().split(",")]
    valid = [a for a in agents if a in ["infra", "database", "monitoring"]]
    state['agents_to_call'] = valid
    print(f"   → Will call agents: {valid}")
    return state

# ════════════════════════════════════
# ROUTER — decides next node
# ════════════════════════════════════
def route_next(state: SupervisorState) -> str:
    remaining = state['agents_to_call']
    if not remaining:
        return "create_report"
    next_agent = remaining[0]
    if next_agent == "infra":
        return "infra_agent"
    elif next_agent == "database":
        return "database_agent"
    elif next_agent == "monitoring":
        return "monitoring_agent"
    return "create_report"

# ════════════════════════════════════
# SPECIALIST AGENTS
# Each removes itself from queue first!
# ════════════════════════════════════
def infra_agent(state: SupervisorState) -> SupervisorState:
    # ✅ Remove self from queue to prevent infinite loop
    state['agents_to_call'] = [
        a for a in state['agents_to_call'] if a != 'infra'
    ]
    print("\n🔵 INFRA AGENT running...")
    r = llm.invoke([
        SystemMessage(content="""Expert Kubernetes/ArgoCD engineer.
        Analyze infra issues: pods, deployments, syncs.
        Give 2 bullet point findings only."""),
        HumanMessage(content=state['incident'])
    ])
    state['infra_findings'] = r.content
    print(f"   → Done: {r.content[:80]}...")
    return state

def database_agent(state: SupervisorState) -> SupervisorState:
    # ✅ Remove self from queue to prevent infinite loop
    state['agents_to_call'] = [
        a for a in state['agents_to_call'] if a != 'database'
    ]
    print("\n🟢 DATABASE AGENT running...")
    r = llm.invoke([
        SystemMessage(content="""Expert MySQL HA/Vault engineer.
        Analyze DB issues: replication, tokens, LB weights.
        Give 2 bullet point findings only."""),
        HumanMessage(content=state['incident'])
    ])
    state['db_findings'] = r.content
    print(f"   → Done: {r.content[:80]}...")
    return state

def monitoring_agent(state: SupervisorState) -> SupervisorState:
    # ✅ Remove self from queue to prevent infinite loop
    state['agents_to_call'] = [
        a for a in state['agents_to_call'] if a != 'monitoring'
    ]
    print("\n🟡 MONITORING AGENT running...")
    r = llm.invoke([
        SystemMessage(content="""Expert Dynatrace/Splunk engineer.
        Analyze monitoring: logs, alerts, metrics.
        Give 2 bullet point findings only."""),
        HumanMessage(content=state['incident'])
    ])
    state['monitoring_findings'] = r.content
    print(f"   → Done: {r.content[:80]}...")
    return state

def create_report(state: SupervisorState) -> SupervisorState:
    print("\n📊 SUPERVISOR creating final report...")
    findings = []
    if state['infra_findings']:
        findings.append(f"Infra findings:\n{state['infra_findings']}")
    if state['db_findings']:
        findings.append(f"Database findings:\n{state['db_findings']}")
    if state['monitoring_findings']:
        findings.append(f"Monitoring findings:\n{state['monitoring_findings']}")

    r = llm.invoke([
        SystemMessage(content="""Senior SRE supervisor.
        Create concise incident report with:
        - Root cause (1 sentence)
        - Fix steps (max 3 numbered steps)
        - Prevention (1 sentence)"""),
        HumanMessage(content=f"""
        Incident: {state['incident']}

        {chr(10).join(findings)}
        """)
    ])
    state['final_report'] = r.content
    return state

# ✅ BUILD GRAPH
workflow = StateGraph(SupervisorState)

workflow.add_node("supervisor", supervisor_router)
workflow.add_node("infra_agent", infra_agent)
workflow.add_node("database_agent", database_agent)
workflow.add_node("monitoring_agent", monitoring_agent)
workflow.add_node("create_report", create_report)

workflow.set_entry_point("supervisor")

# ✅ Conditional routing from supervisor
workflow.add_conditional_edges(
    "supervisor", route_next,
    {
        "infra_agent": "infra_agent",
        "database_agent": "database_agent",
        "monitoring_agent": "monitoring_agent",
        "create_report": "create_report"
    }
)

# ✅ After each agent → route to next or report
for agent_node in ["infra_agent", "database_agent", "monitoring_agent"]:
    workflow.add_conditional_edges(
        agent_node, route_next,
        {
            "infra_agent": "infra_agent",
            "database_agent": "database_agent",
            "monitoring_agent": "monitoring_agent",
            "create_report": "create_report"
        }
    )

workflow.add_edge("create_report", END)
app = workflow.compile()

# ✅ TEST — 3 incidents, smart routing
incidents = [
    {
        "name": "PURE DB ISSUE",
        "text": """MySQL replica in stage-mobile receiving write traffic.
        OpenStack LB misconfigured. Data inconsistency for 15 users."""
    },
    {
        "name": "PURE INFRA ISSUE",
        "text": """Pod bizx-car CrashLoopBackOff in bizx namespace.
        ArgoCD showing OutOfSync. Deployment failing in dev-prod."""
    },
    {
        "name": "FULL OUTAGE",
        "text": """Complete BizX outage. Vault token expired.
        MySQL broken. Dynatrace showing 500 alerts.
        500 SAP SuccessFactors engineers blocked."""
    }
]

for incident in incidents:
    print(f"\n{'='*55}")
    print(f"🚨 {incident['name']}")
    print(f"{'='*55}")

    result = app.invoke({
        "incident": incident['text'],
        "agents_to_call": [],
        "infra_findings": "",
        "db_findings": "",
        "monitoring_findings": "",
        "final_report": ""
    })

    print(f"\n📊 FINAL REPORT:")
    print(result['final_report'][:400])
    print(f"\n{'='*55}")
