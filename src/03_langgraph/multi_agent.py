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

# ✅ STATE — shared across all agents
class MultiAgentState(TypedDict):
    incident: str           # Original incident
    infra_findings: str     # From Infrastructure Agent
    db_findings: str        # From Database Agent
    monitoring_findings: str # From Monitoring Agent
    final_report: str       # From Supervisor Agent
    agents_called: List[str] # Track which agents ran

# ════════════════════════════════════════
# SPECIALIZED AGENTS
# Each expert in their own domain
# ════════════════════════════════════════

def infra_agent(state: MultiAgentState) -> MultiAgentState:
    """Infrastructure Agent — Kubernetes, ArgoCD, Pods"""
    print("\n🔵 INFRA AGENT analyzing...")
    r = llm.invoke([
        SystemMessage(content="""You are an expert Kubernetes 
        and ArgoCD engineer for SAP SCI environment.
        Analyze infrastructure aspects only: pods, deployments,
        ArgoCD sync, Helm releases, Gardener clusters.
        Be concise — 3 bullet points max."""),
        HumanMessage(content=f"Analyze this incident from infra perspective: {state['incident']}")
    ])
    state['infra_findings'] = r.content
    state['agents_called'].append("InfraAgent")
    print(f"   → {r.content[:100]}...")
    return state

def database_agent(state: MultiAgentState) -> MultiAgentState:
    """Database Agent — MySQL, Vault, Secrets"""
    print("\n🟢 DATABASE AGENT analyzing...")
    r = llm.invoke([
        SystemMessage(content="""You are an expert MySQL HA 
        and HashiCorp Vault engineer for SAP SCI environment.
        Analyze database aspects only: MySQL replication, 
        Vault tokens, secrets, OpenStack LB weights.
        Be concise — 3 bullet points max."""),
        HumanMessage(content=f"Analyze this incident from database perspective: {state['incident']}")
    ])
    state['db_findings'] = r.content
    state['agents_called'].append("DatabaseAgent")
    print(f"   → {r.content[:100]}...")
    return state

def monitoring_agent(state: MultiAgentState) -> MultiAgentState:
    """Monitoring Agent — Dynatrace, Splunk, Logs"""
    print("\n🟡 MONITORING AGENT analyzing...")
    r = llm.invoke([
        SystemMessage(content="""You are an expert in Dynatrace, 
        Splunk and Vector log monitoring for SAP SCI environment.
        Analyze monitoring aspects only: log patterns, 
        Dynatrace alerts, Splunk queries, Vector shipping.
        Be concise — 3 bullet points max."""),
        HumanMessage(content=f"Analyze this incident from monitoring perspective: {state['incident']}")
    ])
    state['monitoring_findings'] = r.content
    state['agents_called'].append("MonitoringAgent")
    print(f"   → {r.content[:100]}...")
    return state

def supervisor_agent(state: MultiAgentState) -> MultiAgentState:
    """Supervisor — combines all agent findings"""
    print("\n🔴 SUPERVISOR AGENT creating final report...")
    r = llm.invoke([
        SystemMessage(content="""You are the senior SRE supervisor
        for SAP SuccessFactors HCM on SCI environment.
        Synthesize findings from multiple specialist agents
        into ONE clear incident report with:
        1. Root cause
        2. Immediate action
        3. Prevention measure"""),
        HumanMessage(content=f"""
        Original Incident: {state['incident']}

        Infrastructure Agent findings:
        {state['infra_findings']}

        Database Agent findings:
        {state['db_findings']}

        Monitoring Agent findings:
        {state['monitoring_findings']}

        Create a concise final incident report.
        """)
    ])
    state['final_report'] = r.content
    state['agents_called'].append("SupervisorAgent")
    return state

# ✅ BUILD MULTI-AGENT GRAPH
workflow = StateGraph(MultiAgentState)

# Add all agents as nodes
workflow.add_node("infra_agent", infra_agent)
workflow.add_node("database_agent", database_agent)
workflow.add_node("monitoring_agent", monitoring_agent)
workflow.add_node("supervisor_agent", supervisor_agent)

# Flow: all 3 specialists run, then supervisor combines
workflow.set_entry_point("infra_agent")
workflow.add_edge("infra_agent", "database_agent")
workflow.add_edge("database_agent", "monitoring_agent")
workflow.add_edge("monitoring_agent", "supervisor_agent")
workflow.add_edge("supervisor_agent", END)

app = workflow.compile()

# ✅ TEST with real SAP incidents
incidents = [
    """SAP SuccessFactors BizX is completely down.
    Pod bizx-car in bizx namespace CrashLoopBackOff.
    Vault token expired 3 hours ago.
    MySQL replica receiving write traffic.
    500 engineers cannot login. Production environment.""",

    """ArgoCD shows hxm-platform OutOfSync in dev-prod shoot.
    Multiple microservices failing to deploy.
    Dynatrace showing pod restart alerts.
    Dev team blocked for 2 hours."""
]

for i, incident in enumerate(incidents, 1):
    print(f"\n{'='*55}")
    print(f"🚨 INCIDENT {i}: {incident[:60]}...")
    print(f"{'='*55}")

    result = app.invoke({
        "incident": incident,
        "infra_findings": "",
        "db_findings": "",
        "monitoring_findings": "",
        "final_report": "",
        "agents_called": []
    })

    print(f"\n{'='*55}")
    print(f"📊 FINAL INCIDENT REPORT:")
    print(f"{'='*55}")
    print(f"Agents called: {' → '.join(result['agents_called'])}")
    print(f"\n{result['final_report']}")