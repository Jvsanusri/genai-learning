from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r"D:\genai-learning\.env")
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

SKILLS_DIR = Path(r"D:\genai-learning\src\skills")

# ✅ Skill loader — reads SKILL.md files at runtime
def load_skill(skill_name: str) -> str:
    """Load a skill from markdown file"""
    skill_file = SKILLS_DIR / f"{skill_name}.md"
    if skill_file.exists():
        content = skill_file.read_text()
        print(f"   📖 Loaded skill: {skill_name}.md")
        return content
    return f"Skill {skill_name} not found"

# ✅ STATE
class DeepAgentState(TypedDict):
    incident: str
    triage_result: str
    runbook_result: str
    jira_ticket: str

# ✅ NODE 1 — Triage Agent uses incident_triage skill
def triage_agent(state: DeepAgentState) -> DeepAgentState:
    print("\n🔍 TRIAGE AGENT loading skill...")
    skill = load_skill("incident_triage")
    r = llm.invoke([
        SystemMessage(content=f"""You are a SAP SCI DevOps agent.
        Follow this skill EXACTLY:

        {skill}

        Do not deviate from the output format."""),
        HumanMessage(content=f"Incident: {state['incident']}")
    ])
    state['triage_result'] = r.content
    print(f"   ✅ Triage complete")
    print(f"   {r.content[:150]}...")
    return state

# ✅ NODE 2 — Runbook Agent uses runbook_lookup skill
def runbook_agent(state: DeepAgentState) -> DeepAgentState:
    print("\n📚 RUNBOOK AGENT loading skill...")
    skill = load_skill("runbook_lookup")
    r = llm.invoke([
        SystemMessage(content=f"""You are a SAP SCI runbook expert.
        Follow this skill EXACTLY:

        {skill}

        Use the triage results to find the right runbook."""),
        HumanMessage(content=f"""
        Original incident: {state['incident']}
        Triage result: {state['triage_result']}

        Find the fix procedure.""")
    ])
    state['runbook_result'] = r.content
    print(f"   ✅ Runbook lookup complete")
    print(f"   {r.content[:150]}...")
    return state

# ✅ NODE 3 — JIRA Agent uses jira_ticket skill
def jira_agent(state: DeepAgentState) -> DeepAgentState:
    print("\n🎫 JIRA AGENT loading skill...")
    skill = load_skill("jira_ticket")
    r = llm.invoke([
        SystemMessage(content=f"""You are a SAP SCI JIRA specialist.
        Follow this skill EXACTLY:

        {skill}

        Create a complete JIRA ticket."""),
        HumanMessage(content=f"""
        Incident: {state['incident']}
        Triage: {state['triage_result']}
        Fix steps: {state['runbook_result']}

        Create the JIRA ticket.""")
    ])
    state['jira_ticket'] = r.content
    print(f"   ✅ JIRA ticket created")
    return state

# ✅ BUILD GRAPH
workflow = StateGraph(DeepAgentState)
workflow.add_node("triage", triage_agent)
workflow.add_node("runbook", runbook_agent)
workflow.add_node("jira", jira_agent)

workflow.set_entry_point("triage")
workflow.add_edge("triage", "runbook")
workflow.add_edge("runbook", "jira")
workflow.add_edge("jira", END)

app = workflow.compile()

# ✅ TEST
print("🤖 DEEP AGENT WITH SKILLS — SAP HCM INCIDENT SYSTEM")
print("="*55)

incident = """
Production pod bizx-car in bizx namespace CrashLoopBackOff.
Vault token expired 2 hours ago.
500 SAP SuccessFactors engineers cannot login.
SCI Prod environment. Revenue impact ongoing.
"""

result = app.invoke({
    "incident": incident,
    "triage_result": "",
    "runbook_result": "",
    "jira_ticket": ""
})

print(f"\n{'='*55}")
print("📋 TRIAGE RESULT:")
print("="*55)
print(result['triage_result'])

print(f"\n{'='*55}")
print("📚 RUNBOOK FIX:")
print("="*55)
print(result['runbook_result'])

print(f"\n{'='*55}")
print("🎫 JIRA TICKET:")
print("="*55)
print(result['jira_ticket'])