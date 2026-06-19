from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from datetime import datetime
import os
import time

load_dotenv(dotenv_path=r"D:\genai-learning\.env")
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ════════════════════════════════════
# MIDDLEWARE LAYER
# Wraps every tool call with:
# 1. Validation
# 2. Logging
# 3. Rate limiting
# 4. Safety checks
# ════════════════════════════════════

# Tool call log — tracks all executions
tool_call_log = []

# Rate limiter — tracks calls per tool
call_counts = {}
RATE_LIMIT = 5  # max calls per tool per session

# Production namespaces — require extra caution
PROD_NAMESPACES = ["bizx", "mobile", "platform-cockpit",
                   "encryption-kms", "platformfoundations"]

def middleware(tool_name: str, params: dict,
               tool_func, *args) -> str:
    """
    Middleware wrapper for all tool calls.
    Validates, logs, rate-limits before executing.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")

    # ✅ MIDDLEWARE 1: Rate limiting
    call_counts[tool_name] = call_counts.get(tool_name, 0) + 1
    if call_counts[tool_name] > RATE_LIMIT:
        msg = f"🚫 RATE LIMIT: {tool_name} called too many times ({call_counts[tool_name]})"
        print(f"   ⚠️  {msg}")
        return msg

    # ✅ MIDDLEWARE 2: Safety check for production namespaces
    namespace = params.get("namespace", "")
    if namespace in PROD_NAMESPACES:
        print(f"   ⚠️  PROD namespace detected: {namespace}")
        answer = input(f"   🔐 Confirm action on PROD namespace '{namespace}'? (yes/no): ")
        if answer.lower() != "yes":
            return f"🚫 BLOCKED: Action on production namespace '{namespace}' rejected"

    # ✅ MIDDLEWARE 3: Log the tool call
    log_entry = {
        "time": timestamp,
        "tool": tool_name,
        "params": params,
        "call_number": call_counts[tool_name]
    }
    tool_call_log.append(log_entry)
    print(f"\n   📋 [{timestamp}] TOOL CALL: {tool_name}")
    print(f"   📋 Params: {params}")

    # ✅ Execute the actual tool
    result = tool_func(*args)
    print(f"   📋 Result: {str(result)[:80]}...")

    return result

# ════════════════════════════════════
# REAL TOOLS — simulate SAP operations
# ════════════════════════════════════

@tool
def get_pod_status(namespace: str, pod_name: str) -> str:
    """Get the status of a Kubernetes pod in SAP SCI cluster."""
    def _execute():
        # Simulated kubectl output
        pods = {
            ("bizx", "bizx-car"): "CrashLoopBackOff — Restarts: 47 — Last exit: OOMKilled",
            ("mobile", "mobile-api"): "Running — Ready: 1/1 — Age: 2d",
            ("platformfoundations", "vault-agent"): "Running — Ready: 1/1 — Age: 5d",
        }
        return pods.get((namespace, pod_name),
               f"Pod {pod_name} not found in {namespace}")

    return middleware(
        "get_pod_status",
        {"namespace": namespace, "pod_name": pod_name},
        _execute
    )

@tool
def get_pod_logs(namespace: str, pod_name: str) -> str:
    """Get logs from a Kubernetes pod."""
    def _execute():
        logs = {
            ("bizx", "bizx-car"): """
ERROR: vault: context deadline exceeded
ERROR: auth/kubernetes: permission denied — token expired
ERROR: failed to initialize vault client: token TTL=0
FATAL: cannot start application without vault credentials""",
            ("mobile", "mobile-api"): "INFO: Server started on :8080\nINFO: Connected to MySQL",
        }
        return logs.get((namespace, pod_name), "No logs found")

    return middleware(
        "get_pod_logs",
        {"namespace": namespace, "pod_name": pod_name},
        _execute
    )

@tool
def check_argocd_status(app_name: str) -> str:
    """Check ArgoCD application sync status."""
    def _execute():
        apps = {
            "hxm-platform": "OutOfSync — Missing CRD: bizxconfiguration.hxm.sap.com",
            "bizx-app": "Synced — Healthy — Last sync: 2min ago",
            "mobile-app": "Synced — Healthy — Last sync: 5min ago",
        }
        return apps.get(app_name, f"App {app_name} not found in ArgoCD")

    return middleware(
        "check_argocd_status",
        {"app_name": app_name},
        _execute
    )

@tool
def check_vault_token(service_name: str) -> str:
    """Check Vault token status for a service."""
    def _execute():
        tokens = {
            "bizx-car": "EXPIRED — TTL: 0s — Expired 2h ago — Accessor: abc123",
            "mobile-api": "VALID — TTL: 18h — Expires: 2026-06-20 06:00",
            "vault-agent": "VALID — TTL: 23h — Auto-renewal: enabled",
        }
        return tokens.get(service_name,
               f"No token found for {service_name}")

    return middleware(
        "check_vault_token",
        {"service_name": service_name},
        _execute
    )

@tool
def create_jira_ticket(title: str, priority: str,
                       description: str) -> str:
    """Create a JIRA ticket for the incident."""
    def _execute():
        ticket_id = f"SCHCM-{len(tool_call_log) + 5800}"
        return (f"✅ JIRA ticket created: {ticket_id}\n"
                f"   Title: {title}\n"
                f"   Priority: {priority}\n"
                f"   URL: https://jira.sap.com/browse/{ticket_id}")

    return middleware(
        "create_jira_ticket",
        {"title": title, "priority": priority},
        _execute
    )

# ════════════════════════════════════
# AI AGENT WITH MIDDLEWARE TOOLS
# ════════════════════════════════════
tools = [
    get_pod_status,
    get_pod_logs,
    check_argocd_status,
    check_vault_token,
    create_jira_ticket
]

agent = create_react_agent(llm, tools)

print("🤖 SAP HCM AI AGENT WITH MIDDLEWARE")
print("="*55)
print("All tool calls are: validated + logged + rate-limited")
print("="*55)

# Test the agent
query = """
Investigate why bizx-car pod in bizx namespace is failing.
Check pod status, get logs, check vault token.
Then create a JIRA ticket with your findings.
"""

print(f"\n❓ Query: {query.strip()}")
print("="*55)

result = agent.invoke({
    "messages": [
        SystemMessage(content="""You are a SAP SCI DevOps AI agent.
        Investigate incidents step by step using tools.
        Always check pod status first, then logs, then vault.
        Create a JIRA ticket after investigation."""),
        HumanMessage(content=query)
    ]
})

print(f"\n{'='*55}")
print("📊 AGENT FINAL ANSWER:")
print("="*55)
print(result["messages"][-1].content)

# Show middleware audit log
print(f"\n{'='*55}")
print("📋 MIDDLEWARE AUDIT LOG:")
print("="*55)
for entry in tool_call_log:
    print(f"[{entry['time']}] {entry['tool']} "
          f"(call #{entry['call_number']}) — "
          f"params: {entry['params']}")
print(f"\nTotal tool calls: {len(tool_call_log)}")