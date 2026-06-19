"""
SAP SCI MCP Server
Exposes SAP HCM infrastructure tools via MCP protocol
Any AI agent can connect and use these tools!
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Optional
from datetime import datetime
import uvicorn
import json

# ✅ Create FastAPI app as MCP server
app = FastAPI(
    title="SAP SCI MCP Server",
    description="MCP Server exposing SAP HCM infrastructure tools",
    version="1.0.0"
)

# ════════════════════════════════════
# MCP SERVER STRUCTURE
# Every MCP server exposes 3 things:
# 1. Tools     — actions AI can take
# 2. Resources — data AI can read
# 3. Prompts   — reusable templates
# ════════════════════════════════════

# ✅ TOOL DEFINITIONS — what this MCP server offers
TOOLS = {
    "get_pod_status": {
        "name": "get_pod_status",
        "description": "Get Kubernetes pod status in SAP SCI cluster",
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Kubernetes namespace (e.g. bizx, mobile)"
                },
                "pod_name": {
                    "type": "string",
                    "description": "Name of the pod to check"
                }
            },
            "required": ["namespace", "pod_name"]
        }
    },
    "get_vault_token_status": {
        "name": "get_vault_token_status",
        "description": "Check HashiCorp Vault token status for a service",
        "inputSchema": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "Name of the service to check vault token"
                }
            },
            "required": ["service_name"]
        }
    },
    "get_argocd_status": {
        "name": "get_argocd_status",
        "description": "Check ArgoCD application sync status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "ArgoCD application name"
                }
            },
            "required": ["app_name"]
        }
    },
    "get_mysql_status": {
        "name": "get_mysql_status",
        "description": "Check MySQL HA replication and LB status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "environment": {
                    "type": "string",
                    "description": "Environment: stage-mobile or prod-mobile"
                }
            },
            "required": ["environment"]
        }
    },
    "create_incident_ticket": {
        "name": "create_incident_ticket",
        "description": "Create JIRA incident ticket for SAP SCI issues",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Ticket title"
                },
                "severity": {
                    "type": "string",
                    "description": "high, medium, or low"
                },
                "description": {
                    "type": "string",
                    "description": "Full incident description"
                }
            },
            "required": ["title", "severity", "description"]
        }
    }
}

# ✅ RESOURCE DEFINITIONS — data AI can read
RESOURCES = {
    "sap://runbooks/kubernetes": {
        "uri": "sap://runbooks/kubernetes",
        "name": "Kubernetes Runbook",
        "description": "SAP SCI Kubernetes troubleshooting guide",
        "mimeType": "text/plain"
    },
    "sap://runbooks/mysql": {
        "uri": "sap://runbooks/mysql",
        "name": "MySQL HA Runbook",
        "description": "MySQL HA troubleshooting and fix procedures",
        "mimeType": "text/plain"
    },
    "sap://team/contacts": {
        "uri": "sap://team/contacts",
        "name": "Team Contacts",
        "description": "SAP SCI on-call and team contact information",
        "mimeType": "application/json"
    }
}

# ════════════════════════════════════
# MCP ENDPOINTS
# Standard MCP protocol endpoints
# ════════════════════════════════════

# ✅ List available tools
@app.get("/mcp/tools")
def list_tools():
    return {
        "tools": list(TOOLS.values()),
        "server": "SAP SCI MCP Server v1.0"
    }

# ✅ Execute a tool
class ToolCallRequest(BaseModel):
    name: str
    arguments: dict

@app.post("/mcp/tools/call")
def call_tool(request: ToolCallRequest):
    tool_name = request.name
    args = request.arguments
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n🔧 [{timestamp}] MCP Tool Called: {tool_name}")
    print(f"   Args: {args}")

    # ✅ Tool implementations
    if tool_name == "get_pod_status":
        pods = {
            ("bizx", "bizx-car"): {
                "status": "CrashLoopBackOff",
                "restarts": 47,
                "namespace": "bizx",
                "last_exit": "OOMKilled",
                "age": "2d"
            },
            ("mobile", "mobile-api"): {
                "status": "Running",
                "restarts": 0,
                "namespace": "mobile",
                "ready": "1/1",
                "age": "5d"
            }
        }
        key = (args.get("namespace"), args.get("pod_name"))
        result = pods.get(key, {
            "status": "Unknown",
            "message": f"Pod {args.get('pod_name')} not found"
        })

    elif tool_name == "get_vault_token_status":
        tokens = {
            "bizx-car": {
                "status": "EXPIRED",
                "ttl": 0,
                "expired_since": "2h ago",
                "accessor": "abc123xyz"
            },
            "mobile-api": {
                "status": "VALID",
                "ttl": "18h",
                "expires": "2026-06-20 06:00"
            }
        }
        result = tokens.get(args.get("service_name"), {
            "status": "NOT_FOUND",
            "message": "No token found for this service"
        })

    elif tool_name == "get_argocd_status":
        apps = {
            "hxm-platform": {
                "sync_status": "OutOfSync",
                "health": "Degraded",
                "reason": "Missing CRD: bizxconfiguration.hxm.sap.com"
            },
            "bizx-app": {
                "sync_status": "Synced",
                "health": "Healthy",
                "last_sync": "2min ago"
            }
        }
        result = apps.get(args.get("app_name"), {
            "sync_status": "Unknown",
            "message": "App not found"
        })

    elif tool_name == "get_mysql_status":
        mysql = {
            "stage-mobile": {
                "primary": "Running",
                "replica": "Running",
                "lb_weight_primary": 1,
                "lb_weight_replica": 0,
                "replication_lag": "0s",
                "status": "Healthy"
            },
            "prod-mobile": {
                "primary": "Running",
                "replica": "Running — receiving writes!",
                "lb_weight_primary": 1,
                "lb_weight_replica": 1,
                "replication_lag": "45s",
                "status": "WARNING: Replica getting write traffic!"
            }
        }
        result = mysql.get(args.get("environment"), {
            "status": "Unknown environment"
        })

    elif tool_name == "create_incident_ticket":
        ticket_id = f"SCHCM-{5900 + len(args['title'])}"
        result = {
            "ticket_id": ticket_id,
            "status": "Created",
            "priority": "P1" if args.get("severity") == "high" else "P2",
            "url": f"https://jira.sap.com/browse/{ticket_id}",
            "created_at": timestamp
        }

    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    print(f"   Result: {json.dumps(result)[:100]}...")
    return {
        "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
        "isError": False
    }

# ✅ List resources
@app.get("/mcp/resources")
def list_resources():
    return {"resources": list(RESOURCES.values())}

# ✅ Read a resource
@app.get("/mcp/resources/read")
def read_resource(uri: str):
    resources_content = {
        "sap://runbooks/kubernetes": """
KUBERNETES RUNBOOK — SAP SCI
CrashLoopBackOff Fix:
1. kubectl logs <pod> --previous
2. Check for Vault token expiry
3. vault token renew && kubectl rollout restart
""",
        "sap://runbooks/mysql": """
MYSQL HA RUNBOOK — SAP SCI
Write to Replica Fix:
1. openstack loadbalancer member list <pool-id>
2. Set replica weight to 0
3. Update Terraform IaC
""",
        "sap://team/contacts": json.dumps({
            "on_call": "Prabhakar Aluri",
            "slack": "#sci-devops-alerts",
            "jira_project": "SCHCM",
            "escalation": "senior-sre@sap.com"
        })
    }
    content = resources_content.get(uri, "Resource not found")
    return {"contents": [{"uri": uri, "text": content}]}

# ✅ Server info
@app.get("/mcp/info")
def server_info():
    return {
        "name": "SAP SCI MCP Server",
        "version": "1.0.0",
        "description": "Exposes SAP HCM infrastructure tools via MCP",
        "tools_count": len(TOOLS),
        "resources_count": len(RESOURCES),
        "protocol": "MCP/1.0"
    }

if __name__ == "__main__":
    print("🚀 Starting SAP SCI MCP Server...")
    print("   Tools available:", list(TOOLS.keys()))
    print("   Server: http://localhost:8001")
    print("   Docs:   http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)