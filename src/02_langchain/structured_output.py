from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r"D:\genai-learning\.env")

# ✅ Define the STRUCTURE — like a form with specific fields
class BugReport(BaseModel):
    severity: str = Field(description="high, medium, or low")
    affected_component: str = Field(description="which system is affected")
    suggested_action: str = Field(description="what to do to fix it")
    users_affected: str = Field(description="how many users are impacted")
    summary: str = Field(description="one line summary of the bug")

# ✅ Connect to AI
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ✅ Tell AI to return structured output
structured_llm = llm.with_structured_output(BugReport)

# ✅ Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior SAP DevOps engineer. Analyze bug reports and extract key information."),
    ("human", "Analyze this bug report:\n{bug_report}")
])

chain = prompt | structured_llm

# ✅ Test with YOUR real SAP incidents
incidents = [
    """Pod bizx-car in bizx namespace is CrashLoopBackOff.
    Vault token expired 2 hours ago.
    500 engineers cannot login to SAP SuccessFactors.""",

    """MySQL replica in stage-mobile receiving write traffic.
    OpenStack LB routing writes to read-only node.
    15 mobile app users getting errors since 30 minutes.""",

    """ArgoCD app hxm-platform OutOfSync in dev-prod shoot.
    Missing CRD causing deployment failure.
    Dev environment only, no production impact."""
]

for i, incident in enumerate(incidents, 1):
    print(f"\n{'='*55}")
    print(f"📋 Incident {i}")
    print(f"{'='*55}")
    result = chain.invoke({"bug_report": incident})
    print(f"Severity:          {result.severity}")
    print(f"Component:         {result.affected_component}")
    print(f"Action:            {result.suggested_action}")
    print(f"Users Affected:    {result.users_affected}")
    print(f"Summary:           {result.summary}")
    print(f"\n📦 As JSON dict:")
    print(result.model_dump())