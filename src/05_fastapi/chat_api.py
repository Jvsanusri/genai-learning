from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r"D:\genai-learning\.env")

# ✅ Create FastAPI app
app = FastAPI(
    title="SAP HCM AI Assistant API",
    description="AI API for SAP SuccessFactors HCM team by Anuradha",
    version="1.0.0"
)

# ✅ Connect to AI
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ✅ Request and Response models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_count: int

# ✅ In-memory session storage
# Each session_id = one user's conversation
sessions = {}

# ✅ Health check — test if API is running
@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "SAP HCM AI Assistant is live!",
        "team": "SCI DevOps"
    }

# ✅ Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Get or create session for this user
    if request.session_id not in sessions:
        sessions[request.session_id] = [
            SystemMessage(content="""You are an expert SAP SuccessFactors 
            HCM DevOps assistant. You help with Kubernetes, ArgoCD, 
            MySQL, Vault, and SAP BizX issues on SAP Sovereign Cloud.""")
        ]

    # Add user message
    sessions[request.session_id].append(
        HumanMessage(content=request.message)
    )

    # Get AI response
    try:
        response = llm.invoke(sessions[request.session_id])
        sessions[request.session_id].append(
            AIMessage(content=response.content)
        )
        return ChatResponse(
            response=response.content,
            session_id=request.session_id,
            message_count=len(sessions[request.session_id])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Get all active sessions
@app.get("/sessions")
def get_sessions():
    return {
        "active_sessions": list(sessions.keys()),
        "total": len(sessions)
    }

# ✅ Clear a session
@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": "Session not found"}

# ✅ Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)