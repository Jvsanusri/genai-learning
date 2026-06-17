from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load API key
load_dotenv(dotenv_path=r"D:\genai-learning\.env")

# Connect to AI
chat = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ✅ Define a reusable template with variables
template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {role}. Explain things clearly and simply."),
    ("human", "Explain {topic} in simple terms. Give a real world example.")
])

# 🎯 Test 1: AI acts as Python Tutor
print("=" * 50)
print("🐍 Python Tutor Mode")
print("=" * 50)
chain = template | chat
response = chain.invoke({
    "role": "Python programming tutor",
    "topic": "functions in Python"
})
print(response.content)

# 🎯 Test 2: AI acts as DevOps Expert
print("\n" + "=" * 50)
print("⚙️ DevOps Expert Mode")
print("=" * 50)
response = chain.invoke({
    "role": "DevOps and Kubernetes expert",
    "topic": "what is a Kubernetes pod"
})
print(response.content)

# 🎯 Test 3: AI acts as GenAI Teacher
print("\n" + "=" * 50)
print("🤖 GenAI Teacher Mode")
print("=" * 50)
response = chain.invoke({
    "role": "GenAI and LangChain teacher",
    "topic": "what is a Large Language Model (LLM)"
})
print(response.content)