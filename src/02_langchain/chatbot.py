from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

# Load API key
load_dotenv(dotenv_path=r"D:\genai-learning\.env")

# Connect to AI model
chat = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# System message tells AI how to behave
history = [
    SystemMessage(content="You are a helpful assistant. Answer the user's questions in a friendly and concise way.")
]

print("🤖 Chatbot ready! Type 'quit' to exit\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        print("👋 Goodbye!")
        break

    # Add user message to history
    history.append(HumanMessage(content=user_input))
    
    # Send full history to AI
    response = chat.invoke(history)
    
    # Add AI reply to history
    history.append(AIMessage(content=response.content))
    
    print(f"🤖 AI: {response.content}\n")