from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r"D:\genai-learning\.env")
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ✅ STATE — Annotated List means messages ACCUMULATE
# operator.add means: new messages ADD to existing ones
# (not replace them — that's the key!)
class ChatState(TypedDict):
    messages: Annotated[List, operator.add]

# ✅ NODE — chat with AI
def chat_node(state: ChatState) -> ChatState:
    response = llm.invoke(state['messages'])
    return {"messages": [AIMessage(content=response.content)]}

# ✅ BUILD GRAPH with checkpointer
workflow = StateGraph(ChatState)
workflow.add_node("chat", chat_node)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)

# MemorySaver = checkpointer that saves state in memory
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# ✅ thread_id = unique session identifier
# Same thread_id = same conversation every time!
THREAD_ID = "sap-incident-investigation-001"
config = {"configurable": {"thread_id": THREAD_ID}}

print("="*55)
print("💾 LANGGRAPH CHECKPOINTER DEMO")
print("="*55)
print(f"Thread ID: {THREAD_ID}")
print("This conversation is checkpointed automatically!")
print("Type 'quit' to exit, 'history' to see full state\n")

# System message — set context once
app.invoke(
    {"messages": [SystemMessage(
        content="""You are an expert SAP HCM DevOps assistant
        for Anuradha's SCI production environment.
        Remember everything discussed in this session."""
    )]},
    config=config
)

# ✅ CONVERSATION LOOP — fully checkpointed
turn = 0
while True:
    user_input = input(f"\n[Turn {turn+1}] You: ")

    if user_input.lower() == 'quit':
        print("\n👋 Session saved! Restart program with same")
        print(f"   thread_id '{THREAD_ID}' to continue.")
        break

    if user_input.lower() == 'history':
        # Show the full checkpointed state
        state = app.get_state(config)
        messages = state.values.get("messages", [])
        print(f"\n📜 FULL CONVERSATION HISTORY ({len(messages)} messages):")
        for i, msg in enumerate(messages):
            role = type(msg).__name__.replace("Message", "")
            print(f"   [{i+1}] {role}: {str(msg.content)[:80]}...")
        continue

    # Send message — checkpointer saves state automatically!
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )

    # Get last AI message
    ai_response = result["messages"][-1].content
    print(f"\n🤖 AI: {ai_response}")
    turn += 1

# ✅ PROVE IT WORKS — show state was saved
print("\n" + "="*55)
print("📊 CHECKPOINT STATE SUMMARY:")
print("="*55)
state = app.get_state(config)
messages = state.values.get("messages", [])
print(f"Total messages saved in checkpoint: {len(messages)}")
print(f"Thread ID: {THREAD_ID}")
print(f"Next node: {state.next}")
print("\n✅ If you restart and use same thread_id,")
print("   the AI remembers this ENTIRE conversation!")