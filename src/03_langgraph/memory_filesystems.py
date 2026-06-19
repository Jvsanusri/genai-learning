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

# ✅ STATE
# Annotated + operator.add means messages ACCUMULATE
# New messages ADD to existing — not replace them!
class ChatState(TypedDict):
    messages: Annotated[List, operator.add]

# ✅ NODE
def chat_node(state: ChatState) -> ChatState:
    response = llm.invoke(state['messages'])
    return {"messages": [AIMessage(content=response.content)]}

# ✅ BUILD GRAPH with checkpointer
workflow = StateGraph(ChatState)
workflow.add_node("chat", chat_node)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)

# ✅ MemorySaver = saves state automatically after each node
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# ✅ thread_id = unique session key
# Same thread_id = same conversation always!
THREAD_ID = "sap-incident-20260620"
config = {"configurable": {"thread_id": THREAD_ID}}

print("="*55)
print("💾 LANGGRAPH CHECKPOINTER DEMO")
print("="*55)
print(f"Thread ID : {THREAD_ID}")
print("Commands  : 'quit' to exit | 'history' to see state")
print("="*55)

# Set context once at start
app.invoke(
    {"messages": [SystemMessage(
        content="""You are a SAP HCM DevOps assistant
        for Anuradha's SCI production environment.
        Remember everything in this session."""
    )]},
    config=config
)

# ✅ CHAT LOOP — fully checkpointed automatically
turn = 0
while True:
    user_input = input(f"\n[Turn {turn+1}] You: ")

    if user_input.lower() == 'quit':
        print(f"\n✅ Session saved with thread_id: {THREAD_ID}")
        print("   Restart and use same thread_id to resume!")
        break

    if user_input.lower() == 'history':
        # Show full saved state
        state = app.get_state(config)
        msgs = state.values.get("messages", [])
        print(f"\n📜 CHECKPOINT HISTORY ({len(msgs)} messages saved):")
        for i, msg in enumerate(msgs):
            role = type(msg).__name__.replace("Message", "")
            print(f"   [{i+1}] {role}: {str(msg.content)[:70]}...")
        continue

    # Send message — state saved automatically!
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )

    ai_response = result["messages"][-1].content
    print(f"\n🤖 AI: {ai_response}")
    turn += 1

# ✅ Show final checkpoint summary
print("\n" + "="*55)
print("📊 FINAL CHECKPOINT SUMMARY:")
print("="*55)
state = app.get_state(config)
msgs = state.values.get("messages", [])
print(f"Messages checkpointed : {len(msgs)}")
print(f"Thread ID             : {THREAD_ID}")
print(f"\n✅ Same thread_id tomorrow = full history restored!")
