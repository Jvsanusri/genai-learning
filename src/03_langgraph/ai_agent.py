from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_react_agent
from dotenv import load_dotenv
import os

# Load API key
load_dotenv(dotenv_path=r"D:\genai-learning\.env")

# Connect to AI
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ✅ Define TOOLS the agent can use

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Fake weather data for learning
    weather_data = {
        "dallas": "Sunny, 95°F (35°C)",
        "london": "Cloudy, 62°F (17°C)",
        "tokyo": "Rainy, 75°F (24°C)",
        "berlin": "Partly cloudy, 68°F (20°C)"
    }
    return weather_data.get(city.lower(), "Weather data not available")

# ✅ Create the Agent with tools
tools = [add_numbers, multiply_numbers, get_weather]
agent = create_react_agent(model, tools)

print("🤖 AI Agent Ready!\n")

# 🎯 Test 1: Math problem
print("=" * 50)
print("Test 1: Math")
print("=" * 50)
response = agent.invoke({
    "messages": [{"role": "user", "content": "What is 25 multiplied by 48? Is it greater than 1000?"}]
})
print(response["messages"][-1].content)

# 🎯 Test 2: Weather
print("\n" + "=" * 50)
print("Test 2: Weather")
print("=" * 50)
response = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather in Dallas?"}]
})
print(response["messages"][-1].content)

# 🎯 Test 3: Multi-step
print("\n" + "=" * 50)
print("Test 3: Multi-step thinking")
print("=" * 50)
response = agent.invoke({
    "messages": [{"role": "user", "content": "Add 150 and 200, then tell me the weather in Tokyo"}]
})
print(response["messages"][-1].content)