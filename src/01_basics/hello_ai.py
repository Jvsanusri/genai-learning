# Step 1: Import the libraries we need
from groq import Groq
from dotenv import load_dotenv
import os

# Step 2: Load the API key from .env file
# We need to tell it exactly WHERE the .env file is
load_dotenv(dotenv_path=r"D:\genai-learning\.env")

# Step 3: Check if key loaded (for debugging)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ API key not found! Check your .env file")
    exit()
else:
    print("✅ API key loaded successfully!")

# Step 4: Connect to Groq AI
client = Groq(api_key=api_key)

# Step 5: Send a message to AI and get a response
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Hello! Who are you and what can you do?"}
    ]
)

# Step 6: Print the AI's response
print("\n🤖 AI says:")
print(response.choices[0].message.content)