from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
response = llm.invoke("What is an AI agent? Answer in one sentence.")
print(response.content)