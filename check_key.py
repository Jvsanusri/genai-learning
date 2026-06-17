from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
print("Key starts with:", key[:15] if key else "NOT FOUND")
print("Key length:", len(key) if key else 0)