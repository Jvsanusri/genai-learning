from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import FakeEmbeddings
from dotenv import load_dotenv
import os

# Load API key
load_dotenv(dotenv_path=r"D:\genai-learning\.env")

print("📚 Loading document...")

# Step 1: Load document
loader = TextLoader(r"D:\genai-learning\src\04_rag\my_document.txt")
documents = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
chunks = splitter.split_documents(documents)
print(f"✅ Document split into {len(chunks)} chunks")

# Step 3: Store in vector database
# Using FakeEmbeddings for learning (no heavy downloads needed)
print("💾 Storing in vector database...")
embeddings = FakeEmbeddings(size=384)
vectorstore = Chroma.from_documents(chunks, embeddings)

# Step 4: Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Step 5: Connect to AI
print("🤖 Connecting to AI...")
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# Step 6: Build RAG chain
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If the answer is not in the context, say "I don't know".

Context: {context}

Question: {question}
""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n✅ RAG System Ready!\n")

# Test questions
questions = [
    "How do I access the Kubernetes cluster?",
    "What should I do if a pod is in CrashLoopBackOff?",
    "Who is the Infrastructure Lead?",
    "What is the ArgoCD UI URL?"
]

for question in questions:
    print(f"❓ {question}")
    response = rag_chain.invoke(question)
    print(f"✅ {response}\n")
    print("-" * 50)