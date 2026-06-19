from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r"D:\genai-learning\.env")
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ✅ STEP 1: Create a rich SAP runbook document
RUNBOOK_PATH = r"D:\genai-learning\src\04_rag\sap_runbook.txt"

runbook_content = """
# SAP SCI HCM Infrastructure Runbook
# Team: Aarna Technologies | Environment: SCI Dev & Prod

## SECTION 1: Kubernetes Pod Troubleshooting

### Issue: CrashLoopBackOff
Symptom: Pod repeatedly crashes and restarts.
Root Causes: Vault token expired, missing secrets, OOM kill, bad image.

Fix Steps:
Step 1: Run kubectl logs <pod-name> -n <namespace> --previous
Step 2: Check if error mentions Vault token expired
Step 3: If Vault issue, run: vault token renew <token>
Step 4: Update Kubernetes secret: kubectl create secret generic vault-token
Step 5: Restart pod: kubectl rollout restart deployment/<name>
Step 6: Verify pod is Running: kubectl get pods -n <namespace>

Prevention: Set Vault token TTL to 24h and add renewal cron job.

### Issue: ImagePullBackOff
Symptom: Pod cannot pull container image from registry.
Root Causes: Wrong image tag, registry credentials expired, network issue.

Fix Steps:
Step 1: Check pod events: kubectl describe pod <pod-name>
Step 2: Verify image exists in registry
Step 3: Check imagePullSecret is configured correctly
Step 4: If credentials expired, rotate registry secret

## SECTION 2: MySQL HA Troubleshooting

### Issue: Write Traffic Routing to Read Replica
Symptom: Write operations failing, data inconsistency errors.
Root Cause: OpenStack Load Balancer sending writes to read-only replica.

Fix Steps:
Step 1: Check LB member weights via OpenStack CLI
Step 2: Run: openstack loadbalancer member list <pool-id>
Step 3: Set replica weight to 0: openstack loadbalancer member set --weight 0
Step 4: Verify primary receives all writes
Step 5: Update Terraform IaC to persist the fix permanently
Step 6: Test with write query to confirm

Prevention: Always set replica LB member weight to 0 in Terraform.

### Issue: MySQL Replication Broken
Symptom: Replica lag increasing, data out of sync.
Root Cause: GTID replication error after failover.

Fix Steps:
Step 1: Check replica status: SHOW SLAVE STATUS\G
Step 2: Note the error in Last_SQL_Error field
Step 3: Stop replica: STOP SLAVE
Step 4: Reset GTID: RESET MASTER
Step 5: Restart replication: START SLAVE
Step 6: Monitor lag: SHOW SLAVE STATUS\G

## SECTION 3: ArgoCD Troubleshooting

### Issue: Application OutOfSync
Symptom: ArgoCD shows app as OutOfSync, deployment not matching Git.
Root Causes: Missing CRD, manual cluster changes, Helm value drift.

Fix Steps:
Step 1: Check sync status: argocd app get <app-name>
Step 2: View diff: argocd app diff <app-name>
Step 3: If missing CRD: kubectl apply -f <crd-file.yaml>
Step 4: Force sync: argocd app sync <app-name> --force
Step 5: Verify all resources are Synced and Healthy

### Issue: ArgoCD Cannot Connect to Cluster
Symptom: Cluster shows Unknown status in ArgoCD.
Root Cause: Kubeconfig expired or cluster endpoint changed.

Fix Steps:
Step 1: Regenerate kubeconfig using Gardener shoot script
Step 2: Update ArgoCD cluster secret
Step 3: Test connection: argocd cluster get <cluster-name>

## SECTION 4: Dynatrace & Monitoring

### Setup: Deploy Dynatrace OneAgent via Ansible
Run the Ansible playbook for your environment:
ansible-playbook -i inventory.hcm-dev deploy_dynatrace.yml --vault-password-file .pass

Verify agent is reporting to:
apm.btp.sci-prod.scs.sap/e/c41fce75-4c98-4150-81a6-0a7f2f9b89f4

### Setup: Vector Log Forwarding to Splunk
Vector ships logs via NATS TLS to:
nats.security.sci-dev.scs.sap:4222 on subject hcm.logs
Splunk at: splunk.security.sci-prod.scs.sap
"""

# Write runbook to file
with open(RUNBOOK_PATH, 'w') as f:
    f.write(runbook_content)
print(f"✅ Runbook saved to: {RUNBOOK_PATH}")

# ✅ STEP 2: Compare 3 chunking strategies
print("\n" + "="*55)
print("📊 COMPARING CHUNKING STRATEGIES")
print("="*55)

# Strategy 1: Small fixed chunks (bad)
splitter_small = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""]
)

# Strategy 2: Medium chunks with overlap (better)
splitter_medium = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

# Strategy 3: Section-aware chunks (best for runbooks)
splitter_best = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["## ", "### ", "\n\n", "\n", " "]
)

# Load document
loader = TextLoader(RUNBOOK_PATH)
docs = loader.load()

chunks_small  = splitter_small.split_documents(docs)
chunks_medium = splitter_medium.split_documents(docs)
chunks_best   = splitter_best.split_documents(docs)

print(f"\nStrategy 1 (size=100,  overlap=0)  : {len(chunks_small):3d} chunks")
print(f"Strategy 2 (size=300,  overlap=50) : {len(chunks_medium):3d} chunks")
print(f"Strategy 3 (size=500,  overlap=100): {len(chunks_best):3d} chunks")

print(f"\n📌 Example — Strategy 1 chunk (too small, cuts mid-step):")
print(f"   '{chunks_small[10].page_content[:120]}'")

print(f"\n📌 Example — Strategy 3 chunk (keeps steps together):")
print(f"   '{chunks_best[3].page_content[:200]}'")

# ✅ STEP 3: Build RAG with best chunking strategy
print("\n" + "="*55)
print("🤖 BUILDING RAG WITH BEST CHUNKING STRATEGY")
print("="*55)

embeddings = FakeEmbeddings(size=384)
vectorstore = Chroma.from_documents(chunks_best, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
You are a SAP HCM DevOps expert.
Answer using ONLY the context below.
If not in context, say "Not in runbook."

Context:
{context}

Question: {question}
""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ✅ STEP 4: Test with real SAP questions
questions = [
    "How do I fix a CrashLoopBackOff pod?",
    "What steps fix MySQL write traffic going to replica?",
    "How do I resolve ArgoCD OutOfSync issue?",
    "Where does Vector ship logs to?"
]

print()
for q in questions:
    print(f"❓ {q}")
    answer = rag_chain.invoke(q)
    print(f"✅ {answer[:200]}")
    print("-"*55)