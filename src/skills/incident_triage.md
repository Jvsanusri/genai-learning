# SKILL: Incident Triage
## Purpose
Analyze SAP HCM incidents and classify them for the SCI team.

## Inputs Required
- incident_description: Full text of the incident

## Steps to Follow
1. Read the incident description carefully
2. Identify the affected component (Kubernetes/MySQL/ArgoCD/Vault/Monitoring)
3. Classify severity: high (production impact) / medium (dev blocked) / low (minor)
4. Identify root cause category: infrastructure / database / security / monitoring
5. Suggest immediate action in ONE sentence

## Output Format
Return EXACTLY this structure:
- Component: [component name]
- Severity: [high/medium/low]
- Category: [infrastructure/database/security/monitoring]
- Root Cause: [one sentence]
- Action: [one sentence immediate fix]

## Rules
- Never guess — only state what is clear from the description
- If production users are impacted → always high severity
- Vault token issues → always security category