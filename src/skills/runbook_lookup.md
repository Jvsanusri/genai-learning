# SKILL: Runbook Lookup
## Purpose
Find the correct fix procedure from SAP SCI runbooks.

## Inputs Required
- component: affected system (kubernetes/mysql/argocd/vault)
- issue_type: what went wrong

## Steps to Follow
1. Identify the component and issue type
2. Search for matching runbook section
3. Extract the exact fix steps
4. Present steps in numbered format
5. Add a prevention note at the end

## Output Format
Return EXACTLY this structure:
## Fix Steps for [component] - [issue_type]
1. [step 1]
2. [step 2]
3. [step 3]

## Prevention
[one prevention tip]

## Rules
- Only use steps that are specific and actionable
- Always include kubectl/openstack/ansible commands where relevant
- Never skip steps — incomplete fixes cause more incidents