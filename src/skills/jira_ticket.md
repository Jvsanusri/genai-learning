# SKILL: JIRA Ticket Creator
## Purpose
Create a well-structured JIRA ticket for SAP SCI incidents.

## Inputs Required
- incident: original incident description
- severity: high/medium/low
- component: affected system
- root_cause: identified root cause
- fix_steps: steps to resolve

## Steps to Follow
1. Create a clear, searchable ticket title
2. Write a concise summary (2 sentences max)
3. List environment details
4. Add reproduction steps
5. Add fix steps
6. Set correct priority based on severity

## Output Format
**Title:** [SAP SCI] [Component] - [Issue] - [Date]
**Priority:** P1/P2/P3 (P1=high, P2=medium, P3=low)
**Component:** [component]
**Environment:** SAP Sovereign Cloud SCI Prod/Dev

**Summary:**
[2 sentence summary]

**Steps to Reproduce:**
1. [step]

**Fix:**
1. [fix step]

**Labels:** sap-sci, [component], [severity]