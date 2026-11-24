ROUTE = """TASK: ROUTE
You route questions about a software codebase to a search strategy.
Strategies:
- vector: conceptual questions ("how does X work", "where is Y handled")
- keyword: questions containing exact names (function, class, file, error text)
- hybrid: unsure, or the question mixes concepts and exact names
- none: general programming knowledge that does not need this codebase
Return JSON only: {{"strategy": "vector|keyword|hybrid|none", "reason": "<short reason>"}}

Question: {question}"""

GRADE = """TASK: GRADE
Question: {question}

Below are numbered code chunks. A chunk is RELEVANT only if it contains information needed to answer the question.
Return JSON only: {{"relevant": [<numbers of the relevant chunks>]}}

{chunks}"""

REWRITE = """TASK: REWRITE
The search for this question returned irrelevant code.
Question: {question}
Previous search query: {query}
Files that came back (not helpful): {files}
Write ONE better search query using likely identifiers, file names and keywords that would appear in the code.
Return only the query text."""

ANSWER = """TASK: ANSWER
Answer the user's question about this codebase using ONLY the code context below.
Answer exactly what was asked: cover everything in the context that is relevant to the question,
and leave out anything that is not. Do not describe unrelated parts of the code.
If the context is not enough, reply with exactly: I don't know based on the codebase.

Otherwise reply in exactly this format, in plain text:
ANSWER: <one or two sentences that directly answer the question>
EXPLANATION: <3 to 6 sentences in simple language for a beginner. When you use a technical term, explain it in brackets the first time.>
WHERE:
- <path:start-end> - <what that code does>

Cite sources inline like [path:start-end]. Only cite files that appear in the context.
{feedback}
Context:
{context}

Question: {question}"""

VERIFY = """TASK: VERIFY
Check whether every factual claim about THIS CODEBASE in the ANSWER is supported by the CONTEXT.
Ignore general definitions of technical terms (for example a bracketed explanation of what an API is).
Return JSON only: {{"supported": true, "unsupported_claims": []}}
Set "supported" to false and list the claims if anything is not backed by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}"""

DIRECT = """TASK: DIRECT
Answer this general programming question briefly.

Question: {question}"""

JUDGE = """TASK: JUDGE
Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}
Is the candidate answer correct, meaning consistent with the reference and containing its key point?
Return JSON only: {{"correct": true}} or {{"correct": false}}"""

GENQ = """TASK: GENQ
Below is a code chunk from a repository. Write ONE question a developer might ask about this code
that can be answered from it, plus the reference answer. Do not mention "chunk" or line numbers.
Return JSON only: {{"question": "...", "answer": "..."}}

{chunk}"""


OVERVIEW = """TASK: OVERVIEW
Explain this GitHub repository to a beginner in simple language.
Use only the README and file list below. Do not invent features.
Reply in exactly this format, in plain text:
WHAT IT DOES: <2 to 3 sentences>
TECH STACK: <comma-separated list>
KEY FILES:
- <path> - <purpose in a few words>
KEY TERMS:
- <term> - <one-line plain explanation>

README:
{readme}

FILES:
{files}"""
