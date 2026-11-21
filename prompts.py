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
Answer the question using ONLY the code context below.
If the context is not enough, say "I don't know based on the codebase."
Cite sources inline like [path:start-end]. Be concise.
{feedback}
Context:
{context}

Question: {question}"""

VERIFY = """TASK: VERIFY
Check whether every factual claim in the ANSWER is supported by the CONTEXT.
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
