import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from rag.retriever import CodeRetriever

load_dotenv()


class PerfIssue(BaseModel):
    file        : str  = Field(description="filename where issue was found")
    line        : int  = Field(description="line number of the issue")
    severity    : str  = Field(description="LOW, MEDIUM, or HIGH")
    category    : str  = Field(description="n_plus_one, complexity, memory_leak, "
                                           "blocking_io, inefficient_query, or redundant_computation")
    message     : str  = Field(description="clear description of the performance issue")
    suggestion  : str  = Field(description="exactly how to fix it")
    impact      : str  = Field(description="estimated performance impact")


class PerfReview(BaseModel):
    pr_summary    : str = Field(description="one sentence performance summary")
    issues        : List[PerfIssue] = Field(description="all performance issues found")
    perf_score    : int = Field(description="performance score 1-10, 10 being best")
    approved      : bool = Field(description="true if PR passes performance review")


PERF_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are CodeGuard's Performance Agent — an expert in
code performance, algorithmic complexity, and system efficiency.

Performance issues you detect:

1. N+1 Query Problem:
   - Database queries inside loops
   - Missing eager loading / prefetch_related
   - Repeated identical queries

2. Algorithmic Complexity:
   - O(n²) or worse nested loops on large datasets
   - Sorting inside loops
   - Linear search instead of hash lookup

3. Memory Issues:
   - Loading entire dataset into memory
   - Missing pagination
   - Large object creation in tight loops
   - Missing generator usage

4. Blocking I/O:
   - Synchronous I/O in async functions
   - Missing async/await on I/O calls
   - Large file reads without streaming

5. Inefficient Queries:
   - Missing database indexes
   - SELECT * when specific columns needed
   - Missing query result caching

6. Redundant Computation:
   - Same calculation repeated in loop
   - Missing memoization
   - Recomputing values that could be precomputed

Severity:
- LOW    : minor optimization opportunity
- MEDIUM : noticeable performance impact
- HIGH   : significant bottleneck, must fix

{format_instructions}"""
    ),
    (
        "human",
        """Analyze this pull request for performance issues.

{team_context}

PR Diff:
{diff}

Return your performance review in the exact JSON format specified."""
    )
])


class PerformanceAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.parser    = JsonOutputParser(pydantic_object=PerfReview)
        self.chain     = PERF_PROMPT | self.llm | self.parser
        self.retriever = CodeRetriever()

    async def review(self, diff_chunks: list[dict]) -> dict:
        formatted_diff = self._format_diff(diff_chunks)

        print(f"\n⚡ Performance Agent retrieving context...")
        similar_chunks = self.retriever.retrieve(
            query=formatted_diff[:1000],
            repo="Tejesh0209/SentinelAI",
            top_k=3
        )
        team_context = self.retriever.format_for_prompt(similar_chunks)

        print(f"⚡ Performance Agent analyzing bottlenecks...")

        result = await self.chain.ainvoke({
            "diff"                : formatted_diff,
            "team_context"        : team_context,
            "format_instructions" : self.parser.get_format_instructions()
        })

        self._print_review(result)
        return result

    def _format_diff(self, diff_chunks: list[dict]) -> str:
        formatted = []
        for chunk in diff_chunks:
            formatted.append(f"""
File: {chunk['filename']} ({chunk['status']})
Additions: +{chunk['additions']} | Deletions: -{chunk['deletions']}
Changes:
{chunk['patch']}
""")
        return "\n---\n".join(formatted)

    def _print_review(self, review: dict):
        print(f"\n{'='*60}")
        print(f"⚡ PERFORMANCE REVIEW COMPLETE")
        print(f"{'='*60}")
        print(f"Summary    : {review.get('pr_summary', 'N/A')}")
        print(f"Perf Score : {review.get('perf_score', 0)}/10")
        print(f"Approved   : {review.get('approved', False)}")

        issues = review.get('issues', [])
        print(f"\n🔍 Performance Issues Found: {len(issues)}")

        for i, issue in enumerate(issues, 1):
            severity_emoji = {
                "HIGH"  : "🔴",
                "MEDIUM": "🟡",
                "LOW"   : "🟢"
            }.get(issue.get('severity', 'LOW'), "⚪")

            print(f"\n  {i}. {severity_emoji} [{issue.get('severity')}] "
                  f"{issue.get('category', '').upper()}")
            print(f"     File   : {issue.get('file')}")
            print(f"     Line   : {issue.get('line')}")
            print(f"     Impact : {issue.get('impact')}")
            print(f"     Issue  : {issue.get('message')}")
            print(f"     Fix    : {issue.get('suggestion')}")

        print(f"\n{'='*60}\n")