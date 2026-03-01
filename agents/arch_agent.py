import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from rag.retriever import CodeRetriever

load_dotenv()


class ArchIssue(BaseModel):
    file       : str = Field(description="filename where issue was found")
    line       : int = Field(description="line number of the issue")
    severity   : str = Field(description="LOW, MEDIUM, or HIGH")
    principle  : str = Field(description="SRP, OCP, LSP, ISP, DIP, DRY, KISS, or COUPLING")
    message    : str = Field(description="clear description of the architectural issue")
    suggestion : str = Field(description="exactly how to fix it")
    impact     : str = Field(description="long-term impact if not fixed")


class ArchReview(BaseModel):
    pr_summary  : str = Field(description="one sentence architecture summary")
    issues      : List[ArchIssue] = Field(description="all architectural issues found")
    arch_score  : int = Field(description="architecture quality score 1-10")
    approved    : bool = Field(description="true if PR passes architecture review")


ARCH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are CodeGuard's Architecture Agent — an expert in
software design principles and clean architecture.

You review code for violations of:

SOLID Principles:
- SRP (Single Responsibility): class/function does ONE thing only
- OCP (Open/Closed): open for extension, closed for modification
- LSP (Liskov Substitution): subclasses must be substitutable
- ISP (Interface Segregation): no fat interfaces, split them up
- DIP (Dependency Inversion): depend on abstractions, not concretions

Design Principles:
- DRY (Don't Repeat Yourself): no duplicated logic
- KISS (Keep It Simple): unnecessary complexity
- COUPLING: tight coupling between modules, circular dependencies
- COHESION: unrelated responsibilities in same class/module

Code Smells:
- God classes (doing too many things)
- Long parameter lists (more than 4 params)
- Feature envy (method uses another class's data too much)
- Shotgun surgery (one change requires many small changes)
- Dead code (unused functions, variables)

Severity:
- LOW    : minor design smell
- MEDIUM : should be refactored soon
- HIGH   : significant design problem, technical debt

{format_instructions}"""
    ),
    (
        "human",
        """Review this pull request for architectural issues.

{team_context}

PR Diff:
{diff}

Return your architecture review in the exact JSON format specified."""
    )
])


class ArchAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.parser    = JsonOutputParser(pydantic_object=ArchReview)
        self.chain     = ARCH_PROMPT | self.llm | self.parser
        self.retriever = CodeRetriever()

    async def review(self, diff_chunks: list[dict]) -> dict:
        formatted_diff = self._format_diff(diff_chunks)

        print(f"\nArchitecture Agent retrieving context...")
        similar_chunks = self.retriever.retrieve(
            query=formatted_diff[:1000],
            repo="Tejesh0209/SentinelAI",
            top_k=3
        )
        team_context = self.retriever.format_for_prompt(similar_chunks)

        print(f"Architecture Agent analyzing design...")
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
        print(f"🏗️  ARCHITECTURE REVIEW COMPLETE")
        print(f"{'='*60}")
        print(f"Summary    : {review.get('pr_summary', 'N/A')}")
        print(f"Arch Score : {review.get('arch_score', 0)}/10")
        print(f"Approved   : {review.get('approved', False)}")
        issues = review.get('issues', [])
        print(f"\nArchitecture Issues Found: {len(issues)}")
        for i, issue in enumerate(issues, 1):
            severity_emoji = {
                "HIGH"  : "🔴",
                "MEDIUM": "🟡",
                "LOW"   : "🟢"
            }.get(issue.get('severity', 'LOW'), "⚪")
            print(f"\n  {i}. {severity_emoji} [{issue.get('severity')}] "
                  f"{issue.get('principle', '').upper()}")
            print(f"     File    : {issue.get('file')}")
            print(f"     Line    : {issue.get('line')}")
            print(f"     Impact  : {issue.get('impact')}")
            print(f"     Issue   : {issue.get('message')}")
            print(f"     Fix     : {issue.get('suggestion')}")
        print(f"\n{'='*60}\n")