import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

class StyleIssue(BaseModel):
    file: str = Field(description="filename where issue was found")
    line: int = Field(description="line number of the issue")
    severity: str = Field(description="LOW, MEDIUM, or HIGH")
    category: str = Field(description="naming, formatting, complexity, documentation, or best_practice")
    message: str = Field(description="clear description of the issue")
    suggestion: str = Field(description="exactly how to fix it")

class StyleReview(BaseModel):
    pr_summary: str = Field(description="one sentence summary of what this PR does")
    issues: List[StyleIssue] = Field(description="list of all style issues found")
    overall_score: int = Field(description="code quality score from 1-10")
    approved: bool = Field(description="true if PR meets style standards")

STYLE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are CodeGuard's Style Agent — a senior software engineer 
doing a thorough code style review. You review code diffs and identify 
style, formatting, naming, complexity, and documentation issues.

You are strict but fair. You only flag real issues, not personal preferences.

Categories you review:
- naming: variable/function/class names that are unclear or inconsistent
- formatting: indentation, line length, whitespace issues  
- complexity: functions too long, too many nested blocks, high cyclomatic complexity
- documentation: missing docstrings, unclear comments, no type hints
- best_practice: anti-patterns, poor error handling, code duplication

Severity levels:
- LOW: minor issue, nice to fix
- MEDIUM: should be fixed before merge
- HIGH: must be fixed, blocks merge

{format_instructions}"""
    ),
    (
        "human",
        """Review this pull request diff and return a structured style review.

PR Diff:
{diff}

Return your review in the exact JSON format specified."""
    )
])

class StyleAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.parser = JsonOutputParser(pydantic_object=StyleReview)
        self.chain = STYLE_PROMPT | self.llm | self.parser

    async def review(self, diff_chunks: list[dict]) -> dict:
        formatted_diff = self._format_diff(diff_chunks)
        print(f"\n🎨 Style Agent starting review...")
        print(f"   Analyzing {len(diff_chunks)} file(s)...")
        result = await self.chain.ainvoke({
            "diff": formatted_diff,
            "format_instructions": self.parser.get_format_instructions()
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
        print(f"🎨 STYLE REVIEW COMPLETE")
        print(f"{'='*60}")
        print(f"📋 Summary : {review.get('pr_summary', 'N/A')}")
        print(f"⭐ Score   : {review.get('overall_score', 0)}/10")
        print(f"✅ Approved: {review.get('approved', False)}")
        issues = review.get('issues', [])
        print(f"\n🔍 Issues Found: {len(issues)}")
        for i, issue in enumerate(issues, 1):
            severity_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(issue.get('severity', 'LOW'), "⚪")
            print(f"\n  {i}. {severity_emoji} [{issue.get('severity')}] {issue.get('category', '').upper()}")
            print(f"     File      : {issue.get('file')}")
            print(f"     Line      : {issue.get('line')}")
            print(f"     Issue     : {issue.get('message')}")
            print(f"     Fix       : {issue.get('suggestion')}")
        print(f"\n{'='*60}\n")
