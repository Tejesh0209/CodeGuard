import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from rag.retriever import CodeRetriever

load_dotenv()


class SecurityIssue(BaseModel):
    file        : str  = Field(description="filename where issue was found")
    line        : int  = Field(description="line number of the issue")
    severity    : str  = Field(description="LOW, MEDIUM, HIGH, or CRITICAL")
    category    : str  = Field(description="injection, auth, crypto, exposure, misconfiguration, or insecure_design")
    cwe_id      : str  = Field(description="CWE ID e.g. CWE-89 for SQL injection")
    message     : str  = Field(description="clear description of the vulnerability")
    suggestion  : str  = Field(description="exactly how to fix it")
    exploitable : bool = Field(description="true if easily exploitable")


class SecurityReview(BaseModel):
    pr_summary   : str  = Field(description="one sentence security summary")
    issues       : List[SecurityIssue] = Field(description="all security issues found")
    risk_score   : int  = Field(description="overall risk score 1-10")
    has_critical : bool = Field(description="true if any CRITICAL issues found")
    approved     : bool = Field(description="true if PR passes security review")


SECURITY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are CodeGuard's Security Agent — an expert in application security.

You analyze code diffs for security vulnerabilities using OWASP Top 10:
- A03 Injection (SQL, NoSQL, OS, LDAP)
- A02 Cryptographic Failures
- A07 Authentication Failures
- A01 Broken Access Control
- A05 Security Misconfiguration

Additional patterns:
- Hardcoded secrets, passwords, API keys
- Missing input validation
- Path traversal vulnerabilities
- Insecure deserialization

Severity:
- LOW      : minor security concern
- MEDIUM   : should fix before merge
- HIGH     : must fix, significant risk
- CRITICAL : blocks merge immediately, easily exploitable

{format_instructions}"""
    ),
    (
        "human",
        """Analyze this pull request for security vulnerabilities.

{team_context}

PR Diff:
{diff}

Return your security review in the exact JSON format specified."""
    )
])


class SecurityAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.parser    = JsonOutputParser(pydantic_object=SecurityReview)
        self.chain     = SECURITY_PROMPT | self.llm | self.parser
        self.retriever = CodeRetriever()

    async def review(self, diff_chunks: list[dict]) -> dict:
        formatted_diff = self._format_diff(diff_chunks)

        print(f"\n🔒 Security Agent retrieving context...")
        similar_chunks = self.retriever.retrieve(
            query=formatted_diff[:1000],
            repo="Tejesh0209/SentinelAI",
            top_k=3
        )
        team_context = self.retriever.format_for_prompt(similar_chunks)

        print(f"🔒 Security Agent analyzing vulnerabilities...")
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
        print(f"🔒 SECURITY REVIEW COMPLETE")
        print(f"{'='*60}")
        print(f"📋 Summary      : {review.get('pr_summary', 'N/A')}")
        print(f"🎯 Risk Score   : {review.get('risk_score', 0)}/10")
        print(f"🚨 Has Critical : {review.get('has_critical', False)}")
        print(f"✅ Approved     : {review.get('approved', False)}")
        issues = review.get('issues', [])
        print(f"\n🔍 Vulnerabilities Found: {len(issues)}")
        for i, issue in enumerate(issues, 1):
            severity_emoji = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(issue.get('severity', 'LOW'), "⚪")
            print(f"\n  {i}. {severity_emoji} [{issue.get('severity')}] {issue.get('category', '').upper()} — {issue.get('cwe_id')}")
            print(f"     File        : {issue.get('file')}")
            print(f"     Line        : {issue.get('line')}")
            print(f"     Exploitable : {issue.get('exploitable')}")
            print(f"     Issue       : {issue.get('message')}")
            print(f"     Fix         : {issue.get('suggestion')}")
        print(f"\n{'='*60}\n")
