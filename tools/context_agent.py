import os
from dotenv import load_dotenv
from agents.agent import Agent
from agents.run import Runner
from agents.mcp import MCPServerStdio
from agents.mcp.server import MCPServerStdioParams

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://codeguard:codeguard123@localhost:5432/codeguard_db"
)

CONTEXT_INSTRUCTIONS = """You are CodeGuard's Context Agent.

You query the CodeGuard PostgreSQL database to find historical patterns
that enrich the current code review.

Use the PostgreSQL MCP tools to run queries like:

1. Find repeat offenders:
   SELECT category, COUNT(*) as count
   FROM issues
   WHERE repo_name = '{repo}' AND created_at > NOW() - INTERVAL '30 days'
   GROUP BY category ORDER BY count DESC LIMIT 5;

2. Find team patterns:
   SELECT category, severity, COUNT(*) as count
   FROM issues
   WHERE repo_name = '{repo}' AND pr_author = '{author}'
   GROUP BY category, severity ORDER BY count DESC LIMIT 5;

3. Check if auto-fix worked last time:
   SELECT status, COUNT(*) FROM autofix_results
   WHERE repo_name = '{repo}' GROUP BY status;

If tables do not exist return: No historical data yet.
Return a 3-4 sentence summary of patterns found.
"""


async def run_context_lookup(
    repo_name : str,
    pr_author : str,
    pr_number : int
) -> dict:
    print(f"Context Agent querying historical data...")

    prompt = f"""Query CodeGuard database for PR #{pr_number} in {repo_name} by {pr_author}.
Run all queries replacing repo='{repo_name}' and author='{pr_author}'.
Return a 3-4 sentence summary. If no data exists, say so clearly.
"""

    postgres_mcp = MCPServerStdio(
        params=MCPServerStdioParams(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-postgres",
                DATABASE_URL
            ]
        ),
        name="postgresql-mcp"
    )

    try:
        async with postgres_mcp:
            agent = Agent(
                name         = "ContextAgent",
                instructions = CONTEXT_INSTRUCTIONS,
                model        = "gpt-4o",
                tools        = [],
                mcp_servers  = [postgres_mcp]
            )
            result = await Runner.run(agent, prompt)

        print(f"   Context complete: {result.final_output[:100]}...")
        return {"status": "completed", "context": result.final_output}
    except Exception as e:
        print(f"   Context skipped: {e}")
        return {"status": "skipped", "context": "No historical context available yet."}