import asyncio

from claude_agent_sdk import query, ClaudeAgentOptions

schema = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "founded_year": {"type": "number"},
        "headquarters": {"type": "string"}
    },
    "required": ["company_name"]
}

async def test():
    async for message in query(
        prompt="Research Anthropic and provide key company information",
        options=ClaudeAgentOptions(
            # allowed_tools=["Bash"],
            output_format={
                "type": "json_schema",
                "schema": schema
                }
            ) 
    ):
        print(message)
        if hasattr(message, 'structured_output'):
            print("Structured Output:", message.structured_output)
            # {'company_name': 'Anthropic', 'founded_year': 2021, 'headquarters': 'San Francisco, CA'}

asyncio.run(test())
