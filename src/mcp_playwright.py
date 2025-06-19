from dotenv import load_dotenv

load_dotenv()  # .envファイルから環境変数を読み込む

import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
client = MultiServerMCPClient(
    {
        "playwright": {
            "command": "npx",
            "args": ["@playwright/mcp@latest"],
            "transport": "stdio",
        }
    }
    # # またはこちら
    # # npx @playwright/mcp@latest --port 8931
    # {
    #     "playwright": {
    #         "url": "http://localhost:8931/mcp",
    #         "transport": "streamable_http",
    #     }
    # }
)


class Report(BaseModel):
    summary: list[str] = Field(description="概要")


async def main():
    # 通常
    # agent = create_react_agent(llm, await client.get_tools())
    # 構造化出力例
    agent = create_react_agent(llm, await client.get_tools(), response_format=Report)
    response = await agent.ainvoke(
        {
            "messages": "playwrightを使って https://github.com/birdwatcherYT/llm-learn にアクセスして概要を教えて"
        }
    )
    # print(response)
    print(response["messages"][-1].content)
    print(response["structured_response"])


if __name__ == "__main__":
    asyncio.run(main())
