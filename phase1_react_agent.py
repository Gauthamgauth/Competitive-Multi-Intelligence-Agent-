"""
Phase 1: Single ReAct agent with web search tool.
Goal: prove the agent can reason -> search -> observe -> answer, end to end.
No multi-agent complexity yet. Just one agent, one tool, one question.
"""

"""
Phase 1: Single ReAct agent with web search tool.
Goal: prove the agent can reason -> search -> observe -> answer, end to end.
No multi-agent complexity yet. Just one agent, one tool, one question.
"""

import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

load_dotenv()  # reads OPENAI_API_KEY / TAVILY_API_KEY from a .env file

#  tool 
search_tool = TavilySearch(max_results=5)

#  model 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# binds the model + tool(s) into a reasoning loop 
agent = create_react_agent(llm, tools=[search_tool])

#
if __name__ == "__main__":
    question = (
        "compare slacks and microsoft team's pricing for small teams  ."
    )

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # Print only the final answer (last message in the trace)
    final_message = result["messages"][-1]
    print("\n--- FINAL ANSWER ---\n")
    print(final_message.content)

 