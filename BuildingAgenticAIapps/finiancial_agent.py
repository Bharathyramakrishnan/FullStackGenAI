from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

web_search_tool = Agent(
    name="Web Search",
    tools=[DuckDuckGo()],
    model=Groq(id="llama-3.3-70b-versatile"),
    role="Search the web for the information",
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

finance_agent = Agent(
    name="Financial Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True,company_info=True)],
    role="Answer financial questions and provide stock information",
    instructions=["use tables to display the data"],
    show_tool_calls=True,   
    markdown=True
)

multi_ai_Agent= Agent(
    name="Multi AI Agent",
    team=[web_search_tool, finance_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=["Always include sources","use tables to display the data"],
    markdown=True,
    show_tool_calls=True,
)

multi_ai_Agent.print_response("Get analyst recommendations for NVDA",stream=True)