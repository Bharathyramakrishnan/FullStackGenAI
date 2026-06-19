import openai
from dotenv import load_dotenv
import os
from phi.agent import Agent
from phi.model.groq import Groq 
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
import phi
from phi.playground import Playground, serve_playground_app

phi.api_key=os.getenv("PHI_API_KEY")

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
app = Playground(agents=[web_search_tool,finance_agent]).get_app()
  

if __name__ == "__main__":
    serve_playground_app("playground:app",reload=True)