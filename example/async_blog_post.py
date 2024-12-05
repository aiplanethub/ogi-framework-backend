import os
import asyncio
from openagi.actions.files import WriteFileAction
from openagi.actions.tools.ddg_search import DuckDuckGoSearch
from openagi.actions.tools.webloader import WebBaseContextTool
from openagi.agent import Admin
from openagi.memory import Memory
from openagi.planner.task_decomposer import TaskPlanner
from openagi.llms.openai import OpenAIModel, OpenAIConfigModel
from openagi.worker import Worker
from rich.console import Console
from rich.markdown import Markdown


async def main():
    # Load configuration and initialize LLM
    config = OpenAIConfigModel(**{ 
        "openai_api_key": "",
        "model_name": "gpt-4o"
    })
    llm = OpenAIModel(config=config)

    # Define workers
    researcher = Worker(
        role="Research Analyst",
        instructions="""Identify and analyze current AI and data science trends. Use DuckDuckGoNewsSearch for latest news and WebBaseContextTool for deep dives. Assess impact, prioritize findings, and compile a concise report on top 3-5 trends with actionable insights.""",
        actions=[DuckDuckGoSearch, WebBaseContextTool],
    )

    writer = Worker(
        role="Tech Content Strategist",
        instructions="""Craft a compelling article on tech advancements based on the research brief. Choose a unique angle, explain key points with examples, incorporate expert insights, and optimize for engagement. Deliver a complete, narrative-driven article.""",
        actions=[DuckDuckGoSearch, WebBaseContextTool],
    )

    reviewer = Worker(
        role="Review and Editing Specialist",
        instructions="""Review and refine the article for clarity, engagement, and accuracy. Check grammar, facts, and SEO. Ensure alignment with company values. Write the final version using WriteFileAction. Provide file path and a summary of changes.""",
        actions=[DuckDuckGoSearch, WebBaseContextTool, WriteFileAction],
    )

    # Team Manager/Admin
    admin = Admin(
        planner=TaskPlanner(human_intervene=False),
        memory=Memory(),
        llm=llm,
    )
    admin.assign_workers([researcher, writer, reviewer])

    # Run asynchronously
    res = await admin.async_run(
        query="Write a blog post about the future of AI.",
        description="""Create an engaging blog post on AI's future based on 2024 advancements. Research breakthroughs, analyze impacts, and write a post highlighting 3-5 key advancements. Explain their importance, applications, and ethical considerations. Ensure the post is informative, accessible, and exciting for a tech-savvy audience. Save the final post to a file and return the file path with a brief summary."""
    )

    # Print the results from the OpenAGI
    print("-" * 100)  # Separator
    Console().print(Markdown(res))


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())