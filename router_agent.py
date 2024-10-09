from dotenv import load_dotenv
from typing import Dict, Any
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents import create_csv_agent


load_dotenv()


def get_python_agent(
    prompt: ChatPromptTemplate,
) -> AgentExecutor:
    """Simple creator of Python Interpreter Agent"""

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    prompt = prompt.partial(instructions=instructions)
    pytools = [PythonREPLTool()]
    pyagent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=pytools,
    )

    pyagent_executor = AgentExecutor(
        agent=pyagent, tools=pytools, verbose=True
    )

    return pyagent_executor


def get_csv_agent(csv_path: str) -> AgentExecutor:
    """Simple creator of CSV Agent executor"""
    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        path=csv_path,
        verbose=True,
        allow_dangerous_code=True,
    )

    return csv_agent


def call_router(
    prompt: str,
    base_prompt_url: str = "langchain-ai/react-agent-template",
    csv_path: str = "episode_info.csv",
) -> Dict[str, Any]:
    """Implements simple Python-CSV ReAct Agent for demo"""
    base_prompt = hub.pull(base_prompt_url)

    python_agent = get_python_agent(base_prompt)

    # Wrapper for enforcing input keys                      !!!!

    def python_agent_wrapper(prompt: str) -> Dict[str, Any]:
        return python_agent.invoke({'input': prompt})

    csv_agent = get_csv_agent(csv_path)

    # router tools

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_wrapper,
            description="""
            usefull when you need to transform natural
            language to python and execute python code.
            DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent.invoke,
            description="""
            useful when you need to answer questions over
            the csv file. takes the entire question as input
            and returns answer using pandas""",
        ),
    ]

    # Create prompt and collect super-agent

    react_prompt = base_prompt.partial(instructions="")

    router = create_react_agent(
        prompt=react_prompt,
        llm=ChatOpenAI(
            temperature=0, model="gpt-4-turbo"
        ),
        tools=tools,
    )

    router_executor = AgentExecutor(
        agent=router, tools=tools, verbose=True
    )

    # invoke router

    response = router_executor.invoke(
        input={"input": prompt}
    )
    return response


if __name__ == "__main__":
    # prompt = input()
    prompt = 'Find out who produced the biggest number of movies and store the name in the text file in current working directory'
    # prompt = "How many rows are there?"
    res = call_router(prompt)
    print(res.get('output'))
