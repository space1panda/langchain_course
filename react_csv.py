from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_csv_agent


load_dotenv()


def main(path: str, prompt: str) -> None:
    """Runs ReAct Agent for CSV file analysis"""
    print("Start CSV analysis...")
    agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        path=path,
        verbose=True,
        allow_dangerous_code=True,
    )
    response = agent.invoke(input={"input": prompt})
    return response


if __name__ == "__main__":
    res = main("episode_info.csv", input())
    print(res.get("output"))
