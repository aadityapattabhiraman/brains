#!/home/akugyo/Programs/Python/chatbots/bin/python

import os
from langgraph.graph import MessagesState, StateGraph, END, START
from langchain_openai import AzureChatOpenAI



model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)


class State(MessagesState):

    current_route: str


def routing(state: State) -> str:

    if "current_route" in state and state["current_route"]:
        return state["current_route"]

    else:
        return greeting_agent


workflow = StateGraph(State)
workflow.add_conditional_edge(START, routing)
