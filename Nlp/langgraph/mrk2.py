#!/home/akugyo/Programs/Python/chatbots/bin/python

import os
from langgraph.graph import Graph, START, END
from langchain_openai import AzureChatOpenAI



model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)


def call_llm(message):

    return model.invoke(message)


workflow = Graph()
workflow.add_node("llm", call_llm)

workflow.add_edge(START, "llm")
workflow.add_edge("llm", END)

graph = workflow.compile()

response = graph.invoke("hello")
print(response)
