#!/home/akugyo/Programs/Python/chatbots/bin/python

import random
from typing import Literal
from langgraph.graph import Graph, START, END



def weather(str):

    return "Hi! Well.. I have no idea... But... "


def rainy_weather(str):

    return str + "Its going to rain today. Carry an umbrealla."


def sunny_weather(str):

    return str + "Its going to be sunny today. Wear sunscreen."


def forecast_weather(str) -> Literal["rainy", "sunny"]:

    if random.random() < 0.5:
        return "rainy"

    else:
        return "sunny"


workflow = Graph()

workflow.add_node("weather", weather)
workflow.add_node("rainy", rainy_weather)
workflow.add_node("sunny", sunny_weather)

workflow.add_edge(START, "weather")
workflow.add_conditional_edges("weather", forecast_weather)
workflow.add_edge("rainy", END)
workflow.add_edge("sunny", END)

graph = workflow.compile()

response = graph.invoke("hello")
print(response)
