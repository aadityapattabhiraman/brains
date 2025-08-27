#!/home/akugyo/Programs/Python/chatbots/bin/python

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict



class State(TypedDict):

    count: int
    message: str



def counter(state: State):

    state["count"] += 1
    state["message"] = f"Counter function has been called {state["count"]} time(s)"

    return state


workflow = StateGraph(State)

workflow.add_node("Node1", counter)
workflow.add_node("Node2", counter)
workflow.add_node("Node3", counter)

workflow.add_edge(START, "Node1")
workflow.add_edge("Node1", "Node2")
workflow.add_edge("Node2", "Node3")
workflow.add_edge("Node3", END)

graph = workflow.compile()

response = graph.invoke({"count": 0, "message": "Hello boss"})
print(response)
