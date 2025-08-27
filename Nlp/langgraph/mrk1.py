#!/home/akugyo/Programs/Python/chatbots/bin/python

from langgraph.graph import Graph



def node1(str):

    return str + " I reached Node1."


def node2(str):

    return str + " Now at Node2."


workflow = Graph()

workflow.add_node("node_1", node1)
workflow.add_node("node_2", node2)

workflow.add_edge("node_1", "node_2")
workflow.set_entry_point("node_1")
workflow.set_finish_point("node_2")

graph = workflow.compile()

response = graph.invoke("hello")
print(response)
