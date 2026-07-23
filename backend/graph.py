from typing import TypedDict
from langgraph.graph import StateGraph, END

from backend.rag import retrieve_context
from backend.granite import ask_granite


class GraphState(TypedDict):
    question: str
    context: str
    sources: list
    answer: str
def retrieve_node(state: GraphState):

    retrieved = retrieve_context(state["question"])

    return {
        "context": retrieved["context"],
        "sources": retrieved["sources"]
    }
def generate_node(state: GraphState):

    response = ask_granite(
    state["question"],
    state["context"],
    state["sources"]
)
    return {
        "answer": response["answer"],
        "sources": response["sources"]
    }
graph = StateGraph(GraphState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()