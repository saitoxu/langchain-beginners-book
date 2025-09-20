from langgraph.graph import END, StateGraph

from scripts.chapter9.nodes import answering_node, check_node, selection_node
from scripts.chapter9.state import State

workflow = StateGraph(State)
workflow.add_node("selection", selection_node)
workflow.add_node("answering", answering_node)
workflow.add_node("check", check_node)

workflow.set_entry_point("selection")

workflow.add_edge("selection", "answering")
workflow.add_edge("answering", "check")

workflow.add_conditional_edges(
    "check",
    lambda state: state.current_judge,
    {True: END, False: "selection"},
)

compiled = workflow.compile()

initial_state = State(query="生成AIについて教えてください。")
result = compiled.invoke(initial_state)
print(result)

# compiled.get_graph().draw_png("chapter9_graph.png")
