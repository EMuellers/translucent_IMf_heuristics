import networkx as nx
from networkx import MultiDiGraph
from automata.fa.dfa import DFA
from automata.fa.nfa import NFA
import copy


def find_final_nodes(graph: MultiDiGraph):
    nodes_numbers_of_edges = {}
    nodes_to_work_on = set()
    nodes_to_work_on.add(1)
    while len(nodes_to_work_on)>0:
        current_node = nodes_to_work_on.pop()
        if current_node not in nodes_numbers_of_edges:
            nodes_numbers_of_edges[current_node] = 0
            edges_of_node = list(graph.in_edges(current_node))
            for el in edges_of_node:
                temp = graph.edges(el[0], el[1], keys=True)
                for edge in temp:
                    if edge[0] != current_node and edge[1] == current_node:
                        nodes_numbers_of_edges[current_node] += 1
                        if graph.edges[edge[0], edge[1], edge[2]]["transition"].label is None:
                            nodes_numbers_of_edges[current_node] -= 1
                            nodes_to_work_on.add(edge[0])
    target_nodes = set()
    for node in nodes_numbers_of_edges:
        if nodes_numbers_of_edges[node] > 0:
            target_nodes.add(node)
    return target_nodes


def remove_tau_add_successor_edges(graph: MultiDiGraph) -> MultiDiGraph:
    still_tau = True
    edges_to_consider = list(graph.edges)
    while still_tau:
        still_tau = False
        for edge in edges_to_consider:
            if graph.edges[edge[0], edge[1], edge[2]]["transition"].label is None:
                edges_to_add = set()
                still_tau = True
                successors = list(graph.successors(edge[1]))
                for successor in successors:
                    temp = graph.edges(edge[1], successor, keys=True)
                    for edges_between_target_and_successor in temp:
                        if edges_between_target_and_successor[1] == successor:
                            edges_to_add.add((edge[0],
                                             successor,
                                             graph.edges[edge[1], successor, edges_between_target_and_successor[2]]["transition"],
                                            graph.edges[edge[1], successor, edges_between_target_and_successor[2]][
                                               "weight"] + 1))
                for source, target, transition, weight in edges_to_add:
                    graph.add_edge(source,target, transition=transition, weight=weight)
                graph.remove_edge(edge[0], edge[1], edge[2])
                edges_to_consider = list(graph.edges)
    return graph


def construct_dfa(graph: MultiDiGraph, reachability_graph: MultiDiGraph):
    """
    :param graph:
    :return: dfa
    """

    final_nodes = find_final_nodes(reachability_graph)

    initial_state = 0
    input_symbols = set()
    transitions = {}
    for edge in graph.edges:
        transition = graph.edges[edge[0], edge[1], edge[2]]["transition"].name
        if transition not in input_symbols:
            input_symbols.add(transition)
        if edge[0] not in transitions:
            transitions[edge[0]] = {}
        if transition not in transitions[edge[0]]:
            transitions[edge[0]][transition] = set()
        transitions[edge[0]][transition].add(edge[1])
    nodes_in_drg = set(list(graph.nodes))
    for node in final_nodes:
        if node not in nodes_in_drg:
            final_nodes.remove(node)
    nfa = NFA(states=nodes_in_drg,
              input_symbols=input_symbols,
              transitions=transitions,
              initial_state=initial_state,
              final_states=final_nodes)
    dfa = DFA.from_nfa(nfa, retain_names=True)
    resulting_graph = nx.MultiDiGraph()
    resulting_graph.add_nodes_from(dfa.states)
    for start_state in dfa.transitions:
        for transition in dfa.transitions[start_state]:
            resulting_graph.add_edge(start_state, dfa.transitions[start_state][transition], transition=transition)
    #for start_state, transition in dfa.transitions.items():
    #    for end_state in transition.values():
    #        name = get_key_from_value(transition, end_state)
    #        resulting_graph.add_edge(start_state, end_state, transition=name)
    return resulting_graph


def remove_nodes_with_no_ingoing_arcs(graph: MultiDiGraph) -> MultiDiGraph:
    # Identify nodes with no incoming edges
    nodes_to_remove = [node for node, in_degree in graph.in_degree() if in_degree == 0 and node != 0]
    # Remove nodes with no incoming edges
    graph.remove_nodes_from(nodes_to_remove)
    return graph


def add_enabled_activities_to_nodes_drg(graph: MultiDiGraph) -> MultiDiGraph:
    nodes = graph.nodes()
    for node in nodes:
        activities = set()
        successors = list(graph.successors(node))
        for successor in successors:
            edges_between_nodes = graph.edges(node, successor, keys=True)
            edges_between_nodes = [element for element in edges_between_nodes if element[1] == successor]
            for edge in edges_between_nodes:
                transition = graph.edges[edge[:3]]['transition']
                activities.add(transition.label)
        graph.nodes[node]["enabled_activities"] = activities
    return graph


def add_enabled_activities_to_nodes_dfa(dfa: MultiDiGraph, net) -> MultiDiGraph:
    def get_label(transition_name):
        for el in net.transitions:
            if el.name == transition_name:
                return el.label

    nodes = dfa.nodes()
    for node in nodes:
        activities = set()
        successors = list(dfa.successors(node))
        for successor in successors:
            edges_between_nodes = dfa.edges(node, successor, keys=True)
            edges_between_nodes = [element for element in edges_between_nodes if element[1] == successor]
            for edge in edges_between_nodes:
                transition = dfa.edges[edge[:3]]['transition']
                activities.add(get_label(transition))
        dfa.nodes[node]["enabled_activities"] = activities
    return dfa


class DirectReachabilityGraph:
    def __init__(self, r_g):
        self.reachability_graph = r_g.reachability_graph
        self.direct_reachability_graph = self.transform_rg_to_drg()
        self.dfa = add_enabled_activities_to_nodes_dfa(construct_dfa(self.direct_reachability_graph, self.reachability_graph), r_g.net)

    def transform_rg_to_drg(self):
        temp = copy.deepcopy(self.reachability_graph)
        temp = remove_tau_add_successor_edges(temp)
        temp = remove_nodes_with_no_ingoing_arcs(temp)
        return add_enabled_activities_to_nodes_drg(temp)
