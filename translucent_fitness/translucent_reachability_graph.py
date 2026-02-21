from typing import Optional

import networkx as nx
#from bs4 import BeautifulSoup
from pm4py import PetriNet, Marking
from pm4py.algo.analysis.workflow_net.algorithm import apply as is_workflow_net
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.semantics import enabled_transitions, execute
from pm4py.objects.petri_net.utils.align_utils import SKIP
from pm4py.util.typing import AlignmentResult
#from pyvis.network import Network

from translucent_fitness.utils import add_artificial_end_transition, ARTIFICIAL_END_TRANSITION_NAME, ARTIFICIAL_END_TRANSITION_LABEL


class TranslucentReachabilityGraph(nx.MultiDiGraph):
    def __init__(self, accepting_petri_net: tuple[PetriNet, Marking, Marking]):
        if not is_workflow_net(accepting_petri_net[0]):
            raise ValueError("The Petri net is not a workflow net")

        accepting_petri_net = add_artificial_end_transition(accepting_petri_net)
        super().__init__()
        self.transition_labels = {transition.name: transition.label for transition in accepting_petri_net[0].transitions}
        self.marking_map = {accepting_petri_net[1]: 0, accepting_petri_net[2]: 1}
        self.initial_state = 0
        self.final_state = 1
        self.add_node(0, marking=accepting_petri_net[1], enabled=set())
        self.add_node(1, marking=accepting_petri_net[2], enabled=set())
        open_list = [0]

        def get_arcs_from_enabled_transitions(petri_net: PetriNet,
                                              marking: Marking,
                                              visited_states: set[Marking],
                                              firing_sequence: Optional[tuple[str, ...]] = None,
                                              ) -> set[tuple[Marking, Marking, tuple[str, ...], str]]:
            if firing_sequence is None:
                firing_sequence = tuple()
            visited_states.add(marking)
            arcs = set()
            for transition in enabled_transitions(petri_net, marking):
                next_marking = execute(transition, petri_net, marking)
                if transition.label is not None:
                    arcs.add((marking, next_marking, firing_sequence + (transition.name,), transition.label))
                elif next_marking not in visited_states:
                    arcs.update(get_arcs_from_enabled_transitions(petri_net, next_marking, visited_states,
                                                                  firing_sequence=firing_sequence + (transition.name,)))
            return arcs

        while open_list:
            current_node = open_list.pop()
            current_marking = self.nodes[current_node]['marking']
            for arc in get_arcs_from_enabled_transitions(accepting_petri_net[0], current_marking, set()):
                post_marking = arc[1]
                if post_marking not in self.marking_map:
                    self.marking_map[post_marking] = self.number_of_nodes()
                    self.add_node(self.marking_map[post_marking], marking=post_marking, enabled=set())
                    open_list.append(self.marking_map[post_marking])
                if arc[3] is not ARTIFICIAL_END_TRANSITION_LABEL:
                    self.nodes[current_node]['enabled'].add(arc[3])
                self.add_edge(current_node, self.marking_map[post_marking], firing_sequence=arc[2], label=arc[3],
                              cost=0 if arc[2][-1] == ARTIFICIAL_END_TRANSITION_NAME else 1)
        self.best_worst_cost = nx.dijkstra_path_length(self, self.initial_state, self.final_state, weight='cost')
    """
    def view(self) -> None:
        net = Network(width='100%', height='100%', directed=True)
        for node in self.nodes:
            net.add_node(node, label=str(node), size=10, title=str(self.nodes.get(node)['enabled']),
                         color='green' if node == self.initial_state else 'red' if node == self.final_state else 'blue')
        for edge in self.edges:
            net.add_edge(edge[0], edge[1], label=self.edges.get(edge)['label'], title=str(self.edges.get(edge)['firing_sequence']))
        net.show('./output/trg.html')

        # Open the HTML file and parse it with BeautifulSoup
        with open('./output/trg.html', 'r+') as f:
            soup = BeautifulSoup(f, 'html.parser')

            soup.div['style'] = 'height: 100%; width: 100%;'

            # Write the modified HTML back to the file
            f.seek(0)
            f.write(str(soup))
            f.truncate()
    """

def tversky_index(set1: set, set2: set, alpha: float = 1, beta: float = 1) -> float:
    return len(set1.intersection(set2)) / (len(set1.intersection(set2)) + alpha * len(set1.difference(set2)) + beta * len(set2.difference(set1)))


class TranslucentAlignmentStateGraph(nx.MultiDiGraph):
    def __init__(self, translucent_reachability_graph: TranslucentReachabilityGraph, trace: Trace):
        super().__init__()
        self.trace = trace
        self.best_worst_cost = len(trace) + translucent_reachability_graph.best_worst_cost
        self.initial_state = (0, 0)
        self.final_state = (len(trace), 1)
        self.transition_labels = translucent_reachability_graph.transition_labels

        def enabled_set_cost(enabled_set_trace: set[str], enabled_set_model: set[str]) -> float:
            return 1 - tversky_index(enabled_set_trace, enabled_set_model)

        # Add arcs corresponding to moves on model
        for idx in range(len(trace) + 1):
            for node in translucent_reachability_graph.nodes:
                self.add_node((idx, node), marking=translucent_reachability_graph.nodes[node]['marking'], enabled=translucent_reachability_graph.nodes[node]['enabled'])
            for edge in translucent_reachability_graph.edges:
                self.add_edge((idx, edge[0]), (idx, edge[1]),
                              firing_sequence=translucent_reachability_graph.edges[edge]['firing_sequence'],
                              label=None,
                              cost=translucent_reachability_graph.edges[edge]['cost'],
                              classical_cost=translucent_reachability_graph.edges[edge]['cost'],
                              type='model')
        # Add arcs corresponding to moves on log
        for idx in range(len(trace)):
            for node in translucent_reachability_graph.nodes:
                self.add_edge((idx, node), (idx + 1, node),
                              firing_sequence=(),
                              label=trace[idx].get('concept:name'),
                              cost=1,
                              classical_cost=1,
                              type='log')
        # Add arcs corresponding to synchronous moves
        for idx in range(len(trace)):
            for edge in translucent_reachability_graph.edges(data=True):
                if edge[2]['label'] == trace[idx].get('concept:name'):
                    self.add_edge((idx, edge[0]), (idx + 1, edge[1]),
                                  firing_sequence=edge[2]['firing_sequence'],
                                  label=trace[idx].get('concept:name'),
                                  cost=enabled_set_cost(trace[idx].get('enabled'), translucent_reachability_graph.nodes[edge[0]]['enabled']),
                                  classical_cost=0,
                                  type='sync')
        # Add arcs corresponding to execution change moves
                elif edge[2]['label'] != ARTIFICIAL_END_TRANSITION_LABEL:
                    self.add_edge((idx, edge[0]), (idx + 1, edge[1]),
                                  firing_sequence=edge[2]['firing_sequence'],
                                  label=trace[idx].get('concept:name'),
                                  cost=1+enabled_set_cost(trace[idx].get('enabled'), translucent_reachability_graph.nodes[edge[0]]['enabled']),
                                  classical_cost=3,
                                  type='change')
    """
    def view(self) -> None:
        net = Network(width='100%', height='100%', directed=True)
        for node in self.nodes(data=True):
            net.add_node(str(node[0]), size=10,
                         label=str(node[0]),
                         title=str(node[1]['enabled']),
                         color='green' if node[0] == self.initial_state else 'red' if node[0] == self.final_state else 'blue')
        for edge in self.edges(data=True):
            net.add_edge(str(edge[0]), str(edge[1]),
                         label=f"({edge[2]['label']}, {self.transition_labels[firing_sequence[-1]] if (firing_sequence := edge[2]['firing_sequence']) else '>>'})",
                         title=f"{edge[2]['firing_sequence']}: {edge[2]['cost']}",
                         color='green' if edge[2]['type'] == 'sync' else 'black' if edge[2]['type'] == 'log' else 'orange' if edge[2]['type'] == 'change' else 'blue')
        net.show('./output/tasg.html')
        
        # Open the HTML file and parse it with BeautifulSoup
        with open('./output/tasg.html', 'r+') as f:
            soup = BeautifulSoup(f, 'html.parser')

            soup.div['style'] = 'height: 100%; width: 100%;'

            # Write the modified HTML back to the file
            f.seek(0)
            f.write(str(soup))
            f.truncate()
    """
    def get_optimal_alignment_cost(self, ignore_translucent: bool = False) -> float:
        return nx.dijkstra_path_length(self, self.initial_state, self.final_state, weight='classical_cost' if ignore_translucent else 'cost')

    def get_optimal_alignment(self, ignore_translucent: bool = False) -> AlignmentResult:
        alignment = []
        translucent_alignment = []
        cost = 0
        move_cost = []
        trace_idx = 0
        n_sync, n_log, n_model, n_silent, n_enabled_change, n_execution_change, n_execution_enabled_change = 0, 0, 0, 0, 0, 0, 0
        for u, v in nx.utils.pairwise(nx.dijkstra_path(self, self.initial_state, self.final_state, weight='classical_cost' if ignore_translucent else 'cost')):
            edge = (u, v, min(self[u][v], key=lambda k: self[u][v][k].get('classical_cost' if ignore_translucent else 'cost', 1)))
            edge_data = self.edges[edge]
            if (firing_sequence := edge_data.get('firing_sequence')) and firing_sequence[-1] == ARTIFICIAL_END_TRANSITION_NAME:
                continue
            # Add silent moves to the alignment
            alignment.extend([(SKIP, None)] * (len(firing_sequence) - 1))
            translucent_alignment.extend([(SKIP, None)] * (len(firing_sequence) - 1))
            move_cost.extend([0] * (len(firing_sequence) - 1))
            n_silent += len(firing_sequence) - 1 if len(firing_sequence) > 1 else 0
            # Add other moves to the alignment
            alignment.append((label if (label := edge_data.get('label')) else SKIP, self.transition_labels[firing_sequence[-1]] if firing_sequence else SKIP))
            translucent_alignment.append(((label if label else SKIP, self.trace[trace_idx]['enabled'] if label else set()), (self.transition_labels[firing_sequence[-1]] if firing_sequence else SKIP, self.nodes[u]['enabled'])))
            if label:
                if firing_sequence:
                    # Synchronous move, enabled change move, execution change move, execution enabled change move
                    if self.transition_labels[firing_sequence[-1]] == label:
                        # Synchronous move, enabled change move
                        if self.trace[trace_idx]['enabled'] == self.nodes[u]['enabled']:
                            # Synchronous move
                            n_sync += 1
                        else:
                            # Enabled change move
                            n_enabled_change += 1
                    else:
                        # Execution change move, execution enabled change move
                        if self.trace[trace_idx]['enabled'] == self.nodes[u]['enabled']:
                            # Execution change move
                            n_execution_change += 1
                        else:
                            # Execution enabled change move
                            n_execution_enabled_change += 1
                else:
                    # Log move
                    n_log += 1
                trace_idx += 1
            else:
                # Model move
                n_model += 1
            cost += edge_data.get('classical_cost' if ignore_translucent else 'cost')
            move_cost.append(edge_data.get('classical_cost' if ignore_translucent else 'cost'))
        return {
            'alignment': alignment,
            'cost': cost,
            'bwc': self.best_worst_cost,
            'visited_states': len(self.nodes),
            'queued_states': len(self.nodes),
            'traversed_arcs': len(self.edges),
            'lp_solved': 0,
            'fitness': 1 - cost / self.best_worst_cost,
            # Additionally add the translucent alignment to the result
            'translucent_alignment': translucent_alignment,
            'move_cost': move_cost,
            'n_sync': n_sync,
            'n_log': n_log,
            'n_model': n_model,
            'n_silent': n_silent,
            'n_enabled_change': n_enabled_change,
            'n_execution_change': n_execution_change,
            'n_execution_enabled_change': n_execution_enabled_change,
        }
