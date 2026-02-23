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

"""
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

"""
import heapq
import networkx as nx
from typing import Optional
from pm4py.objects.log.obj import Trace
from pm4py.util.typing import AlignmentResult
from pm4py.objects.petri_net.utils.align_utils import SKIP

# Assuming tversky_index, TranslucentReachabilityGraph, ARTIFICIAL_END_TRANSITION_NAME, etc., are imported above.
# This is a new faster Test version
# TODO: Verify that it works correctly!
class TranslucentAlignmentStateGraph:
    def __init__(self, translucent_reachability_graph: TranslucentReachabilityGraph, trace: Trace):
        # We no longer pre-build the massive NetworkX graph here.
        # This makes initialization nearly instantaneous.
        self.trg = translucent_reachability_graph
        self.trace = trace
        self.best_worst_cost = len(trace) + self.trg.best_worst_cost
        self.initial_state = (0, self.trg.initial_state)
        self.final_state = (len(trace), self.trg.final_state)
        self.transition_labels = self.trg.transition_labels

    def get_optimal_alignment(self, trg_rev, ignore_translucent: bool = False) -> AlignmentResult:
        weight_key = 'classical_cost' if ignore_translucent else 'cost'
        
        def enabled_set_cost(enabled_set_trace: set[str], enabled_set_model: set[str]) -> float:
            return 1 - tversky_index(enabled_set_trace, enabled_set_model)

        # Precompute Heuristic: Minimum cost to reach the final model state.
        # This speeds up A* significantly by directing the search toward the goal.
        #trg_rev = nx.MultiDiGraph(self.trg).reverse(copy=False)
        heuristic_costs = nx.single_source_dijkstra_path_length(
            trg_rev, self.trg.final_state, weight="classical_cost"
        )

        # Priority Queue for A* Search: stores (f_score, g_score, state_id, state)
        # f_score = g_score (actual cost) + h_score (heuristic cost)
        open_set = []
        #heapq.heappush(open_set, (heuristic_costs.get(self.initial_state[1], 0), 0, id(self.initial_state), self.initial_state))
        heapq.heappush(open_set, (max(0, heuristic_costs.get(self.initial_state[1], 0) - len(self.trace)), 0, id(self.initial_state), self.initial_state))
        #heapq.heappush(open_set, (0, 0, id(self.initial_state), self.initial_state))
        
        g_score = {self.initial_state: 0}
        came_from = {}
        
        visited_states_count = 0
        queued_states_count = 1

        target_state = self.final_state

        while open_set:
            _, current_g, _, current_state = heapq.heappop(open_set)
            visited_states_count += 1
            
            if current_state == target_state:
                break
                
            # Skip if we've already found a better path to this state
            if current_g > g_score.get(current_state, float('inf')):
                continue
                
            t_idx, m_node = current_state
            m_enabled = self.trg.nodes[m_node]['enabled']
            
            # --- Generator for Lazy Edge Evaluation ---
            def get_neighbors():
                # 1. Log move
                if t_idx < len(self.trace):
                    t_event = self.trace[t_idx]
                    yield (t_idx + 1, m_node), {
                        'type': 'log', 'label': t_event.get('concept:name'), 
                        'cost': 1, 'classical_cost': 1, 'firing_sequence': ()
                    }
                    
                # Setup variables for Sync/Change moves
                if t_idx < len(self.trace):
                    t_event = self.trace[t_idx]
                    t_label = t_event.get('concept:name')
                    t_enabled = t_event.get('enabled', set())
                    sync_change_cost = enabled_set_cost(t_enabled, m_enabled)
                else:
                    t_label = None

                # 2. Model, Sync, and Execution Change moves
                for _, next_m, trg_edge in self.trg.edges(m_node, data=True):
                    m_label = trg_edge.get('label')
                    firing_sequence = trg_edge['firing_sequence']

                    # ---- Model move ----
                    if not (
                        m_label == ARTIFICIAL_END_TRANSITION_LABEL
                        and t_idx < len(self.trace)
                    ):
                        yield (t_idx, next_m), {
                            'type': 'model',
                            'label': None,
                            'cost': trg_edge['cost'],
                            'classical_cost': trg_edge['cost'],
                            'firing_sequence': firing_sequence
                        }

                    # ---- Sync / Change moves ----
                    if t_label is not None and m_label is not None:
                        if m_label == t_label:
                            yield (t_idx + 1, next_m), {
                                'type': 'sync',
                                'label': t_label,
                                'cost': sync_change_cost,
                                'classical_cost': 0,
                                'firing_sequence': firing_sequence
                            }
                        elif m_label != ARTIFICIAL_END_TRANSITION_LABEL:
                            yield (t_idx + 1, next_m), {
                                'type': 'change',
                                'label': t_label,
                                'cost': 1 + sync_change_cost,
                                'classical_cost': 3,
                                'firing_sequence': firing_sequence
                            }

            # Evaluate generated neighbors
            for next_state, edge_data in get_neighbors():
                tentative_g = current_g + edge_data[weight_key]
                
                if tentative_g < g_score.get(next_state, float('inf')):
                    came_from[next_state] = (current_state, edge_data)
                    g_score[next_state] = tentative_g
                    
                    # f(n) = g(n) + h(n)
                    #f_score = tentative_g + heuristic_costs.get(next_state[1], 0)
                    M = heuristic_costs.get(next_state[1], 0) # Min remaining model transitions
                    L = len(self.trace) - next_state[0]       # Remaining log events
                    h_score = max(0, M - L)
        
                    f_score = tentative_g + h_score
                    
                    #f_score = tentative_g + 0
                    heapq.heappush(open_set, (f_score, tentative_g, id(next_state), next_state))
                    queued_states_count += 1

        # --- Reconstruct Path ---
        path_edges = []
        curr = target_state
        while curr in came_from:
            prev, edge_data = came_from[curr]
            path_edges.append((prev, curr, edge_data))
            curr = prev
        path_edges.reverse()

        # --- Build Alignment Result ---
        alignment, translucent_alignment, move_cost = [], [], []
        cost, trace_idx = 0, 0
        n_sync = n_log = n_model = n_silent = n_enabled_change = n_execution_change = n_execution_enabled_change = 0

        for u, v, edge_data in path_edges:
            firing_sequence = edge_data.get('firing_sequence')
            if firing_sequence and firing_sequence[-1] == ARTIFICIAL_END_TRANSITION_NAME:
                continue
                
            len_seq = len(firing_sequence) if firing_sequence else 0
            if len_seq > 1:
                # Add silent moves to the alignment
                alignment.extend([(SKIP, None)] * (len_seq - 1))
                translucent_alignment.extend([(SKIP, None)] * (len_seq - 1))
                move_cost.extend([0] * (len_seq - 1))
                n_silent += len_seq - 1
                
            label = edge_data.get('label')
            transition_label = self.transition_labels[firing_sequence[-1]] if firing_sequence else SKIP
            u_m_node_enabled = self.trg.nodes[u[1]]['enabled']
            
            # Add other moves to the alignment
            alignment.append((label if label else SKIP, transition_label))
            translucent_alignment.append((
                (label if label else SKIP, self.trace[trace_idx]['enabled'] if label else set()), 
                (transition_label, u_m_node_enabled)
            ))
            
            if label:
                if firing_sequence:
                    if transition_label == label:
                        if self.trace[trace_idx]['enabled'] == u_m_node_enabled:
                            n_sync += 1
                        else:
                            n_enabled_change += 1
                    else:
                        if self.trace[trace_idx]['enabled'] == u_m_node_enabled:
                            n_execution_change += 1
                        else:
                            n_execution_enabled_change += 1
                else:
                    n_log += 1
                trace_idx += 1
            else:
                n_model += 1
                
            edge_cost = edge_data.get(weight_key)
            cost += edge_cost
            move_cost.append(edge_cost)

        return {
            'alignment': alignment,
            'cost': cost,
            'bwc': self.best_worst_cost,
            'visited_states': visited_states_count,
            'queued_states': queued_states_count,
            'traversed_arcs': queued_states_count, # Approximated based on queued
            'lp_solved': 0,
            'fitness': 1 - (cost / self.best_worst_cost) if self.best_worst_cost else 0,
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