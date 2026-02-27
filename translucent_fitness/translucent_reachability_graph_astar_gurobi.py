from collections import deque
import heapq
from typing import Optional

import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from pm4py import PetriNet, Marking
from pm4py.algo.analysis.workflow_net.algorithm import apply as is_workflow_net
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.semantics import enabled_transitions, execute
from pm4py.objects.petri_net.utils.align_utils import SKIP
from pm4py.util.typing import AlignmentResult

from translucent_fitness.utils import (
    add_artificial_end_transition,
    ARTIFICIAL_END_TRANSITION_NAME,
    ARTIFICIAL_END_TRANSITION_LABEL,
)


class TranslucentReachabilityGraph(nx.MultiDiGraph):
    def __init__(self, accepting_petri_net: tuple[PetriNet, Marking, Marking]):
        if not is_workflow_net(accepting_petri_net[0]):
            raise ValueError("The Petri net is not a workflow net")

        accepting_petri_net = add_artificial_end_transition(accepting_petri_net)
        super().__init__()
        net, initial_marking, final_marking = accepting_petri_net
        self.transition_labels = {t.name: t.label for t in net.transitions}
        self.marking_map = {initial_marking: 0, final_marking: 1}
        self.initial_state = 0
        self.final_state = 1
        self.add_node(0, marking=initial_marking, enabled=set())
        self.add_node(1, marking=final_marking, enabled=set())
        open_list = deque([0])

        semantics_cache: dict[Marking, tuple] = {}
        observable_arcs_cache: dict[Marking, tuple] = {}

        def get_semantics(marking):
            if marking not in semantics_cache:
                semantics_cache[marking] = tuple(
                    (t, execute(t, net, marking))
                    for t in enabled_transitions(net, marking)
                )
            return semantics_cache[marking]

        def get_observable_arcs(start_marking):
            if start_marking in observable_arcs_cache:
                return observable_arcs_cache[start_marking]
            visited = {start_marking}
            queue = deque([(start_marking, tuple())])
            arcs = []
            while queue:
                marking, seq = queue.popleft()
                for transition, next_marking in get_semantics(marking):
                    next_seq = seq + (transition.name,)
                    if transition.label is not None:
                        arcs.append((next_marking, next_seq, transition.label))
                    elif next_marking not in visited:
                        visited.add(next_marking)
                        queue.append((next_marking, next_seq))
            observable_arcs_cache[start_marking] = tuple(arcs)
            return observable_arcs_cache[start_marking]

        while open_list:
            current_node = open_list.popleft()
            current_marking = self.nodes[current_node]["marking"]
            current_enabled = self.nodes[current_node]["enabled"]
            for post_marking, firing_sequence, label in get_observable_arcs(current_marking):
                if post_marking not in self.marking_map:
                    self.marking_map[post_marking] = self.number_of_nodes()
                    self.add_node(self.marking_map[post_marking], marking=post_marking, enabled=set())
                    open_list.append(self.marking_map[post_marking])
                if label != ARTIFICIAL_END_TRANSITION_LABEL:
                    current_enabled.add(label)
                self.add_edge(
                    current_node,
                    self.marking_map[post_marking],
                    firing_sequence=firing_sequence,
                    label=label,
                    cost=0 if firing_sequence[-1] == ARTIFICIAL_END_TRANSITION_NAME else 1,
                )
        self.best_worst_cost = nx.dijkstra_path_length(
            self, self.initial_state, self.final_state, weight="cost"
        )


def tversky_index(set1: set, set2: set, alpha: float = 1, beta: float = 1) -> float:
    intersection_size = len(set1.intersection(set2))
    denominator = (
        intersection_size
        + alpha * len(set1.difference(set2))
        + beta * len(set2.difference(set1))
    )
    return 1.0 if denominator == 0 else intersection_size / denominator


# ---------------------------------------------------------------------------
# Extended-marking-equation LP heuristic (Gurobi, with hot-starting)
# ---------------------------------------------------------------------------

class _TranslucentMarkingEquationLP:
    """Builds the LP once per (trace, model) pair and hot-starts across A* states.

    Variables (all continuous >= 0):
      log_i          -- log move consuming trace position i  (cost 1)
      model_(e,k)    -- pure model move along TRG edge k from node e  (cost 0 or 1)
      diag_(i,e,k)   -- diagonal move: trace position i paired with TRG edge k
                        from node e  (cost = enabled_cost if label matches,
                                           1 + enabled_cost otherwise)

    Constraints:
      (T) Trace token balance for each trace position 0..remaining
      (M) Model token balance for each TRG node
      (E) Extended ordering: for each prefix length k, total trace-consuming
          transitions fired up to position k-1 <= k  (prevents consuming
          future trace tokens before earlier ones)

    The RHS of (T) and (M) changes per A* state; (E) is state-independent.
    Hot-starting: after solving, the current Gurobi solution is used as a
    starting point (PStart / DStart) for the next solve.
    """

    def __init__(
        self,
        trg: TranslucentReachabilityGraph,
        trace_labels: list[str],
        trace_enabled_sets: list[frozenset],
        model_enabled_sets: dict[int, frozenset],
    ):
        self.trg = trg
        self.trace_len = len(trace_labels)
        self.trace_labels = trace_labels
        self.trace_enabled_sets = trace_enabled_sets
        self.model_enabled_sets = model_enabled_sets

        # Collect all TRG edges with an index key
        self.trg_edges: list[tuple[int, int, dict]] = [
            (u, v, d)
            for u, v, d in trg.edges(data=True)
            if d["label"] != ARTIFICIAL_END_TRANSITION_LABEL
        ]
        # Edge index: (src, tgt, firing_seq) -> position in self.trg_edges
        self.edge_idx: dict[tuple, int] = {
            (u, v, d["firing_sequence"]): k
            for k, (u, v, d) in enumerate(self.trg_edges)
        }

        # Precompute diagonal costs for every (trace_pos, edge_idx) pair.
        # diag_cost[i][k] = actual translucent cost of pairing trace event i
        # with TRG edge k. Only defined where the TRG edge is visible (has a
        # non-None, non-artificial label).
        n_edges = len(self.trg_edges)
        self.diag_cost: list[list[Optional[float]]] = [
            [None] * n_edges for _ in range(self.trace_len)
        ]
        for i in range(self.trace_len):
            ev_label = trace_labels[i]
            ev_enabled = trace_enabled_sets[i]
            for k, (u, v, d) in enumerate(self.trg_edges):
                trg_label = d["label"]
                if trg_label is None:
                    continue  # silent model arc -- no diagonal
                src_enabled = model_enabled_sets[u]
                enabled_cost = 1.0 - tversky_index(ev_enabled, src_enabled)
                if trg_label == ev_label:
                    self.diag_cost[i][k] = enabled_cost          # sync / enabled-change
                else:
                    self.diag_cost[i][k] = 1.0 + enabled_cost    # exec-change / EEC

        self._build_model()

    # ------------------------------------------------------------------
    def _build_model(self):
        """Construct the Gurobi LP with full trace (will be restricted per state)."""
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()
        m = gp.Model(env=env)
        m.Params.OutputFlag = 0
        m.Params.Method = 1          # dual simplex -- best for hot-starting
        m.Params.InfUnbdInfo = 1

        T = self.trace_len
        E = len(self.trg_edges)
        trg_nodes = list(self.trg.nodes())

        # ---- variables ------------------------------------------------
        # log move variables: one per trace position
        log_vars = m.addVars(T, lb=0.0, name="log")

        # model move variables: one per TRG edge
        model_vars = m.addVars(E, lb=0.0, name="model")

        # diagonal variables: one per (trace_pos, edge_idx) where cost is defined
        diag_idx = [
            (i, k)
            for i in range(T)
            for k in range(E)
            if self.diag_cost[i][k] is not None
        ]
        diag_vars = m.addVars(diag_idx, lb=0.0, name="diag")

        self._log_vars = log_vars
        self._model_vars = model_vars
        self._diag_vars = diag_vars
        self._diag_idx = set(diag_idx)

        # ---- objective ------------------------------------------------
        obj = (
            gp.quicksum(log_vars[i] for i in range(T))
            + gp.quicksum(
                self.trg_edges[k][2]["cost"] * model_vars[k] for k in range(E)
            )
            + gp.quicksum(
                self.diag_cost[i][k] * diag_vars[i, k] for i, k in diag_idx
            )
        )
        m.setObjective(obj, GRB.MINIMIZE)

        # ---- trace token balance constraints --------------------------
        # Place p_i in the trace net (shifted so p_0 = current trace position).
        # RHS will be updated per A* state via constraint attributes.
        # We create constraints: net_flow_into_p_i = rhs_i
        #   net_flow = (transitions producing p_i) - (transitions consuming p_i)
        # Transitions consuming p_i:  log_i, diag_(i, *)
        # Transitions producing p_i:  log_{i-1}, diag_(i-1, *)  [if i > 0]
        #                              + "source" token if i == 0
        # We encode this as: flow_in - flow_out = rhs
        # For the full LP (before restricting to a state suffix) rhs = 0 for
        # interior nodes, with the start/end tokens handled via RHS updates.

        trace_cons = []
        for i in range(T + 1):
            # Flow OUT of place p_i (consumed by transitions starting at i)
            out_expr = gp.LinExpr()
            if i < T:
                out_expr += log_vars[i]
                for k in range(E):
                    if (i, k) in self._diag_idx:
                        out_expr += diag_vars[i, k]

            # Flow INTO place p_i (produced by transitions ending at i)
            in_expr = gp.LinExpr()
            if i > 0:
                in_expr += log_vars[i - 1]
                for k in range(E):
                    if (i - 1, k) in self._diag_idx:
                        in_expr += diag_vars[i - 1, k]

            # Constraint: in - out = rhs  (rhs updated per state)
            con = m.addLConstr(in_expr - out_expr, GRB.EQUAL, 0.0, name=f"trace_{i}")
            trace_cons.append(con)
        self._trace_cons = trace_cons  # length T+1

        # ---- model token balance constraints --------------------------
        model_cons = {}
        for n in trg_nodes:
            # Edges leaving n
            out_expr = gp.LinExpr()
            for k, (u, v, d) in enumerate(self.trg_edges):
                if u == n:
                    out_expr += model_vars[k]
                    for i in range(T):
                        if (i, k) in self._diag_idx:
                            out_expr += diag_vars[i, k]

            # Edges entering n
            in_expr = gp.LinExpr()
            for k, (u, v, d) in enumerate(self.trg_edges):
                if v == n:
                    in_expr += model_vars[k]
                    for i in range(T):
                        if (i, k) in self._diag_idx:
                            in_expr += diag_vars[i, k]

            con = m.addLConstr(in_expr - out_expr, GRB.EQUAL, 0.0, name=f"model_{n}")
            model_cons[n] = con
        self._model_cons = model_cons

        # ---- extended marking equation constraints --------------------
        # For each prefix length p (1 .. T): total trace-consuming transitions
        # across positions 0..p-1 must be <= p. This prevents consuming future
        # trace tokens before earlier ones are consumed.
        ext_cons = []
        for p in range(1, T + 1):
            expr = gp.LinExpr()
            for i in range(p):
                expr += log_vars[i]
                for k in range(E):
                    if (i, k) in self._diag_idx:
                        expr += diag_vars[i, k]
            con = m.addLConstr(expr, GRB.LESS_EQUAL, float(p), name=f"ext_{p}")
            ext_cons.append(con)
        self._ext_cons = ext_cons

        m.update()
        self._model = m
        self._last_state: Optional[tuple[int, int]] = None

    # ------------------------------------------------------------------
    def _set_rhs_for_state(self, trace_idx: int, model_node: int):
        """Update the RHS of balance constraints to reflect state (trace_idx, model_node).

        The LP now covers trace positions [trace_idx, trace_len), asking:
        what is the cheapest way to go from model_node to final_state while
        consuming exactly the remaining trace events?

        Trace balance RHS (over the shifted index j = i - trace_idx):
          p_0 (= p_{trace_idx}): net demand -1  (source token consumed)
          p_{remaining} (= p_{trace_len}): net demand +1 (sink token produced)
          all others: 0

        Model balance RHS:
          model_node: -1 (source)
          final_state: +1 (sink)
          others: 0
        """
        remaining = self.trace_len - trace_idx

        # Trace constraints: indices 0..trace_len map to absolute positions.
        # We need to zero out positions before trace_idx (lock them out by
        # forcing the variables for those positions to zero via bounds) and
        # set the RHS for positions [trace_idx, trace_len].
        for i in range(self.trace_len + 1):
            if i < trace_idx:
                # Positions before current: lock all variables to 0 via bounds,
                # RHS = 0 (no tokens here).
                self._trace_cons[i].RHS = 0.0
            elif i == trace_idx:
                self._trace_cons[i].RHS = -1.0   # source: consume 1 token here
            elif i == self.trace_len:
                self._trace_cons[i].RHS = 1.0    # sink: produce 1 token here
            else:
                self._trace_cons[i].RHS = 0.0

        # Freeze variables for positions before trace_idx
        for i in range(trace_idx):
            self._log_vars[i].UB = 0.0
            self._log_vars[i].LB = 0.0
        for i in range(trace_idx, self.trace_len):
            self._log_vars[i].UB = GRB.INFINITY
        for k in range(len(self.trg_edges)):
            for i in range(trace_idx):
                if (i, k) in self._diag_idx:
                    self._diag_vars[i, k].UB = 0.0
                    self._diag_vars[i, k].LB = 0.0
            for i in range(trace_idx, self.trace_len):
                if (i, k) in self._diag_idx:
                    self._diag_vars[i, k].UB = GRB.INFINITY

        # Model constraints
        final = self.trg.final_state
        for n, con in self._model_cons.items():
            if n == model_node and n == final:
                con.RHS = 0.0   # source == sink: no net flow needed
            elif n == model_node:
                con.RHS = -1.0
            elif n == final:
                con.RHS = 1.0
            else:
                con.RHS = 0.0

        # Extended constraints: only cover remaining positions [trace_idx, trace_len)
        # For prefix length p (1..remaining): sum of trace-consuming vars at
        # positions trace_idx .. trace_idx+p-1 <= p.
        # The constraints were built for absolute positions; update RHS accordingly.
        # For p > remaining the constraint is slack (RHS = remaining, always satisfied).
        for p, con in enumerate(self._ext_cons, start=1):
            con.RHS = float(min(p, remaining))

    # ------------------------------------------------------------------
    def solve(self, trace_idx: int, model_node: int) -> float:
        """Return LP lower bound on remaining alignment cost from (trace_idx, model_node)."""
        if model_node not in self.trg.can_reach_final_model_nodes if hasattr(self.trg, "can_reach_final_model_nodes") else False:
            return float("inf")
        if trace_idx == self.trace_len and model_node == self.trg.final_state:
            return 0.0

        self._set_rhs_for_state(trace_idx, model_node)

        # Hot-start: if previous solution exists, use it as a starting point.
        # Gurobi dual simplex reuses the basis automatically after RHS changes
        # when we call update() + optimize() without resetting the basis.
        self._model.update()
        self._model.optimize()

        status = self._model.Status
        if status == GRB.OPTIMAL:
            return max(0.0, self._model.ObjVal)
        elif status == GRB.INFEASIBLE:
            return float("inf")
        else:
            # Fallback: unbounded or other -- should not happen with valid inputs
            return 0.0

    def dispose(self):
        self._model.dispose()


# ---------------------------------------------------------------------------
# TranslucentAlignmentStateGraph
# ---------------------------------------------------------------------------

class TranslucentAlignmentStateGraph(nx.MultiDiGraph):
    def __init__(self, translucent_reachability_graph: TranslucentReachabilityGraph, trace: Trace):
        super().__init__()
        self.trg = translucent_reachability_graph
        self.trace = trace
        trace_len = len(trace)
        self.best_worst_cost = trace_len + translucent_reachability_graph.best_worst_cost
        self.initial_state = (0, 0)
        self.final_state = (trace_len, 1)
        self.transition_labels = translucent_reachability_graph.transition_labels
        self.trace_len = trace_len
        self.trace_labels = [event.get("concept:name") for event in trace]
        self.trace_enabled_sets = [frozenset(event.get("enabled", set())) for event in trace]
        self.model_enabled_sets = {
            node: frozenset(translucent_reachability_graph.nodes[node]["enabled"])
            for node in translucent_reachability_graph.nodes
        }
        self.model_enabled_sets_raw = {
            node: translucent_reachability_graph.nodes[node]["enabled"]
            for node in translucent_reachability_graph.nodes
        }
        self.enabled_cost_cache: dict[tuple[frozenset, frozenset], float] = {}
        self.trg_edges_by_source: dict[int, tuple] = {}
        for node in translucent_reachability_graph.nodes:
            self.trg_edges_by_source[node] = tuple(
                (target, edge_data)
                for _, target, edge_data in translucent_reachability_graph.edges(node, data=True)
            )

        reverse_adjacency: dict[int, list[tuple[int, int]]] = {
            node: [] for node in translucent_reachability_graph.nodes
        }
        for src in translucent_reachability_graph.nodes:
            for target, edge_data in self.trg_edges_by_source[src]:
                reverse_adjacency[target].append((src, edge_data["cost"]))

        self.min_model_cost_to_final: dict[int, float] = {
            translucent_reachability_graph.final_state: 0.0
        }
        cost_queue = deque([translucent_reachability_graph.final_state])
        while cost_queue:
            node = cost_queue.popleft()
            node_cost = self.min_model_cost_to_final[node]
            for predecessor, edge_cost in reverse_adjacency[node]:
                predecessor_cost = node_cost + edge_cost
                if (
                    predecessor not in self.min_model_cost_to_final
                    or predecessor_cost < self.min_model_cost_to_final[predecessor]
                ):
                    self.min_model_cost_to_final[predecessor] = predecessor_cost
                    if edge_cost == 0:
                        cost_queue.appendleft(predecessor)
                    else:
                        cost_queue.append(predecessor)

        self.can_reach_final_model_nodes: set[int] = set()
        queue = deque([translucent_reachability_graph.final_state])
        while queue:
            node = queue.popleft()
            if node in self.can_reach_final_model_nodes:
                continue
            self.can_reach_final_model_nodes.add(node)
            for predecessor, _ in reverse_adjacency[node]:
                if predecessor not in self.can_reach_final_model_nodes:
                    queue.append(predecessor)

        # Precompute reachable labels (used by fallback heuristic)
        self.reachable_labels_from: dict[int, frozenset] = {}
        for node in translucent_reachability_graph.nodes:
            labels: set[str] = set()
            visited_bfs: set[int] = set()
            bfs: deque[int] = deque([node])
            while bfs:
                n = bfs.popleft()
                if n in visited_bfs:
                    continue
                visited_bfs.add(n)
                for target, edata in self.trg_edges_by_source[n]:
                    lbl = edata["label"]
                    if lbl and lbl != ARTIFICIAL_END_TRANSITION_LABEL:
                        labels.add(lbl)
                    if target not in visited_bfs:
                        bfs.append(target)
            self.reachable_labels_from[node] = frozenset(labels)

        # Attach TRG reference for the LP (needed for can_reach check)
        translucent_reachability_graph.can_reach_final_model_nodes = (
            self.can_reach_final_model_nodes
        )

        # Build the LP heuristic (constructed once, reused across A* states)
        self._lp: Optional[_TranslucentMarkingEquationLP] = None

        self._last_search_visited_states = 0
        self._last_search_queued_states = 0
        self._last_search_traversed_arcs = 0
        self._last_search_lp_solved = 0

    def _get_lp(self) -> _TranslucentMarkingEquationLP:
        if self._lp is None:
            self._lp = _TranslucentMarkingEquationLP(
                self.trg,
                self.trace_labels,
                self.trace_enabled_sets,
                self.model_enabled_sets,
            )
        return self._lp

    def get_optimal_alignment_cost(self, ignore_translucent: bool = False) -> float:
        return self._compute_shortest_path(
            ignore_translucent=ignore_translucent, return_path=False
        )["cost"]

    def _enabled_set_cost(
        self, enabled_set_trace: frozenset, enabled_set_model: frozenset
    ) -> float:
        cache_key = (enabled_set_trace, enabled_set_model)
        if cache_key not in self.enabled_cost_cache:
            self.enabled_cost_cache[cache_key] = 1 - tversky_index(
                enabled_set_trace, enabled_set_model
            )
        return self.enabled_cost_cache[cache_key]

    def _heuristic_fallback(self, state: tuple[int, int]) -> float:
        """Fast admissible fallback (used when LP is disabled or model_node is dead-end)."""
        trace_idx, model_node = state
        if model_node not in self.can_reach_final_model_nodes:
            return float("inf")
        min_model_cost = self.min_model_cost_to_final.get(model_node, float("inf"))
        if min_model_cost == float("inf"):
            return float("inf")
        remaining_events = self.trace_len - trace_idx
        bound_a = max(0.0, min_model_cost - remaining_events)
        reachable = self.reachable_labels_from[model_node]
        bound_b = sum(
            1 for i in range(trace_idx, self.trace_len)
            if self.trace_labels[i] not in reachable
        )
        return max(bound_a, float(bound_b))

    def _heuristic(
        self,
        state: tuple[int, int],
        ignore_translucent: bool,
        lp: _TranslucentMarkingEquationLP,
        lp_solved_counter: list[int],
    ) -> float:
        """Extended marking equation LP heuristic with hot-starting.

        The LP is solved at every expanded A* state. Gurobi's dual simplex
        automatically hot-starts from the previous basis since only the RHS
        changes between consecutive states (same variable and constraint
        structure). This makes repeated solves fast.

        For the classical (ignore_translucent) mode we fall back to the
        cheaper precomputed bound, since the LP encodes translucent costs.
        """
        trace_idx, model_node = state
        if model_node not in self.can_reach_final_model_nodes:
            return float("inf")

        if ignore_translucent:
            return self._heuristic_fallback(state)

        h = lp.solve(trace_idx, model_node)
        lp_solved_counter[0] += 1
        return h

    def _compute_shortest_path(self, ignore_translucent: bool, return_path: bool) -> dict:
        weight_key = "classical_cost" if ignore_translucent else "cost"
        initial_state = self.initial_state
        final_state = self.final_state
        trace_len = self.trace_len

        lp = self._get_lp()
        lp_solved_counter = [0]

        initial_h = self._heuristic(initial_state, ignore_translucent, lp, lp_solved_counter)
        if initial_h == float("inf"):
            raise ValueError("No alignment path found between initial and final TASG states.")

        frontier: list[tuple[float, float, int, tuple[int, int]]] = [
            (initial_h, 0.0, 0, initial_state)
        ]
        best_cost: dict[tuple[int, int], float] = {initial_state: 0.0}
        predecessor: dict[tuple[int, int], tuple[tuple[int, int], dict]] = {}
        tie_breaker = 0
        visited_states = 0
        queued_states = 1
        traversed_arcs = 0

        while frontier:
            _, current_cost, _, state = heapq.heappop(frontier)
            if current_cost > best_cost.get(state, float("inf")):
                continue

            visited_states += 1
            trace_idx, model_node = state

            if model_node not in self.can_reach_final_model_nodes:
                continue

            if state == final_state:
                self._last_search_visited_states = visited_states
                self._last_search_queued_states = queued_states
                self._last_search_traversed_arcs = traversed_arcs
                self._last_search_lp_solved = lp_solved_counter[0]
                if not return_path:
                    return {"cost": current_cost}
                path_edges = []
                cursor = final_state
                while cursor in predecessor:
                    prev_state, edge_data = predecessor[cursor]
                    path_edges.append((prev_state, cursor, edge_data))
                    cursor = prev_state
                path_edges.reverse()
                return {"cost": current_cost, "path_edges": path_edges}

            # --- log move ---
            if trace_idx < trace_len:
                traversed_arcs += 1
                next_state = (trace_idx + 1, model_node)
                edge_data = {
                    "firing_sequence": (),
                    "label": self.trace_labels[trace_idx],
                    "cost": 1,
                    "classical_cost": 1,
                    "type": "log",
                }
                next_cost = current_cost + edge_data[weight_key]
                if next_cost < best_cost.get(next_state, float("inf")):
                    best_cost[next_state] = next_cost
                    next_h = self._heuristic(next_state, ignore_translucent, lp, lp_solved_counter)
                    if next_h == float("inf"):
                        continue
                    if return_path:
                        predecessor[next_state] = (state, edge_data)
                    tie_breaker += 1
                    heapq.heappush(
                        frontier,
                        (next_cost + next_h, next_cost, tie_breaker, next_state),
                    )
                    queued_states += 1

            event_label = self.trace_labels[trace_idx] if trace_idx < trace_len else None
            event_enabled = self.trace_enabled_sets[trace_idx] if trace_idx < trace_len else None
            model_enabled = self.model_enabled_sets[model_node]

            for target_model_node, trg_edge_data in self.trg_edges_by_source[model_node]:
                traversed_arcs += 1

                # --- pure model move ---
                model_move_data = {
                    "firing_sequence": trg_edge_data["firing_sequence"],
                    "label": None,
                    "cost": trg_edge_data["cost"],
                    "classical_cost": trg_edge_data["cost"],
                    "type": "model",
                }
                model_next_state = (trace_idx, target_model_node)
                model_next_cost = current_cost + model_move_data[weight_key]
                if model_next_cost < best_cost.get(model_next_state, float("inf")):
                    best_cost[model_next_state] = model_next_cost
                    next_h = self._heuristic(
                        model_next_state, ignore_translucent, lp, lp_solved_counter
                    )
                    if next_h == float("inf"):
                        continue
                    if return_path:
                        predecessor[model_next_state] = (state, model_move_data)
                    tie_breaker += 1
                    heapq.heappush(
                        frontier,
                        (model_next_cost + next_h, model_next_cost, tie_breaker, model_next_state),
                    )
                    queued_states += 1

                if event_label is None:
                    continue

                trg_label = trg_edge_data["label"]
                if trg_label == ARTIFICIAL_END_TRANSITION_LABEL:
                    continue

                # --- diagonal move (sync / enabled-change / exec-change / EEC) ---
                enabled_cost = self._enabled_set_cost(event_enabled, model_enabled)
                if trg_label == event_label:
                    move_data = {
                        "firing_sequence": trg_edge_data["firing_sequence"],
                        "label": event_label,
                        "cost": enabled_cost,
                        "classical_cost": 0,
                        "type": "sync",
                    }
                else:
                    move_data = {
                        "firing_sequence": trg_edge_data["firing_sequence"],
                        "label": event_label,
                        "cost": 1 + enabled_cost,
                        "classical_cost": 3,
                        "type": "change",
                    }

                traversed_arcs += 1
                next_state = (trace_idx + 1, target_model_node)
                next_cost = current_cost + move_data[weight_key]
                if next_cost < best_cost.get(next_state, float("inf")):
                    best_cost[next_state] = next_cost
                    next_h = self._heuristic(
                        next_state, ignore_translucent, lp, lp_solved_counter
                    )
                    if next_h == float("inf"):
                        continue
                    if return_path:
                        predecessor[next_state] = (state, move_data)
                    tie_breaker += 1
                    heapq.heappush(
                        frontier,
                        (next_cost + next_h, next_cost, tie_breaker, next_state),
                    )
                    queued_states += 1

        raise ValueError("No alignment path found between initial and final TASG states.")

    def get_optimal_alignment(self, ignore_translucent: bool = False) -> AlignmentResult:
        alignment = []
        translucent_alignment = []
        move_cost = []
        trace_idx = 0
        weight_key = "classical_cost" if ignore_translucent else "cost"
        transition_labels = self.transition_labels
        shortest_path_result = self._compute_shortest_path(
            ignore_translucent=ignore_translucent, return_path=True
        )
        path_edges = shortest_path_result["path_edges"]
        total_cost = shortest_path_result["cost"]
        n_sync = n_log = n_model = n_silent = 0
        n_enabled_change = n_execution_change = n_execution_enabled_change = 0

        for u, _, edge_data in path_edges:
            firing_sequence = edge_data.get("firing_sequence")
            if firing_sequence and firing_sequence[-1] == ARTIFICIAL_END_TRANSITION_NAME:
                continue
            alignment.extend([(SKIP, None)] * (len(firing_sequence) - 1))
            translucent_alignment.extend([(SKIP, None)] * (len(firing_sequence) - 1))
            move_cost.extend([0] * (len(firing_sequence) - 1))
            n_silent += len(firing_sequence) - 1 if len(firing_sequence) > 1 else 0

            label = edge_data.get("label")
            model_label = (
                transition_labels[firing_sequence[-1]] if firing_sequence else SKIP
            )
            alignment.append((label if label else SKIP, model_label))
            translucent_alignment.append(
                (
                    (label if label else SKIP, self.trace[trace_idx]["enabled"] if label else set()),
                    (model_label, self.model_enabled_sets_raw[u[1]]),
                )
            )

            if label:
                if firing_sequence:
                    if transition_labels[firing_sequence[-1]] == label:
                        if self.trace[trace_idx]["enabled"] == self.model_enabled_sets_raw[u[1]]:
                            n_sync += 1
                        else:
                            n_enabled_change += 1
                    else:
                        if self.trace[trace_idx]["enabled"] == self.model_enabled_sets_raw[u[1]]:
                            n_execution_change += 1
                        else:
                            n_execution_enabled_change += 1
                else:
                    n_log += 1
                trace_idx += 1
            else:
                n_model += 1
            move_cost.append(edge_data.get(weight_key))

        # Clean up Gurobi model
        if self._lp is not None:
            self._lp.dispose()
            self._lp = None

        return {
            "alignment": alignment,
            "cost": total_cost,
            "bwc": self.best_worst_cost,
            "visited_states": self._last_search_visited_states,
            "queued_states": self._last_search_queued_states,
            "traversed_arcs": self._last_search_traversed_arcs,
            "lp_solved": self._last_search_lp_solved,
            "fitness": 1 - total_cost / self.best_worst_cost,
            "translucent_alignment": translucent_alignment,
            "move_cost": move_cost,
            "n_sync": n_sync,
            "n_log": n_log,
            "n_model": n_model,
            "n_silent": n_silent,
            "n_enabled_change": n_enabled_change,
            "n_execution_change": n_execution_change,
            "n_execution_enabled_change": n_execution_enabled_change,
        }
