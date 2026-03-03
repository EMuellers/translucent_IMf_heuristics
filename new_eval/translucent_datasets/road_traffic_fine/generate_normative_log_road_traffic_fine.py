"""
generate_normative_log.py

Like the original generate_log_without_noise() but:
  - Uses the manually crafted normative DPN instead of IMf discovery.
  - Computes delay attributes (delaySend/delayJudge/delayPrefecture) on the fly.
  - Filters traces with log-only alignment moves (control-flow violation).
  - Filters traces where any executed transition's data guard fails.
  - Annotates each event with guard-aware enabled activities.
  - Processes every trace individually (no variant deduplication).
"""

import copy
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignment
from pm4py.algo.conformance.alignments.petri_net.variants import state_equation_a_star as star
from datetime import datetime

from utils.DirectReachabilityGraph import DirectReachabilityGraph
from utils.ReachabilityGraph import ReachabilityGraph

def get_node_from_transition(graph, node, transition):
    edges = graph.out_edges(node, data= True, keys=True)
    for edge in edges:
        if graph.edges[edge[0], edge[1], edge[2]]["transition"] == transition:
            return edge[1]
    return None

def replace_spaces_in_activity_names(df, enabled_activities_name="enabled_activities"):
    """
    Replace spaces with underscores in activity names in a pm4py-style DataFrame.
    Applied to:
      - column "concept:name"
      - column `enabled_activities_name` (comma-separated list of activity labels)
    Returns a new DataFrame (does not mutate the input).
    """
    df = df.copy()

    if "concept:name" in df.columns:
        df["concept:name"] = df["concept:name"].str.replace(" ", "_", regex=False)

    if enabled_activities_name in df.columns:
        df[enabled_activities_name] = (
            df[enabled_activities_name]
            .str.split(", ")
            .apply(lambda acts: ", ".join(a.replace(" ", "_") for a in acts)
                   if isinstance(acts, list) else acts)
        )

    return df


# ================================================================
# 1.  Normative Petri net
# ================================================================

def build_normative_net():
    net = PetriNet("Road_Traffic_Fine_Normative")

    def p(name):
        pl = PetriNet.Place(name)
        net.places.add(pl)
        return pl

    def t(name, label):
        tr = PetriNet.Transition(name, label)
        net.transitions.add(tr)
        return tr

    source  = p("source")   # initial marking
    p1      = p("p1")       # after Create Fine
    p2      = p("p2")       # after Send Fine
    p3      = p("p3")       # after Insert Fine Notification
    p5      = p("p5")       # after Appeal to Judge filed
    p6      = p("p6")       # after Notify Result Appeal to Offender
    p7      = p("p7")       # after Appeal to Prefecture filed
    p8      = p("p8")       # after Send Appeal to Prefecture
    sink    = p("sink")     # final marking

    t_create        = t("Create Fine",                    "Create Fine") #
    t_send_fine     = t("Send Fine",                      "Send Fine") #
    t_notification  = t("Insert Fine Notification",       "Insert Fine Notification") #
    t_payment1      = t("Payment1",                       "Payment") #
    t_payment2      = t("Payment2",                       "Payment") #
    t_payment3      = t("Payment3",                       "Payment") #
    t_add_penalty   = t("Add penalty",                    "Add penalty") #
    t_credit        = t("Send for Credit Collection",     "Send for Credit Collection") #
    t_judge         = t("Appeal to Judge",                "Appeal to Judge") #
    t_notify        = t("Notify Result Appeal to Offender",                  "Notify Result Appeal to Offender") #
    t_prefecture    = t("Insert Date Appeal to Prefecture",           "Insert Date Appeal to Prefecture")
    t_send_appeal   = t("Send Appeal to Prefecture",                    "Send Appeal to Prefecture") #
    t_receive       = t("Receive Result Appeal from Prefecture",                 "Receive Result Appeal from Prefecture") #
    t_inv1          = t("Inv1",                           None)
    t_inv2          = t("Inv2",                           None)
    t_inv3          = t("Inv3",                           None)
    t_inv4          = t("Inv4",                           None)
    t_inv5          = t("Inv5",                           None)   
    t_inv6          = t("Inv6",                           None)


    # ---------------------------------------------------------------------------
    # Arcs
    # ---------------------------------------------------------------------------
    A = add_arc_from_to

    # Process entry
    A(source,  t_create,       net)
    A(t_create, p1,            net)

    # Early payment (right after fine creation)
    A(p1,      t_payment1,     net)
    A(t_payment1, p1,       net)

    A(p1,      t_inv1,        net)
    A(t_inv1,  sink,          net)


    # Send Fine (notification posted)
    A(p1,      t_send_fine,    net)
    A(t_send_fine, p2,         net)

    # Payment after Send Fine
    A(p2,      t_payment2,     net)
    A(t_payment2, p2,       net)
    A(t_inv2,  sink,           net)
    A(p2,      t_inv2,         net)


    # Offender receives notification
    A(p2,      t_notification, net)
    A(t_notification, p3,      net)

    # Add Penalty (180 days elapsed unpaid)
    A(p3,      t_add_penalty,  net)
    A(t_add_penalty, p3,       net)

    # Payment after penalty
    A(p3,      t_payment3,     net)
    A(t_payment3, p3,       net)
    A(p3,   t_inv3,         net)
    A(t_inv3,  sink,           net)

    # Credit collection (still unpaid)
    A(p3,      t_credit,       net)
    A(t_credit, sink,          net)

    # Appeal to Judge
    A(p3,      t_judge,        net)
    A(t_judge, p5,             net)
    A(p6,      t_notify,       net)
    A(t_notify, p3,            net)
    A(t_inv4,  sink,           net)
    A(t_inv5,  p3,             net)   # [UNCERTAIN] verify against Figure 12.1a

    # Appeal to Prefecture
    A(p3,      t_prefecture,   net)
    A(t_prefecture, p7,        net)
    A(p7,      t_send_appeal,  net)
    A(t_send_appeal, p8,       net)
    A(p8,      t_inv6,         net)   # dismissed (#) → close
    A(t_inv6,  sink,           net)
    A(p8,      t_receive,      net)   # not dismissed → receive result
    A(t_receive, p6,           net)   # [UNCERTAIN] verify against Figure 12.1a

    # Rest
    A(p5,      t_inv4,   net)
    A(p5, t_inv5,   net)
    A(t_inv4,  sink,     net) 

    # ---------------------------------------------------------------------------
    # Markings
    # ---------------------------------------------------------------------------
    im = Marking({source: 1})
    fm = Marking({sink:   1})
    return net, im, fm


# ================================================================
# 2.  Delay preprocessing
# ================================================================

_DELAY_SPECS = [
    ("Create Fine",               "Send Fine",                        "delaySend"),
    #("Insert Fine Notification",  "Appeal to Judge",                  "delayJudge"),
    #("Insert Fine Notification",  "Insert Date Appeal to Prefecture", "delayPrefecture"),
]

def _days_between(a, b):
    def _dt(v):
        if isinstance(v, datetime): return v
        try: return datetime.fromisoformat(str(v))
        except Exception: return None
    a, b = _dt(a), _dt(b)
    return (b - a).total_seconds() / 86400.0 if a and b else None

def preprocess_delays(log):
    """
    Compute delay attributes and store them as TRACE-level attributes.
    Only the first qualifying target event per trace is used.
    """
    for trace in log:
        last_ts   = {}
        recorded  = set()   # attrs already written for this trace
        for event in trace:
            act = event.get("concept:name", "")
            ts  = event.get("time:timestamp")
            for anchor, target, attr in _DELAY_SPECS:
                if act == target and anchor in last_ts and attr not in recorded:
                    d = _days_between(last_ts[anchor], ts)
                    if d is not None:
                        trace.attributes[attr] = d
                        recorded.add(attr)
            last_ts[act] = ts
    return log


def initial_state(trace):
    """
    Initialise the per-trace state dict.
    Delay values are read from trace-level attributes written by
    preprocess_delays(); all other variables start at their
    Create Fine defaults.
    """
    def _fd(k):
        try: return float(trace.attributes.get(k, float("inf")))
        except (TypeError, ValueError): return float("inf")

    return {
        "amount":          0.0,
        "payment":         0.0,
        "points":          0,
        "dismissal":       "NIL",
        "expenses":        0.0,
        "delaySend":       _fd("delaySend"),
        "last_notification": None,   # set by Insert Fine Notification
                                     # and Receive Result Appeal from Prefecture
    }

def update_state(state, activity, event):
    """
    Mutate `state` to reflect what `activity` writes.
    Called AFTER guard evaluation and annotation for this event.

    Attribute name assumptions (Road Traffic Fine XES log):
      "amount"      – fine amount on Create Fine event
      "points"      – licence points on Create Fine event
      "amount"      – payment amount on Payment event (accumulated)
      "expense"     – postal expense on Send Fine event
      "dismissal"   – dismissal code on Notify/Receive Result events
      delaySend/delayJudge/delayPrefecture – precomputed by preprocess_delays

    Requires Python >= 3.10 for match/case.
    """
    def _f(k, dfl=0.0):
        try: return float(event.get(k, dfl))
        except (TypeError, ValueError): return dfl
    def _s(k, dfl="NIL"):
        v = event.get(k, dfl); return str(v) if v is not None else dfl

    match activity:

        case "Create Fine":
            state["amount"]    = _f("amount")
            state["payment"]   = 0.0
            state["points"]    = int(_f("points"))
            state["dismissal"] = "NIL"

        case "Payment":
            # Each Payment event carries the amount paid in THAT payment.
            # Accumulate into the running total.
            state["payment"] += _f("paymentAmount")

        case "Send Fine":
            state["expenses"]  = _f("expense")
            #state["delaySend"] = _f("delaySend", float("inf"))

        case "Appeal to Judge":
            #state["delayJudge"] = _f("delayJudge", float("inf"))
            state["dismissal"] = _s("dismissal")
        
        case "Insert Fine Notification":
            state["last_notification"] = event.get("time:timestamp")

        case "Insert Date Appeal to Prefecture":
            pass
            #state["delayPrefecture"] = _f("delayPrefecture", float("inf"))
        
        case "Send Appeal to Prefecture":
            state["dismissal"] = _s("dismissal")
            
        case "Receive Result Appeal from Prefecture":
            state["last_notification"] = event.get("time:timestamp")

        case _:
            pass  # all other activities leave the state unchanged


# ================================================================
# 4.  Guards evaluated against state dict  (Figure 12.1b)
# ================================================================
"""
GUARDS = {
    
    "Send Fine":
        lambda s: s["delaySend"] < 90,
    "Appeal to Judge":
        lambda s: s["delayJudge"] < 60,
    "Insert Date Appeal to Prefecture":
        lambda s: s["delayPrefecture"] < 60,
    
    "Receive Result Appeal from Prefecture":
        lambda s: s["dismissal"] == "NIL",
    "Send for Credit Collection":
        lambda s: s["payment"] < s["amount"] + s["expenses"],
    "Inv1":
        lambda s: (s["dismissal"] != "NIL"
                   or (s["payment"] >= s["amount"] and s["points"] == 0)),
    "Inv2": lambda s: s["payment"] >= s["amount"] + s["expenses"],
    "Inv3": lambda s: s["payment"] >= s["amount"] + s["expenses"],
    "Inv4": lambda s: s["dismissal"] == "G",
    "Inv5": lambda s: s["dismissal"] == "NIL",
    "Inv6": lambda s: s["dismissal"] == "#",
}
"""

def _delay_from_notification(state, event):
    """
    Compute elapsed days between state["last_notification"] and
    the current event's timestamp.  Returns inf if anchor is unset.
    """
    anchor = state.get("last_notification")
    ts     = event.get("time:timestamp")
    if anchor is None or ts is None:
        return float("inf")
    try:
        return (ts - anchor).total_seconds() / 86400.0
    except TypeError:
        return _days_between(anchor, ts)

GUARDS = {
    "Send Fine":
        lambda s, e: s["delaySend"] < 90,
    "Appeal to Judge":
        lambda s, e: _delay_from_notification(s, e) < 60,
    "Insert Date Appeal to Prefecture":
        lambda s, e: _delay_from_notification(s, e) < 60,
    "Receive Result Appeal from Prefecture":
        lambda s, e: s["dismissal"] == "NIL",
    "Send for Credit Collection":
        lambda s, e: s["payment"] < s["amount"] + s["expenses"],
    "Inv1":
        lambda s, e: (s["dismissal"] != "NIL"
                      or (s["payment"] >= s["amount"] and s["points"] == 0)),
    "Inv2": lambda s, e: s["payment"] >= s["amount"] + s["expenses"],
    "Inv3": lambda s, e: s["payment"] >= s["amount"] + s["expenses"],
    "Inv4": lambda s, e: s["dismissal"] == "G",
    "Inv5": lambda s, e: s["dismissal"] == "NIL",
    "Inv6": lambda s, e: s["dismissal"] == "#",
}
"""
def data_enabled(label, state):
    g = GUARDS.get(label)
    return g(state) if g else True

def guard_aware_enabled(cf_enabled, state):
    return {a for a in cf_enabled if data_enabled(a, state)}
"""

def data_enabled(label, state, event):
    g = GUARDS.get(label)
    return g(state, event) if g else True

def guard_aware_enabled(cf_enabled, state, event):
    return {a for a in cf_enabled if data_enabled(a, state, event)}

# ================================================================
# 4.  Main function
# ================================================================

def generate_normative_log(
        path_to_log,
        enabled_activities_name="enabled_activities",
        pnml_path=None):
    """
    Returns an annotated EventLog.  Every trace in the input log is
    processed individually (no variant deduplication).  Traces that
    have a log-only alignment move or a data guard violation are
    discarded.  Retained traces have each event annotated with
    `enabled_activities_name`.
    """
    log = pm4py.read_xes(path_to_log, return_legacy_log_object=True)
    #log = clean_activity_names(log)
    log = preprocess_delays(log)

    net, im, fm = build_normative_net()
    if pnml_path is not None:
        pm4py.write_pnml(net, im, fm, pnml_path)

    parameters = {star.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}

    print("Building reachability graph")
    rg  = ReachabilityGraph(net, im, fm, 1)
    print("Building direct reachability graph and DFA")
    drg = DirectReachabilityGraph(rg).dfa

    annotated_log = EventLog(
        attributes=log.attributes, extensions=log.extensions,
        classifiers=log.classifiers, omni_present=log.omni_present,
        properties=log.properties)

    stats = {"cf_discarded": 0, "guard_discarded": 0, "kept": 0}
    n = len(log)
    

    print(f"Processing {n} traces individually ...")
    for trace_idx, trace in enumerate(log):
        print(f"  Trace {trace_idx + 1}/{n}")

        trace     = copy.deepcopy(trace)
        node      = frozenset({frozenset({0})})
        idx       = 0          # cursor into the real trace events
        discard_cf    = False
        discard_guard = False
        state = initial_state(trace)

        alignment_result = alignment.apply(trace, net, im, fm, parameters=parameters)
        
        if alignment_result["fitness"] < 1.0:
            stats["cf_discarded"] += 1
            continue
        
        for aligned_event in alignment_result["alignment"]: # Format: (Trace / Log, Model)
            move_log   = aligned_event[0][0]
            move_model = aligned_event[0][1]
            trans_name = aligned_event[1][1]

            if move_log == ">>":
                # Model-only move: advance DFA state (skip tau)
                if trans_name is not None:
                    node = get_node_from_transition(drg, node, move_model)

            elif move_model == ">>":
                # Log-only move: control-flow violation -> discard trace
                discard_cf = True
                break

            else:
                # Synchronous move
                event    = trace[idx]
                executed = event.get("concept:name", "")

                # --- Guard check (uses current state, BEFORE update) ---
                if not data_enabled(executed, state, event):
                    discard_guard = True
                    break

                # --- Annotation (uses current state, BEFORE update) ---
                cf_enabled = drg.nodes[node]["enabled_activities"]
                da_enabled = guard_aware_enabled(cf_enabled, state, event)
                da_enabled.add(executed)  # executed activity always included

                event[enabled_activities_name] = ", ".join(sorted(da_enabled))

                node = get_node_from_transition(drg, node, move_model)
                # --- Update variable state (AFTER guard + annotation) ---
                update_state(state, executed, event)
                idx += 1

        if discard_cf:
            stats["cf_discarded"] += 1
        elif discard_guard:
            stats["guard_discarded"] += 1
        else:
            annotated_log.append(trace)
            stats["kept"] += 1

    print(f"\nResults ({n} traces):")
    print(f"  Kept                    : {stats['kept']}")
    print(f"  Discarded (control-flow): {stats['cf_discarded']}")
    print(f"  Discarded (data guards) : {stats['guard_discarded']}")
    return annotated_log

if __name__ == "__main__":
    path_to_log = r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\original_datasets\Road_Traffic_Fine_Management_Process.xes.gz"
    annotated_log = generate_normative_log(path_to_log, enabled_activities_name="enabled_activities")
    output_log_path = r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\translucent_datasets\road_traffic_fine\normative_log.csv"
    # Check correctness: Is the excecuted activity always included in the enabled activities?
    for trace in annotated_log:
        for i in range(len(trace)):
            event = trace[i]
            executed_activity = event["concept:name"]
            enabled_activities = set(event["enabled_activities"].split(', '))
            if executed_activity not in enabled_activities:
                print(f"Error: Executed activity '{executed_activity}' is not in the set of enabled activities {enabled_activities} for event {i} in trace {trace.attributes['concept:name']}")
    df_log = pm4py.convert_to_dataframe(annotated_log)
    df_log = replace_spaces_in_activity_names(df_log, enabled_activities_name="enabled_activities")
    df_log.to_csv(output_log_path, index=False)
    print(f'Generated normative log at {output_log_path}')