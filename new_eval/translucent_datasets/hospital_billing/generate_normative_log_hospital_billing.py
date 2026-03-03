"""
generate_normative_log_hospital_billing.py

Adapted version of generate_normative_log() for the Hospital Billing log.
Source: Mannhardt (2018), Section 15.2, Figure 15.3

Guard variables (all read from event / trace attributes, tracked in state):
  caseType    – set by NEW event
  speciality  – set by NEW event
  closeCode   – set by events that write it (FIN, CODE OK, CODE NOK, ...)
  isClosed    – set by events that write it

No time-based delays: preprocess_delays is a no-op for this log.
"""

import copy
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignment
from pm4py.algo.conformance.alignments.petri_net.variants import state_equation_a_star as star
from utils.DirectReachabilityGraph import DirectReachabilityGraph
from utils.ReachabilityGraph import ReachabilityGraph

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



def get_node_from_transition(graph, node, transition):
    edges = graph.out_edges(node, data= True, keys=True)
    for edge in edges:
        if graph.edges[edge[0], edge[1], edge[2]]["transition"] == transition:
            return edge[1]
    return None

# ================================================================
# 1.  Normative Petri net
#     Topology from Figure 15.3a.
#     Activities use abbreviated labels matching the Hospital Billing XES log.
# ================================================================

def build_normative_net():
    net = PetriNet("Hospital_Billing_Normative")

    def p(name):
        pl = PetriNet.Place(name); net.places.add(pl); return pl

    def t(name, label):
        tr = PetriNet.Transition(name, label); net.transitions.add(tr); return tr

    source = p("source")
    p1 = p("p1"); p2 = p("p2"); p3 = p("p3"); p4 = p("p4")
    p5 = p("p5"); p6 = p("p6"); p7 = p("p7"); p8 = p("p8")
    p9 = p("p9"); p10 = p("p10"); p11 = p("p11"); p12 = p("p12")
    p13 = p("p13"); p14 = p("p14"); p15 = p("p15")
    sink = p("sink")

    # Visible transitions
    # Where an activity name appears multiple times in the net, transitions
    # are suffixed with #1, #2, #3 matching Figure 15.3a.
    t_new         = t("NEW",           "NEW") #
    t_change_diagn= t("CHANGE DIAGN",  "CHANGE DIAGN") # 
    t_code_ok     = t("CODE OK",       "CODE OK") #
    t_code_nok    = t("CODE NOK",      "CODE NOK")
    t_release     = t("RELEASE",       "RELEASE") #
    t_billed1     = t("BILLED #1",     "BILLED") #
    t_billed2     = t("BILLED #2",     "BILLED") #
    t_reopen1     = t("REOPEN #1",     "REOPEN") #
    t_reopen2     = t("REOPEN #2",     "REOPEN") #
    t_reopen3     = t("REOPEN #3",     "REOPEN") #
    t_reject      = t("REJECT",        "REJECT") #
    t_storno1     = t("STORNO #1",     "STORNO") #
    t_storno2     = t("STORNO #2",     "STORNO") #
    t_set_status  = t("SET STATUS",    "SET STATUS") #
    t_fin         = t("FIN",           "FIN") #
    t_end         = t("CHANGE END",           "CHANGE END") #
    t_delete1     = t("DELETE #1",     "DELETE") #
    t_delete2     = t("DELETE #2",     "DELETE") #
    t_delete3     = t("DELETE #3",     "DELETE") #
    t_empty       = t("EMPTY",         "EMPTY") #

    # Silent transitions
    t_tau0 = t("tau0", None)
    t_tau1 = t("tau1", None)
    t_tau2 = t("tau2", None)
    t_tau3 = t("tau3", None)
    t_tau4 = t("tau4", None)
    t_tau5 = t("tau5", None)
    t_tau6 = t("tau6", None)
    t_tau7 = t("tau7", None)
    t_tau8 = t("tau8", None)
    t_tau9 = t("tau9", None)
    t_tau10 = t("tau10", None)
    t_tau11 = t("tau11", None)
    t_tau12 = t("tau12", None)
    t_tau13 = t("tau13", None)
    t_tau14 = t("tau14", None)
    t_tau15 = t("tau15", None)
    

    A = add_arc_from_to

    # Until p6
    A(source, t_new, net);          A(t_new, p1, net)
    A(p1, t_tau7, net);          
    A(t_tau7, p2, net);             A(t_tau7, p3, net)
    
    A(p2, t_change_diagn, net);     A(t_change_diagn, p4, net)
    A(p2, t_tau0, net);             A(t_tau0, p4, net)
    
    A(p3, t_end, net);            A(t_end, p5, net)
    A(p3, t_tau8, net);           A(t_tau8, p5, net)
    
    A(p4,t_tau9, net);          A(p5, t_tau9, net)
    A(t_tau9, p6, net)
    
    #p6 to sink and p7
    A(p6, t_delete3, net);          A(t_delete3, sink, net)
    A(p6, t_tau1, net);            A(t_tau1, sink, net)
    A(p6, t_fin, net);             A(t_fin, p7, net)
    
    #p7
    A(p7, t_tau2, net);          A(t_tau2, sink, net)
    
    A(p7, t_release, net);          A(t_release, p8, net)
    
    A(p7, t_delete1, net);          A(t_delete1, p14, net)
    
    A(p7, t_reopen1, net);          A(t_reopen1, p14, net)
    
    #p8
    A(p8, t_code_ok, net);           A(t_code_ok, p9, net)
    A(p8, t_code_nok, net);          A(t_code_nok, p9, net)
    A(p8, t_tau6, net);             A(t_tau6, p9, net)
    
    #p9
    A(p9, t_set_status, net);        A(t_set_status, p10, net)
    A(p9, t_tau10, net);             A(t_tau10, p10, net)
    
    #p10
    A(p10, t_billed1, net);          A(t_billed1, p11, net)
    A(p10, t_tau3, net);             A(t_tau3, sink, net)
    A(p10, t_reopen2, net);          A(t_reopen2, p14, net)
    A(p10, t_tau4, net);             A(t_tau4, p14, net)
    
    #p11
    A(p11, t_tau11, net);             A(t_tau11, p15, net)
    A(p11, t_storno1, net);          A(t_storno1, p12, net)
    A(p11, t_storno2, net);          A(t_storno2, p15, net)
    
    #p12
    A(p12, t_reject, net);           A(t_reject, p15, net)
    
    #p15
    A(p15, t_billed2, net);          A(t_billed2, p13, net)
    A(p15, t_reopen3, net);          A(t_reopen3, p13, net)
    A(p15, t_tau12, net);            A(t_tau12, p13, net)
    
    #p13
    A(p13, t_tau13, net);            A(t_tau13, p11, net)
    A(p13, t_tau14, net);            A(t_tau14, p14, net)
    
    #p14
    A(p14, t_tau15, net);            A(t_tau15, sink, net)
    A(p14, t_delete2, net);          A(t_delete2, sink, net)
    A(p14, t_empty, net);            A(t_empty, sink, net)
    A(p14, t_tau5, net);             A(t_tau5, p6, net)
    
    

    
    """
    # p1: CHANGE DIAGN (caseType=B) or tau0 (caseType≠B)
    A(p1, t_change_diagn, net);     A(t_change_diagn, p2, net)
    A(p1, t_tau0, net);             A(t_tau0, p2, net)

    # p2: coding loop - CHANGE or CODE OK/NOK
    A(p2, t_change, net);           A(t_change, p2, net)
    A(p2, t_code_ok, net);          A(t_code_ok, p3, net)
    A(p2, t_code_nok, net);         A(t_code_nok, p3, net)

    # p3: speciality split - tau1 (speciality=K → FIN path) or tau2 (speciality≠K)
    A(p3, t_tau1, net);             A(t_tau1, p4, net)
    A(p3, t_tau2, net);             A(t_tau2, p5, net)

    # p4: FIN path (speciality=K)
    A(p4, t_fin, net);              A(t_fin, p5, net)

    # p5: RELEASE
    A(p5, t_release, net);          A(t_release, p6, net)

    # p6: closeCode split
    A(p6, t_tau3, net);             A(t_tau3, p7, net)
    A(p6, t_reopen1, net);          A(t_reopen1, p2, net)

    # p7: billing split
    A(p7, t_billed1, net);          A(t_billed1, p8, net)
    A(p7, t_reject, net);           A(t_reject, p2, net)
    A(p7, t_tau6, net);             A(t_tau6, p2, net)

    # p8: post-billing split
    A(p8, t_storno1, net);          A(t_storno1, sink, net)
    A(p8, t_storno2, net);          A(t_storno2, sink, net)
    A(p8, t_billed2, net);          A(t_billed2, p8, net)
    A(p8, t_reopen2, net);          A(t_reopen2, p2, net)
    A(p8, t_reopen3, net);          A(t_reopen3, p2, net)
    A(p8, t_tau4, net);             A(t_tau4, sink, net)
    A(p8, t_tau5, net);             A(t_tau5, sink, net)
    A(p8, t_delete2, net);          A(t_delete2, sink, net)
    A(p8, t_end, net);              A(t_end, sink, net)
    A(p8, t_set_status, net);       A(t_set_status, p8, net)

    # DELETE #1 and EMPTY come from earlier in the flow
    A(p2, t_delete1, net);          A(t_delete1, sink, net)
    A(p2, t_empty, net);            A(t_empty, sink, net)
    """

    im = Marking({source: 1})
    fm = Marking({sink:   1})
    return net, im, fm


# ================================================================
# 2.  No delay preprocessing needed for Hospital Billing
# ================================================================

def preprocess_delays(log):
    return log   # no time-based guards in this DPN


# ================================================================
# 3.  Per-trace state
# ================================================================

def initial_state(trace):
    return {
        "caseType":   None,
        "speciality": None,
        "closeCode":  None,
        "isClosed":   None,
    }

def update_state(state, activity, event):
    """
    Update per-trace state after each synchronous move.
    Attribute name assumptions match the Hospital Billing XES log.
    Requires Python >= 3.10.
    """
    def _s(k, dfl=None):
        v = event.get(k, dfl)
        return str(v).strip() if v is not None else dfl
    def _b(k):
        v = event.get(k)
        if v is None: return None
        if isinstance(v, bool): return v
        return str(v).strip().lower() in ("true", "1", "yes")

    match activity:

        case "NEW":
            state["caseType"]   = _s("caseType")
            state["speciality"] = _s("speciality")
            state["isClosed"]   = _b("isClosed")

        case "FIN":
            v = _s("closeCode")
            if v is not None:
                state["closeCode"] = v
        case _:
            pass


# ================================================================
# 4.  Guards  (verbatim from Figure 15.3b)
#     Callable: (state, event) -> bool
# ================================================================

def _ct(s):  return s.get("caseType")   or ""
def _sp(s):  return s.get("speciality") or ""
def _cc(s):  return s.get("closeCode")  or ""
def _ic(s):  return s.get("isClosed")

GUARDS = {
    "CHANGE DIAGN":
        lambda s, e: _ct(s) == "B",
    "tau0":
        lambda s, e: _ct(s) != "B",
    "tau1":
        lambda s, e: _sp(s) == "K",
    "FIN":
        lambda s, e: _sp(s) != "K",
    "tau2":
        lambda s, e: _cc(s) == "H",
    "REOPEN #1":
        lambda s, e: _cc(s) != "H",
    "tau6":
        lambda s, e: (_ct(s) == "F"
                      or (_ct(s) != "F" and _ct(s) != "C" and _cc(s) != "A")),
    "CODE NOK":
        lambda s, e: ((_ct(s) != "F" and _ct(s) == "C")
                      or (_ct(s) != "F" and _ct(s) != "C" and _cc(s) == "A")),
    "REOPEN #2":
        lambda s, e: (_ct(s) == "B"
                      or (_ct(s) != "B" and _cc(s) != "A")),
    "tau4":
        lambda s, e: _ct(s) != "B" and _cc(s) == "A",
    "STORNO #1":
        lambda s, e: (_ic(s) == True and _cc(s) != "A") or _ic(s) != True,
    "STORNO #2":
        lambda s, e: _ic(s) == True and _cc(s) == "A",
    "BILLED #1":
        lambda s, e: _ic(s) == True,
    "REOPEN #3":
        lambda s, e: _ic(s) != True,
    "DELETE #2":
        lambda s, e: _ic(s) != True,
    "tau5":
        lambda s, e: _ic(s) == True,
}

def data_enabled(label, state, event):
    g = GUARDS.get(label)
    return g(state, event) if g else True

def guard_aware_enabled(cf_enabled, state, event):
    return {a for a in cf_enabled if data_enabled(a, state, event)}


# ================================================================
# 5.  Main function
# ================================================================

def generate_normative_log(
        path_to_log,
        enabled_activities_name="enabled_activities",
        pnml_path=None):
    """
    Returns an annotated EventLog for the Hospital Billing process.
    Traces with control-flow violations or data guard violations are discarded.
    Each retained event is annotated with guard-aware enabled activities.
    """
    net, im, fm = build_normative_net()
    #pm4py.view_petri_net(net, im, fm)
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

    print(f"Processing {n} traces ...")
    for trace_idx, trace in enumerate(log):
        print(f"  Trace {trace_idx + 1}/{n}")

        trace = copy.deepcopy(trace)
        node  = frozenset({frozenset({0})})
        idx   = 0
        discard_cf    = False
        discard_guard = False
        state = initial_state(trace)

        alignment_result = alignment.apply(trace, net, im, fm, parameters=parameters)
        if alignment_result["fitness"] < 1.0:
            stats["cf_discarded"] += 1
            continue

        for aligned_event in alignment_result["alignment"]:
            move_log   = aligned_event[0][0]
            move_model = aligned_event[0][1]
            trans_name = aligned_event[1][1]

            if move_log == ">>":
                # Model-only move: advance DFA state
                if trans_name is not None:
                    node = get_node_from_transition(drg, node, move_model)

            elif move_model == ">>":
                # Log-only move: control-flow violation
                discard_cf = True
                break

            else:
                # Synchronous move
                event    = trace[idx]
                executed = event.get("concept:name", "")

                # Guard check (state BEFORE this event fires)
                if not data_enabled(trans_name or executed, state, event):
                    discard_guard = True
                    break

                # Annotate with guard-aware enabled activities
                cf_enabled = drg.nodes[node]["enabled_activities"]
                da_enabled = guard_aware_enabled(cf_enabled, state, event)
                da_enabled.add(executed)
                event[enabled_activities_name] = ", ".join(sorted(da_enabled))

                # Advance DFA state
                node = get_node_from_transition(drg, node, move_model)

                # Update variable state
                update_state(state, executed, event)

                idx += 1

        if discard_cf:
            stats["cf_discarded"] += 1
        elif discard_guard:
            stats["guard_discarded"] += 1
        else:
            trace.attributes["concept:name"] = str(stats["kept"] + 1)
            annotated_log.append(trace)
            stats["kept"] += 1

    print(f"\nResults ({n} traces):")
    print(f"  Kept                    : {stats['kept']}")
    print(f"  Discarded (control-flow): {stats['cf_discarded']}")
    print(f"  Discarded (data guards) : {stats['guard_discarded']}")
    return annotated_log

if __name__ == "__main__":
    path_to_log = r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\original_datasets\Hospital Billing - Event Log.xes.gz"
    annotated_log = generate_normative_log(path_to_log, enabled_activities_name="enabled_activities")
    # Check correctness: Is the excecuted activity always included in the enabled activities?
    for trace in annotated_log:
        for i in range(len(trace)):
            event = trace[i]
            executed_activity = event["concept:name"]
            enabled_activities = set(event["enabled_activities"].split(', '))
            if executed_activity not in enabled_activities:
                print(f"Error: Executed activity '{executed_activity}' is not in the set of enabled activities {enabled_activities} for event {i} in trace {trace.attributes['concept:name']}")
    output_log_path = r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\translucent_datasets\hospital_billing\normative_log.csv"
    df_log = pm4py.convert_to_dataframe(annotated_log)
    df_log = replace_spaces_in_activity_names(df_log, enabled_activities_name="enabled_activities")
    df_log.to_csv(output_log_path, index=False)
    print(f'Generated normative log at {output_log_path}')