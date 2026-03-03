"""
generate_normative_log_sepsis.py

Adapted version of generate_normative_log() for the Sepsis log.
Source: Mannhardt (2018), Section 13.3, Figure 13.6

Guard variables (tracked in state):
  DiagnosticLacticAcid  -- bool, written by ER Triage (or set when LacticAcid fires)
  SIRSCriteria2OrMore   -- bool, written by ER Sepsis Triage
  LacticAcid            -- float, written by LacticAcid event (measurement value)
  Hypotensie            -- bool, written by ER Sepsis Triage or relevant triage event

No time-based delays in this DPN.
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
# 1.  Normative Petri net  (Figure 13.6a)
#
# Topology reading from Figure 13.6a (left to right):
#   source -> ER Registration -> p1
#   p1 -> ER Triage -> p2
#   p2 -> Inv1 (DiagnosticLacticAcid=false) -> p3
#   p2 -> LacticAcid (DiagnosticLacticAcid=true) -> p3
#   p3 -> IV Antibiotics -> p3  (self-loop)
#   p3 -> IV Liquid -> p3       (self-loop)
#   p3 -> CRP #1 -> p3          (self-loop)
#   p3 -> Leucocytes #1 -> p3   (self-loop)
#   p3 -> ER Sepsis Triage -> p4
#   p4 -> Inv2 (SIRSCriteria2OrMore=true)  -> p5
#   p4 -> Inv3 (SIRSCriteria2OrMore=false) -> p5
#   p5 -> Admission NC #1 -> p6
#   p5 -> Admission IC    -> p7
#   p5 -> Inv4             -> p8  (leave hospital)
#   p6 -> Admission IC    -> p7   (NC then IC)
#   p6 -> Transfer NC     -> p8
#   p7 -> Release A       -> p8
#   p7 -> Release B       -> p8
#   p7 -> Release C       -> p8
#   p7 -> Release D       -> p8
#   p7 -> Release E       -> p8
#   p8 -> Return ER       -> p3   (loop back)
#   p8 -> Inv5            -> sink
#   p8 -> CRP #2          -> p8   (self-loop)
#   p8 -> Leucocytes #2   -> p8   (self-loop)
#   p8 -> Admission NC #2 -> p8   (self-loop)
# ================================================================

def build_normative_net():
    net = PetriNet("Sepsis_Normative")

    def p(name):
        pl = PetriNet.Place(name); net.places.add(pl); return pl

    def t(name, label):
        tr = PetriNet.Transition(name, label); net.transitions.add(tr); return tr

    source = p("source")
    p1 = p("p1")
    p2 = p("p2")
    p3 = p("p3")
    p4 = p("p4")
    p5 = p("p5")
    p6 = p("p6")
    p7 = p("p7")
    p8 = p("p8")
    p9 = p("p9")
    p10 = p("p10")
    p11 = p("p11")
    p12 = p("p12")
    p13 = p("p13")
    p14 = p("p14")
    p15 = p("p15")
    p16 = p("p16")
    p17 = p("p17")
    p18 = p("p18")
    p19 = p("p19")
    p20 = p("p20")
    p21 = p("p21")
    p22 = p("p22") # Transfer NC self loop
    sink = p("sink")

    # Visible transitions
    t_er_reg        = t("ER Registration",    "ER Registration") #
    t_er_triage     = t("ER Triage",          "ER Triage") #
    t_lactic_1        = t("LacticAcid #1",      "LacticAcid")#
    t_lactic_2        = t("LacticAcid #2",   "LacticAcid")#
    t_iv_antibiotics= t("IV Antibiotics",     "IV Antibiotics")#
    t_iv_liquid     = t("IV Liquid",          "IV Liquid")#
    t_crp_1         = t("CRP #1",             "CRP")#
    t_crp_2         = t("CRP #2",             "CRP")#
    t_leucocytes_1  = t("Leucocytes #1",      "Leucocytes")#
    t_leucocytes_2  = t("Leucocytes #2",      "Leucocytes")#
    t_sepsis_triage = t("ER Sepsis Triage",   "ER Sepsis Triage") #
    t_admission_nc_1= t("Admission NC #1",    "Admission NC")#
    t_admission_nc_2= t("Admission NC #2",    "Admission NC")#
    t_admission_ic_1= t("Admission IC #1",    "Admission IC")#
    t_admission_ic_2= t("Admission IC #2",       "Admission IC")#
    #t_transfer_nc   = t("Transfer NC",        "Transfer NC") # Does not appear in publically available sepsis log, therefore we leave it out
    t_release_a     = t("Release A",          "Release A")
    t_release_b     = t("Release B",          "Release B")
    t_release_c     = t("Release C",          "Release C")
    t_release_d     = t("Release D",          "Release D")
    t_release_e     = t("Release E",          "Release E")
    t_return_er     = t("Return ER",          "Return ER")

    # Silent transitions
    t_inv1 = t("Inv1", None)
    t_inv2 = t("Inv2", None)
    t_inv3 = t("Inv3", None)
    t_inv4 = t("Inv4", None)
    t_inv5 = t("Inv5", None)
    t_inv6 = t("Inv6", None)
    t_inv7 = t("Inv7", None)
    t_inv8 = t("Inv8", None)
    t_inv9 = t("Inv9", None)
    t_inv10 = t("Inv10", None)
    t_inv11 = t("Inv11", None)
    t_inv12 = t("Inv12", None)
    t_inv13 = t("Inv13", None) # p5 to p8
    t_inv14 = t("Inv14", None) # p15 to p17


    A = add_arc_from_to

    # source
    A(source, t_er_reg, net);        A(t_er_reg, p1, net);  A(t_er_reg, p2, net)
    A(t_er_reg, p3, net);            A(t_er_reg, p12, net)
    
    # p1
    A(p1, t_inv6, net); A(t_inv6, p7, net)
    A(p1, t_crp_1, net); A(t_crp_1, p4, net)
    
    # p2
    A(p2, t_inv7, net); A(t_inv7, p8, net)
    A(p2, t_leucocytes_1, net); A(t_leucocytes_1, p5, net)
    
    # p3
    A(p3, t_er_triage, net); A(t_er_triage, p6, net)
    
    # p4
    A(p4, t_inv8, net); A(t_inv8, p7, net)
    A(p4, t_crp_2, net); A(t_crp_2, p4, net)
    
    # p5
    A(p5, t_inv13, net); A(t_inv13, p8, net)
    A(p5, t_leucocytes_2, net); A(t_leucocytes_2, p5, net)
    
    # p6
    A(p6, t_sepsis_triage, net); A(t_sepsis_triage, p9, net)
    
    # p7
    A(p7, t_inv11, net); A(t_inv11, p20, net)
    
    # p8
    A(p8, t_inv11, net)
    
    # p9
    A(p9, t_inv3, net); A(t_inv3, p16, net)
    A(p9, t_inv2, net); A(t_inv2, p10, net); A(t_inv2, p11, net)
    
    # p10
    A(p10, t_iv_antibiotics, net); A(t_iv_antibiotics, p13, net)
    
    # p11
    A(p11, t_iv_liquid, net); A(t_iv_liquid, p14, net)
    
    # p12
    A(p12, t_lactic_1, net); A(t_lactic_1, p15, net)
    A(p12, t_inv1, net); A(t_inv1, p17, net)
    
    # p13
    A(p13, t_inv9, net); A(t_inv9, p16, net)
    
    # p14
    A(p14, t_inv9, net)
    
    # p15
    A(p15, t_lactic_2, net); A(t_lactic_2, p15, net)
    A(p15, t_inv14, net); A(t_inv14, p17, net)
    
    # p16
    A(p16, t_admission_nc_2, net); A(t_admission_nc_2, p18, net)
    A(p16, t_admission_nc_1, net); A(t_admission_nc_1, p22, net)
    A(p16, t_admission_ic_1, net); A(t_admission_ic_1, p22, net)
    A(p16, t_inv4, net); A(t_inv4, p19, net)
    
    # p17
    A(p17, t_inv11, net)
    
    # p18
    A(p18, t_admission_ic_2, net); A(t_admission_ic_2, p22, net)
    
    # p22
    A(p22, t_inv10, net); A(t_inv10, p19, net)
    
    # p19
    A(p19, t_inv11, net)
    
    # p20
    A(p20, t_release_c, net); A(t_release_c, p21, net)
    A(p20, t_release_b, net); A(t_release_b, p21, net)
    A(p20, t_release_e, net); A(t_release_e, p21, net)
    A(p20, t_release_d, net); A(t_release_d, p21, net)
    A(p20, t_release_a, net); A(t_release_a, p21, net)
    A(p20, t_inv5, net); A(t_inv5, p21, net)
    
    # p21
    A(p21, t_inv12, net); A(t_inv12, sink, net) 
    A(p21, t_return_er, net); A(t_return_er, sink, net)

    
    
    
    
    """
    # Entry
    A(source, t_er_reg, net);        A(t_er_reg, p1, net)
    A(p1, t_er_triage, net);         A(t_er_triage, p2, net)

    # Lactic acid diagnostic split
    A(p2, t_inv1, net);              A(t_inv1, p3, net)
    A(p2, t_lactic, net);            A(t_lactic, p3, net)

    # Diagnostic / treatment self-loops on p3
    A(p3, t_iv_antibiotics, net);    A(t_iv_antibiotics, p3, net)
    A(p3, t_iv_liquid, net);         A(t_iv_liquid, p3, net)
    A(p3, t_crp_1, net);             A(t_crp_1, p3, net)
    A(p3, t_leucocytes_1, net);      A(t_leucocytes_1, p3, net)

    # ER Sepsis Triage
    A(p3, t_sepsis_triage, net);     A(t_sepsis_triage, p4, net)

    # SIRS split
    A(p4, t_inv2, net);              A(t_inv2, p5, net)
    A(p4, t_inv3, net);              A(t_inv3, p5, net)

    # Admission split
    A(p5, t_admission_nc_1, net);    A(t_admission_nc_1, p6, net)
    A(p5, t_admission_ic, net);      A(t_admission_ic, p7, net)
    A(p5, t_inv4, net);              A(t_inv4, p8, net)

    # From p6: NC can escalate to IC, or transfer
    A(p6, t_admission_ic, net);      A(t_admission_ic, p7, net)
    A(p6, t_transfer_nc, net);       A(t_transfer_nc, p8, net)

    # Releases from IC
    A(p7, t_release_a, net);         A(t_release_a, p8, net)
    A(p7, t_release_b, net);         A(t_release_b, p8, net)
    A(p7, t_release_c, net);         A(t_release_c, p8, net)
    A(p7, t_release_d, net);         A(t_release_d, p8, net)
    A(p7, t_release_e, net);         A(t_release_e, p8, net)

    # p8: post-admission / post-release
    A(p8, t_return_er, net);         A(t_return_er, p3, net)
    A(p8, t_inv5, net);              A(t_inv5, sink, net)
    A(p8, t_crp_2, net);             A(t_crp_2, p8, net)
    A(p8, t_leucocytes_2, net);      A(t_leucocytes_2, p8, net)
    A(p8, t_admission_nc_2, net);    A(t_admission_nc_2, p8, net)
    """
    im = Marking({source: 1})
    fm = Marking({sink:   1})
    return net, im, fm


# ================================================================
# 2.  No time-based delay preprocessing needed
# ================================================================

def preprocess_delays(log):
    return log


# ================================================================
# 3.  Per-trace state
# ================================================================

def initial_state(trace):
    return {
        "DiagnosticLacticAcid": None,   # bool: set by ER Triage
        "SIRSCriteria2OrMore":  None,   # bool: set by ER Sepsis Triage
        "LacticAcid":           0.0,    # float: updated by LacticAcid event
        "Hypotensie":           None,   # bool: set by ER Sepsis Triage
    }

def update_state(state, activity, event):
    """
    Update per-trace state after each synchronous move.
    Attribute name assumptions match the Sepsis XES log.
    Requires Python >= 3.10.
    """
    def _b(k):
        v = event.get(k)
        if v is None: return None
        if isinstance(v, bool): return v
        return str(v).strip().lower() in ("true", "1", "yes")
    def _f(k, dfl=0.0):
        try: return float(event.get(k, dfl))
        except (TypeError, ValueError): return dfl

    match activity:

        case "ER Triage":
            # DiagnosticLacticAcid flag is determined at triage
            b = _b("DiagnosticLacticAcid")
            if b is not None:
                state["DiagnosticLacticAcid"] = b

        case "LacticAcid":
            # LacticAcid measurement value
            state["LacticAcid"] = _f("LacticAcid")
            # Firing this transition implies DiagnosticLacticAcid was true
            state["DiagnosticLacticAcid"] = True

        case "ER Sepsis Triage":
            b = _b("SIRSCriteria2OrMore")
            if b is not None:
                state["SIRSCriteria2OrMore"] = b
            h = _b("Hypotensie")
            if h is not None:
                state["Hypotensie"] = h

        case _:
            pass


# ================================================================
# 4.  Guards  (verbatim from Figure 13.6b)
#     Keyed by internal transition name.
#     Callable: (state, event) -> bool
# ================================================================

def _dla(s): return s.get("DiagnosticLacticAcid")
def _sirs(s): return s.get("SIRSCriteria2OrMore")
def _la(s):  return s.get("LacticAcid", 0.0)
def _hyp(s): return s.get("Hypotensie")

GUARDS = {
    "Inv1":
        lambda s, e: _dla(s) == False,
    "LacticAcid":
        lambda s, e: _dla(s) == True,
    "Inv2":
        lambda s, e: _sirs(s) == True,
    "Inv3":
        lambda s, e: _sirs(s) == False,
    "Admission NC #1":
        lambda s, e: _la(s) > 0,
    "Admission IC":
        lambda s, e: _la(s) > 0 and _hyp(s) == True,
    "Inv4":
        lambda s, e: (_la(s) > 0 and _hyp(s) == False) or _la(s) <= 0,
    "Inv5":
        lambda s, e: _la(s) <= 0,
    "Release A":
        lambda s, e: _la(s) > 0,
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
    Returns an annotated EventLog for the Sepsis process.
    Traces with control-flow violations or data guard violations are discarded.
    Each retained event is annotated with guard-aware enabled activities.
    """
    net, im, fm = build_normative_net()
    #pm4py.view_petri_net(net, im, fm)

    log = pm4py.read_xes(path_to_log, return_legacy_log_object=True)
    #log = clean_activity_names(log)
    log = preprocess_delays(log)

    
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
                # Model-only move: advance DFA state (skip tau)
                if trans_name is not None:
                    node = get_node_from_transition(drg, node, move_model)

            elif move_model == ">>":
                # Log-only move: control-flow violation, discard trace
                discard_cf = True
                break

            else:
                # Synchronous move
                event    = trace[idx]
                executed = event.get("concept:name", "")

                # Guard check using internal transition name (state BEFORE update)
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

                # Update variable state (AFTER guard check and annotation)
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
    path_to_log = r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\original_datasets\Sepsis Cases - Event Log.xes.gz"
    annotated_log = generate_normative_log(path_to_log, enabled_activities_name="enabled_activities")
    output_log_path = r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\translucent_datasets\Sepsis\normative_log.csv"
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