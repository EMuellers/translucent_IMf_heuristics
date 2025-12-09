from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.dfg.obj import DFG
from translucent_discovery.translucent_inductive_miner.translucent_datatype import TCL


def get_translucent_trace_variants(event_log: EventLog, enabled_activities_key="enabled_activities"
                                   ) -> dict[tuple[tuple[str, frozenset[str]], ...], tuple[Trace, list[int]]]:
    variants: dict[tuple[tuple[str, str], ...], tuple[Trace, list[int]]] = {}
    for idx, trace in enumerate(event_log):
        variant = tuple((event["concept:name"], event[enabled_activities_key]) for event in trace)
        if variant not in variants:
            variants[variant] = (trace, [idx])
        else:
            variants[variant][1].append(idx)
    return variants

def get_delta_arcs(tDFG: DFG, DFG: DFG) -> set[tuple[str, str]]:
    """
    This method recieves the tDFG and the DFG and returns the arcs in the tDFG that are not in the DFG
    """
    delta_arcs = set()
    for arc in tDFG.graph:
        if arc not in DFG.graph:
            delta_arcs.add(arc)
    return delta_arcs

def get_sorted_delta_arcs(delta_arcs, obj, criterion = "dependency_score"):
    """
    Returns the delta arcs sorted according to the given criterion. The list is sorted in ascending order (worst to best).
    """
    match criterion:
        case "dependency_score":
            return sorted(_calculate_dependency_scores(delta_arcs, obj), key=lambda x: x[1])
        case _:
            raise NotImplementedError(f"Sorting criterion {criterion} not implemented.")
    

def _calculate_dependency_scores_old(delta_arcs, obj):
    """
    Returns the dependency scores for the given arcs. Calculated based on the translucent directly follows relationships.
    #TODO Übersehe ich was?
    """
    delta_activities = {act for arc in delta_arcs for act in arc}
    translucent_dfr = get_directly_follow_relationships_frequent(obj.log, delta_activities)
    for arc in delta_arcs:
        if arc[0] != arc[1]:
            freq_forward = translucent_dfr.get(arc, 0)
            freq_backward = translucent_dfr.get((arc[1], arc[0]), 0)
            score = (freq_forward - freq_backward) / (freq_forward + freq_backward + 1)
            yield (arc, score)
        else:
            freq = translucent_dfr.get(arc, 0)
            score = freq / (freq + 1)
            yield (arc, score)


# Adjusted for tcl logs
def _calculate_dependency_scores(delta_arcs, obj):
    """
    Returns the dependency scores for the given arcs. Calculated based on the translucent directly follows relationships.
    """
    delta_activities = {act for arc in delta_arcs for act in arc}
    translucent_dfr = get_directly_follow_relationships_frequent_tcl(obj.tcl, delta_activities)
    for arc in delta_arcs:
        if arc[0] != arc[1]:
            freq_forward = translucent_dfr.get(arc, 0)
            freq_backward = translucent_dfr.get((arc[1], arc[0]), 0)
            score = (freq_forward - freq_backward) / (freq_forward + freq_backward + 1)
            yield (arc, score)
        else:
            freq = translucent_dfr.get(arc, 0)
            score = freq / (freq + 1)
            yield (arc, score)
    
def get_directly_follow_relationships_frequent(log, executed_activities, enabled_activities_key="enabled_activities") -> dict:
    activity_follow = {}
    variants = get_translucent_trace_variants(log)
    for variant in variants:
        number_of_occurrence = len(variants[variant][1])
        trace = variants[variant][0]
        for index, current_event in enumerate(trace):
            if index < len(trace)-1:
                executed_activity = current_event["concept:name"]
                enabled_activities_next = [el.strip() for el in trace[index+1][enabled_activities_key].split(",") if el.strip() in executed_activities]
                for next_activity in enabled_activities_next:
                    if (executed_activity, next_activity) not in activity_follow:
                        activity_follow[(executed_activity, next_activity)] = 0
                    activity_follow[(executed_activity, next_activity)] += number_of_occurrence
    return activity_follow

# get_directly_follow_relationships_frequent for tcl logs
def get_directly_follow_relationships_frequent_tcl(log: TCL, executed_activities) -> dict:
    activity_follow = {}
    for trace in log:
        number_of_occurrence = log[trace]
        for index, current_event in enumerate(trace):
            if index < len(trace)-1:
                executed_activity = current_event[0]
                enabled_activities_next = trace[index+1][1].intersection(executed_activities)
                for next_activity in enabled_activities_next:
                    if (executed_activity, next_activity) not in activity_follow:
                        activity_follow[(executed_activity, next_activity)] = 0
                    activity_follow[(executed_activity, next_activity)] += number_of_occurrence
    return activity_follow