from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.dfg.obj import DFG
from translucent_discovery.translucent_inductive_miner.translucent_datatype import TCL, get_executed_activity_frequencies_tcl


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
    Returns the delta arcs sorted according to the given criterion. The list is sorted in ascending order (worst to best) regarding the specified criterion.
    """
    match criterion:
        case "dependency_score":
            return sorted(_calculate_dependency_scores(delta_arcs, obj), key=lambda x: x[1])
        case "exclusive_choice_frequency":
            return sorted(get_choice_relationships_frequent_tcl(obj.tcl, delta_arcs), key=lambda x: x[1], reverse=True)
        case "confidence":
            return sorted(_calculate_confidence_scores(delta_arcs, obj), key=lambda x: x[1])
        case "support":
            return sorted(_calculate_support_scores(delta_arcs, obj), key=lambda x: x[1])
        case _:
            raise NotImplementedError(f"Sorting criterion {criterion} not implemented.")
    

def _calculate_dependency_scores_old(delta_arcs, obj):
    """
    Returns the dependency scores for the given arcs. Calculated based on the translucent directly follows relationships.
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

def get_choice_relationships_frequent_tcl(log: TCL, delta_arcs) -> dict:
    executed_activities = {act for arc in delta_arcs for act in arc}
    activity_choice = {}
    for trace in log:
        number_of_occurrence = log[trace]
        for index, current_event in enumerate(trace):
            if index < len(trace)-1:
                executed_activity = current_event[0]
                enabled_activities_current = current_event[1].intersection(executed_activities)
                enabled_activities_next = trace[index + 1][1].intersection(executed_activities)
                removed_activities = enabled_activities_current.difference(enabled_activities_next)
                for activity in removed_activities:
                    if (executed_activity, activity) not in activity_choice:
                        activity_choice[(executed_activity, activity)] = 0
                    activity_choice[(executed_activity, activity)] += number_of_occurrence
                    if (activity, executed_activity) not in activity_choice:
                        activity_choice[(activity, executed_activity)] = 0
                    if activity != executed_activity: # Avoid counting self-choice relationships twice #TODO: Ask Harry if this is fine
                        activity_choice[(activity, executed_activity)] += number_of_occurrence
    # filter only delta arcs
    activity_choice = {arc: freq for arc, freq in activity_choice.items() if arc in delta_arcs}
    return activity_choice

def _calculate_confidence_scores(delta_arcs, obj):
    """Calculates confidence scores for the given delta arcs based on the translucent directly follows relationships. """
    activities_denominator = [arc[0] for arc in delta_arcs]
    activity_frequencies = get_executed_activity_frequencies_tcl(obj.tcl)
    delta_activities = {act for arc in delta_arcs for act in arc}
    translucent_dfr = get_directly_follow_relationships_frequent_tcl(obj.tcl, delta_activities)
    for arc in delta_arcs:
        score = translucent_dfr[arc] / activity_frequencies[arc[0]]
        yield (arc, score)

def _calculate_support_scores(delta_arcs, obj):
    """Uses translucent directly follows frequencies as scoring."""
    delta_activities = {act for arc in delta_arcs for act in arc}
    translucent_dfr = get_directly_follow_relationships_frequent_tcl(obj.tcl, delta_activities)
    for arc in delta_arcs:
        score = translucent_dfr[arc]
        yield (arc, score)