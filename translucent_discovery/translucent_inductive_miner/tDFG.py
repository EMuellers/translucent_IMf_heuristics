from networkx import edges
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.dfg.obj import DFG
import pm4py
from translucent_discovery.utils.translucent_activity_relationships import get_parallel_relationships, get_directly_follow_relationships, get_start_activities, get_end_activities
from translucent_discovery.utils.translucent_activity_relationships import get_parallel_relationships_tcl, get_directly_follow_relationships_tcl, get_start_activities_tcl, get_end_activities_tcl
from translucent_discovery.utils.translucent_activity_relationships import get_parallel_relationships_frequent, get_directly_follow_relationships_frequent, get_choice_relationships_frequent, get_start_activities_frequent, get_end_activities_frequent
from translucent_discovery.utils.translucent_activity_relationships import get_parallel_relationships_frequent_tcl, get_choice_relationships_frequent_tcl, get_directly_follow_relationships_frequent_tcl, get_start_activities_frequent_tcl, get_end_activities_frequent_tcl
from translucent_discovery.translucent_inductive_miner.translucent_datatype import TCL, get_executed_activities


def discover_dfg(log, parameters={}) -> DFG:
    if isinstance(log, pd.DataFrame):
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
    dfg = DFG()
    executed_activities = set()
    variants = pm4py.statistics.variants.log.get.get_variants(log)
    for variant in variants:
        trace = variants[variant][0]
        for event in trace:
            executed_activities.add(event["concept:name"])
    parallel = get_parallel_relationships(log, executed_activities)
    for source in parallel:
        for target in parallel[source]:
            dfg.graph.update({(source, target): 1})
    directly_follow = get_directly_follow_relationships(log, executed_activities)
    for source in directly_follow:
        for target in directly_follow[source]:
            dfg.graph.update({(source, target): 1})
    start_activities = get_start_activities(log, executed_activities)
    for act in start_activities:
        dfg.start_activities.update(({act: 1}))
    end_activities = get_end_activities(log, executed_activities, strict_end_activities=parameters.get("strict_end_activities", False))
    # Heuristic: If two activities are in translucent parallel relation and one is an end activity, the other is also considered an end activity
    if parameters.get("parallel_end_activities_heuristic", False):
        edges = set()
        for source, targets in parallel.items():
            for target in targets:
                if source != target: # Exclude self-loops
                    edges.add(tuple(sorted((source, target))))
        for (source, target) in edges:
            if source in end_activities and target not in end_activities:
                end_activities.add(target)
            if target in end_activities and source not in end_activities:
                end_activities.add(source)
    for act in end_activities:
        dfg.end_activities.update({act: 1})
    return dfg


def discover_dfg_tcl(log: TCL, parameters={}, self_loops=None) -> DFG:
    dfg = DFG()
    executed_activities = get_executed_activities(log)
    parallel = get_parallel_relationships_tcl(log, executed_activities)
    for source in parallel:
        for target in parallel[source]:
            dfg.graph.update({(source, target): 1})
    directly_follow = get_directly_follow_relationships_tcl(log, executed_activities)
    for source in directly_follow:
        for target in directly_follow[source]:
            dfg.graph.update({(source, target): 1})
    start_activities = get_start_activities_tcl(log, executed_activities)
    for act in start_activities:
        dfg.start_activities.update(({act: 1}))
    end_activities = get_end_activities_tcl(log, executed_activities, strict_end_activities=parameters.get("strict_end_activities", False))
    # Heuristic: If two activities are in translucent parallel relation and one is an end activity, the other is also considered an end activity
    if parameters.get("parallel_end_activities_heuristic", False):
        edges = set()
        for source, targets in parallel.items():
            for target in targets:
                if source != target: # Exclude self-loops (shouldn't be in there anyway but just to be sure)
                    edges.add(tuple(sorted((source, target))))
        for (source, target) in edges:
            if source in end_activities and target not in end_activities:
                end_activities.add(target)
            if target in end_activities and source not in end_activities:
                end_activities.add(source)
    for act in end_activities:
        dfg.end_activities.update({act: 1})
    if parameters.get("translucent_self_loops", False) and len(executed_activities) == 1:
        if self_loops[list(executed_activities)[0]] > 0:
            dfg.update({(list(executed_activities)[0], list(executed_activities)[0]): 1}) # only one activity executed and non-frequent dfg
    return dfg

#entspricht comut.discover_dfg_uvcl in pm4py
def discover_frequent_dfg(log, subtract_xor=True, parameters={}) -> DFG:
    if isinstance(log, pd.DataFrame):
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
    dfg = DFG()
    executed_activities = set()
    variants = pm4py.statistics.variants.log.get.get_variants(log) #Elias: Only used for executed activities, can stay this way
    for variant in variants:
        trace = variants[variant][0]
        for event in trace:
            executed_activities.add(event["concept:name"])
    parallel = get_parallel_relationships_frequent(log, executed_activities)
    xor = get_choice_relationships_frequent(log, executed_activities)
    directly_follow = get_directly_follow_relationships_frequent(log, executed_activities)
    for (source, target) in directly_follow:
        count = directly_follow[(source, target)]
        if subtract_xor:
            xor_count = 0
            if (source, target) in xor:
                xor_count = xor[(source, target)]
            if count-xor_count > 0:
                dfg.graph.update({(source, target): count - xor_count})
        else:
            dfg.graph.update({(source, target): count})
    start_activities = get_start_activities_frequent(log, executed_activities)
    for act in start_activities:
        dfg.start_activities.update(({act: start_activities[act]}))
    end_activities = get_end_activities_frequent(log, executed_activities, strict_end_activities=parameters.get("strict_end_activities", False))
    # Heuristic: If two activities are in translucent parallel relation and one is an end activity, the other is also considered an end activity
    if parameters.get("parallel_end_activities_heuristic", False):
        added_parallel_arcs = set()
        for (source, target) in parallel:
            count = parallel[(source, target)]
            if subtract_xor:
                xor_count = 0
                if (source, target) in xor:
                    xor_count = xor[(source, target)]
                if count-xor_count > 0:
                    dfg.graph.update({(source, target): count-xor_count})
                    added_parallel_arcs.add((source, target))
            else:
                dfg.graph.update({(source, target): count})
                added_parallel_arcs.add((source, target))
        for (source, target) in added_parallel_arcs:
            if source != target: # Exclude self-loops
                if source in end_activities and target not in end_activities:
                    end_activities.update({target: end_activities[source]})
                if target in end_activities and source not in end_activities:
                    end_activities.update({source: end_activities[target]})
    for act in end_activities:
        dfg.end_activities.update({act: end_activities[act]})
    return dfg

def discover_frequent_dfg_tcl(log: TCL, subtract_xor=True, parameters={}, self_loops=None) -> DFG:
    dfg = DFG()
    executed_activities = get_executed_activities(log)
    parallel = get_parallel_relationships_frequent_tcl(log, executed_activities)
    xor = get_choice_relationships_frequent_tcl(log, executed_activities)
    directly_follow = get_directly_follow_relationships_frequent_tcl(log, executed_activities)
    for (source, target) in directly_follow:
        count = directly_follow[(source, target)]
        if subtract_xor:
            xor_count = 0
            if (source, target) in xor:
                xor_count = xor[(source, target)]
            if count-xor_count > 0:
                dfg.graph.update({(source, target): count - xor_count})
        else:
            dfg.graph.update({(source, target): count})
    start_activities = get_start_activities_frequent_tcl(log, executed_activities)
    for act in start_activities:
        dfg.start_activities.update(({act: start_activities[act]}))
    end_activities = get_end_activities_frequent_tcl(log, executed_activities, strict_end_activities=parameters.get("strict_end_activities", False))
    # Heuristic: If two activities are in translucent parallel relation and one is an end activity, the other is also considered an end activity
    if parameters.get("parallel_end_activities_heuristic", False):
        added_parallel_arcs = set()
        for (source, target) in parallel:
            count = parallel[(source, target)]
            if subtract_xor:
                xor_count = 0
                if (source, target) in xor:
                    xor_count = xor[(source, target)]
                if count-xor_count > 0:
                    dfg.graph.update({(source, target): count-xor_count})
                    added_parallel_arcs.add((source, target))
            else:
                dfg.graph.update({(source, target): count})
                added_parallel_arcs.add((source, target))
        for (source, target) in added_parallel_arcs:
            if source != target: # Exclude self-loops
                if source in end_activities and target not in end_activities:
                    end_activities.update({target: end_activities[source]})
                if target in end_activities and source not in end_activities:
                    end_activities.update({source: end_activities[target]})
    for act in end_activities:
        dfg.end_activities.update({act: end_activities[act]})
    if parameters.get("translucent_self_loops", False) and len(executed_activities) == 1:
        if self_loops[list(executed_activities)[0]] > 0:
            dfg.update({(list(executed_activities)[0], list(executed_activities)[0]): 1}) # only one activity executed
    """ #TODO: Decide how to handle this. Probably not needed, as cut for this heuristic will be found in the non filtered tdfg anyways
    if parameters.get("translucent_self_loops", False) and len(executed_activities) == 1:
        if self_loops[list(executed_activities)[0]] > 0:
            dfg.update({(list(executed_activities)[0], list(executed_activities)[0]): self_loops[list(executed_activities)[0]]}) # only one activity executed and non-frequent dfg
    """
    return dfg

#TODO: Decide with Harry whether to use this or the other approach (see notes)

def check_translucent_self_loop_with_filtering(log: TCL, activity: str, frequency_threshold: int, translucent_self_loop_frequency: int) -> bool:
    """
    Checks whether self-loops in the given log plus translucent self-loops from the initial log are above the frequency threshold * end activity frequency.
    :param log: log only containing a single activity
    :type log: TCL
    :param activity: activity considered for self_loop
    :type activity: str
    :param frequency_threshold: Inductive Miner frequency threshold
    :type frequency_threshold: int
    :param translucent_self_loop_frequency: Self-loop frequency from initial log
    :type translucent_self_loop_frequency: int
    :return: True if translucent self-loops are above threshold
    :rtype: bool
    """
    end_frequency = get_end_activities_frequent_tcl(log, {activity}, strict_end_activities=True)
    df_frequency = get_directly_follow_relationships_frequent_tcl(log, {activity})
    if activity in end_frequency and activity in df_frequency:
        total_self_loops = translucent_self_loop_frequency + df_frequency[(activity, activity)]
        if total_self_loops >= frequency_threshold * end_frequency[activity]:
            return True # Means add the self-loop arc
    return False
