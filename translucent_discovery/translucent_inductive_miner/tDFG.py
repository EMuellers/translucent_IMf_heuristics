import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.dfg.obj import DFG
import pm4py
from translucent_discovery.utils.translucent_activity_relationships import get_parallel_relationships, get_directly_follow_relationships, get_start_activities, get_end_activities
from translucent_discovery.utils.translucent_activity_relationships import get_parallel_relationships_frequent, get_directly_follow_relationships_frequent, get_choice_relationships_frequent, get_start_activities_frequent, get_end_activities_frequent



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
    for act in end_activities:
        dfg.end_activities.update({act: 1})
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
    for (source, target) in parallel:
        count = parallel[(source, target)]
        if subtract_xor:
            xor_count = 0
            if (source, target) in xor:
                xor_count = xor[(source, target)]
            if count-xor_count > 0:
                dfg.graph.update({(source, target): count-xor_count})
        else:
            dfg.graph.update({(source, target): count})
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
    for act in end_activities:
        dfg.end_activities.update({act: end_activities[act]})
    return dfg
