import copy

from utils.DirectReachabilityGraph import DirectReachabilityGraph
from utils.ReachabilityGraph import ReachabilityGraph
from pm4py.algo.conformance.alignments.petri_net.variants import state_equation_a_star as star
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignment
import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignment
from evaluation.modify_logs import add_start_and_end_translucent
from pm4py.objects.log.obj import EventLog


def get_node_from_transition(graph, node, transition):
    edges = graph.out_edges(node, data= True, keys=True)
    for edge in edges:
        if graph.edges[edge[0], edge[1], edge[2]]["transition"] == transition:
            return edge[1]
    return None


def create_translucent_event_log(log, net, im, fm, enabled_activities_name="enabled_activities"):
    variants = pm4py.statistics.variants.log.get.get_variants(log)
    variant_log = EventLog(attributes=log.attributes, extensions=log.extensions, classifiers=log.classifiers, omni_present=log.omni_present, properties=log.properties)
    variant_counter = {}
    counter = 0
    for variant in variants:
        variant_log.append(variants[variant][0])
        variant_counter[counter] = len(variants[variant])
        counter += 1
    parameters = {star.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
    rg = ReachabilityGraph(net, im, fm, 1)
    drg = DirectReachabilityGraph(rg).dfa
    global_case_id_counter = 1
    filtered_log = EventLog(attributes=log.attributes, extensions=log.extensions, classifiers=log.classifiers, omni_present=log.omni_present, properties=log.properties)
    for index, trace in enumerate(variant_log):
        node = frozenset({frozenset({0})})
        alignment_result = alignment.apply(trace, net, im, fm, parameters=parameters)
        if alignment_result["fitness"] == 1.0:
            index_in_real_trace = 0
            for aligned_event in alignment_result["alignment"]:
                if aligned_event[1][1] is not None:
                    enabled_activities = drg.nodes[node]["enabled_activities"]
                    trace[index_in_real_trace][enabled_activities_name] = ', '.join(enabled_activities)
                    node = get_node_from_transition(drg, node, aligned_event[0][1])
                    index_in_real_trace += 1
            trace_to_append = copy.deepcopy(trace)
            counter = 0
            while counter < variant_counter[index]:
                trace_to_append._set_attributes({"concept:name": str(global_case_id_counter)})
                filtered_log.append(copy.deepcopy(trace_to_append))
                counter +=1
                global_case_id_counter +=1
    return filtered_log


def create_logs(ground_truth):
    results = []
    variants = pm4py.statistics.variants.log.get.get_variants(ground_truth)
    i = 1
    while i <= len(variants):
        variants_to_consider = pm4py.filter_variants_top_k(ground_truth, i)
        dataframe = pm4py.convert_to_dataframe(variants_to_consider)
        results.append(dataframe)
        i += 1
    return results

