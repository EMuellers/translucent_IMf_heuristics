"""
Script that generates a translucent log with noise by
1. discovering a ground truth model with the IMf
2. Aligns every trace of the original dataset to the discovered model to get the enabled activities:
    2.1 Create optimal alignment
        2.2.1 For synchronous moves, get all enabled activities from the model at that point
        2.2.2 For others first perform all log only moves before advancing in the model
"""
import random
import copy

import pm4py
from pm4py.objects.log.obj import EventLog
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


"""
Generate a translucent log with noise based on the given parameters.
Expects a path to a xes log as input.
"""
def generate_log_with_noise(path_to_log, noise_threshold, alignment_parameters, enabled_activities_name="enabled_activities"):
    log = pm4py.read_xes(path_to_log, return_legacy_log_object=True)
    
    # First discover a ground truth model with the IMf
    net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=noise_threshold)
    
    # Debug: Print the discovered model
    #pm4py.view_petri_net(net, im, fm)
    
    rg = ReachabilityGraph(net, im, fm, 1)
    drg = DirectReachabilityGraph(rg).dfa
    
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
    annotated_log = EventLog(attributes=log.attributes, extensions=log.extensions, classifiers=log.classifiers, omni_present=log.omni_present, properties=log.properties)
    
    for index, trace in enumerate(variant_log):
        node = frozenset({frozenset({0})}) # Indicates position in state space of the model
        alignment_result = alignment.apply(trace, net, im, fm, parameters=parameters) # Format: (Trace / Log, Model)
        # No sorting of alignment, as order can be argued either way. Also we can have totally different optimal alignments.
        index_in_real_trace = 0
        for aligned_event in alignment_result["alignment"]:
            if aligned_event[1][1] == '>>': # Log only move
                enabled_activities = drg.nodes[node]["enabled_activities"]
                trace[index_in_real_trace][enabled_activities_name] = ', '.join(enabled_activities)
                index_in_real_trace += 1
                
            elif aligned_event[1][0] == '>>': # Model only move
                if aligned_event[1][1] is not None:
                # If not tau move advance model
                    node = get_node_from_transition(drg, node, aligned_event[0][1])
           
            else: # synchronous move
                enabled_activities = drg.nodes[node]["enabled_activities"]
                trace[index_in_real_trace][enabled_activities_name] = ', '.join(enabled_activities)
                node = get_node_from_transition(drg, node, aligned_event[0][1])
                index_in_real_trace += 1
                
        
        counter = 0
        while counter < variant_counter[index]:
            trace_to_append = copy.deepcopy(trace)
            # Introduce noise by selecting a random subset of enabled activities per event
            for i in range(len(trace_to_append)):
                executed_activity = trace_to_append[i]["concept:name"]
                activities = set(trace_to_append[i][enabled_activities_name].split(', '))
                activities.discard(executed_activity)
                # Introduce noise by randomly removing some enabled activities
                k = random.randint(0, len(activities))
                noisy_activities = random.sample(list(activities), k)
                # Ensure that the executed activity is always included (Definition of enabled activities!)
                noisy_activities.append(executed_activity)
                trace_to_append[i][enabled_activities_name] = ', '.join(noisy_activities)
            trace_to_append._set_attributes({"concept:name": str(global_case_id_counter)})
            annotated_log.append(trace_to_append)
            counter +=1
            global_case_id_counter +=1
    return annotated_log
    

def generate_log_without_noise(path_to_log, noise_threshold, alignment_parameters, enabled_activities_name="enabled_activities", pnml_path=None):
    log = pm4py.read_xes(path_to_log, return_legacy_log_object=True)
    # Clean the activity names for petrify to work correctly
    log = clean_activity_names(log)
    # First discover a ground truth model with the IMf
    net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=noise_threshold)
    
    if pnml_path is not None:
        pm4py.write_pnml(net, im, fm, pnml_path)
    
    # Debug: Write the discovered model to a pnml file
    #pm4py.write_pnml(net, im, fm, r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\utils\petrify_nets\Sepsis_0.2.pnml")
    
    variants = pm4py.statistics.variants.log.get.get_variants(log)
    variant_log = EventLog(attributes=log.attributes, extensions=log.extensions, classifiers=log.classifiers, omni_present=log.omni_present, properties=log.properties)
    variant_counter = {}
    counter = 0
    for variant in variants:
        variant_log.append(copy.deepcopy(variants[variant][0]))
        variant_counter[counter] = len(variants[variant])
        counter += 1
    parameters = {star.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
    print("Start building reachability graph")
    rg = ReachabilityGraph(net, im, fm, 1)
    print("Start building direct reachability graph and dfa")
    drg = DirectReachabilityGraph(rg).dfa
    global_case_id_counter = 1
    annotated_log = EventLog(attributes=log.attributes, extensions=log.extensions, classifiers=log.classifiers, omni_present=log.omni_present, properties=log.properties)
    print("Start aligning traces and annotating enabled activities")
    for index, trace in enumerate(variant_log):
        print(f"Aligning trace and annotating variant {index+1}/{len(variant_log)}")
        node = frozenset({frozenset({0})}) # Indicates position in state space of the model
        alignment_result = alignment.apply(trace, net, im, fm, parameters=parameters) # Format: (Trace / Log, Model)
        # No sorting of alignment, as order can be argued either way. Also we can have totally different optimal alignments.
        index_in_real_trace = 0
        for aligned_event in alignment_result["alignment"]:
            if aligned_event[1][1] == '>>': # Log only move
                # add the excecuted activity to the enabled activities, to stay in line with the definition of enabled activities
                enabled_activities = drg.nodes[node]["enabled_activities"]
                executed_activity = trace[index_in_real_trace]["concept:name"]
                if executed_activity not in enabled_activities:
                    enabled_activities.add(executed_activity)
                trace[index_in_real_trace][enabled_activities_name] = ', '.join(enabled_activities)
                index_in_real_trace += 1
                
            elif aligned_event[1][0] == '>>': # Model only move
                if aligned_event[1][1] is not None:
                # If not tau move advance model
                    node = get_node_from_transition(drg, node, aligned_event[0][1])
           
            else: # synchronous move
                enabled_activities = drg.nodes[node]["enabled_activities"]
                trace[index_in_real_trace][enabled_activities_name] = ', '.join(enabled_activities)
                node = get_node_from_transition(drg, node, aligned_event[0][1])
                index_in_real_trace += 1
                
        
        counter = 0
        while counter < variant_counter[index]:
            trace_to_append = copy.deepcopy(trace)
            """
            # Introduce noise by selecting a random subset of enabled activities per event
            for i in range(len(trace_to_append)):
                executed_activity = trace_to_append[i]["concept:name"]
                activities = set(trace_to_append[i][enabled_activities_name].split(', '))
                activities.discard(executed_activity)
                # Introduce noise by randomly removing some enabled activities
                k = random.randint(0, len(activities))
                noisy_activities = random.sample(list(activities), k)
                # Ensure that the executed activity is always included (Definition of enabled activities!)
                noisy_activities.append(executed_activity)
                trace_to_append[i][enabled_activities_name] = ', '.join(noisy_activities)
            """
            trace_to_append._set_attributes({"concept:name": str(global_case_id_counter)})
            annotated_log.append(trace_to_append)
            counter +=1
            global_case_id_counter +=1
    return annotated_log

def clean_activity_names(log: EventLog) -> EventLog:
    """
    Replaces underscores and hyphens with spaces in the activity names of the event log. Needed for petrify to work correctly.

    :param log: The input event log.
    :type log: EventLog
    :return: The modified event log.
    :rtype: EventLog
    """
    for trace in log:
        for event in trace:
            name = event.get("concept:name")
            if name:
                event["concept:name"] = name.replace("_", " ").replace("-", " ")
    return log

if __name__ == "__main__":
    log_path = r"C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\TranslucentActivityRelationships-main\\evaluation\\sepsis\\Sepsis Cases - Event Log.xes.gz"
    noise_threshold = 0.8  # Example noise threshold
    alignment_params = {}
    #generate_log_with_noise(log_path, noise_threshold, alignment_params)
    #for noise_threshold in [0.2, 0.4, 0.6, 0.8, 1.0]: 
    generate_log_without_noise(log_path, noise_threshold, alignment_params)