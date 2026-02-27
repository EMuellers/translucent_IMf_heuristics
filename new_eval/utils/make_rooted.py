from pm4py.objects.log.obj import EventLog, Trace, Event
import datetime

def add_artificial_start_activity_translucent(log, start_activity_name="__start__"):
    """
    Adds an artificial start activity to each trace in the event log.

    :param log: The input event log.
    :type log: EventLog
    :param start_activity_name: The name of the artificial start activity.
    :type start_activity_name: str
    :return: The modified event log with the artificial start activity added.
    :rtype: EventLog
    """
    for trace in log:
        start_event = Event({"concept:name": start_activity_name, "enabled_activities": start_activity_name})
        if trace:
            if "time:timestamp" in trace[0]:
                start_event["time:timestamp"] = trace[0][
                    "time:timestamp"
                ] - datetime.timedelta(seconds=1)
        trace.insert(0, start_event)
    return log

def add_artificial_start_and_end_activities_translucent(log, start_activity_name="__start__", end_activity_name="__end__"):
    """
    Adds artificial start and end activities to each trace in the event log.

    :param log: The input event log.
    :type log: EventLog
    :param start_activity_name: The name of the artificial start activity.
    :type start_activity_name: str
    :param end_activity_name: The name of the artificial end activity.
    :type end_activity_name: str
    :return: The modified event log with the artificial start and end activities added.
    :rtype: EventLog
    """
    for trace in log:
        start_event = Event({"concept:name": start_activity_name, "enabled_activities": start_activity_name})
        end_event = Event({"concept:name": end_activity_name, "enabled_activities": end_activity_name})
        if trace:
            if "time:timestamp" in trace[0]:
                start_event["time:timestamp"] = trace[0][
                    "time:timestamp"
                ] - datetime.timedelta(seconds=1)
            if "time:timestamp" in trace[-1]:
                end_event["time:timestamp"] = trace[-1][
                    "time:timestamp"
                ] + datetime.timedelta(seconds=1)
        trace.insert(0, start_event)
        trace.append(end_event)
    return log

def get_alignment_fitness_with_processtree(log, net, im, fm):
    import pm4py
    import numpy as np
    from pm4py.algo.conformance.alignments.petri_net.variants import state_equation_a_star
    #alignments = []
    # Create a process tree from the Petri net 
    process_tree = pm4py.objects.conversion.wf_net.variants.to_process_tree.apply(net, im, fm)
    alignments = pm4py.algo.conformance.alignments.process_tree.variants.milp.apply(log, process_tree)
    model_bwc = get_best_worst_model_cost(net, im, fm)
    trace_bwc_values = [model_bwc + len(trace) for trace in log]
    bestworstcost = sum(trace_bwc_values)
    log_fitness = sum(np.round(t["cost"]) for t in alignments) / bestworstcost if bestworstcost > 0 else 0
    log_fitness = 1.0 - log_fitness
    return log_fitness

def get_best_worst_model_cost(net, im, fm):
    import pm4py
    alignment = pm4py.algo.conformance.alignments.petri_net.algorithm.apply(pm4py.objects.log.obj.Trace(), net, im, fm)['alignment']
    cost = 0
    for step in alignment:
        if step [1] is not None:
            cost += 1
    return cost


if __name__ == "__main__":
    import pm4py
    import pandas as pd
    from translucent_discovery.translucent_inductive_miner.translucent_base import discover_petri_net
    
    log = pd.read_csv(r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\translucent_datasets\road_traffic_fine\road_traffic_fine_0.2.csv")
    log = pm4py.format_dataframe(log, case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp", timest_format="%Y-%m-%d %H:%M:%S%z")
    log = pm4py.convert_to_event_log(log)
    algo_parameters = {
        
        ### PARAMETERS FOR NON-FREQUENT ALGORITHM ###
        
        
        ### PARAMETERS FOR FREQUENT ALGORITHM ###
        
        "delta_heuristic_frequent_before": False, # Signifies if to apply the remove arcs heuristic in the frequent case on the unfiltered tDFG
        
        "delta_heuristic_frequent_after": False, # Signifies if to apply the remove arcs heuristic in the frequent case on the filtered tDFG
        
        ### PARAMETERS THAT APPLY TO BOTH ###
        
        "strict_end_activities": True, # Only consider translucent end activities which actually appear at the end of a trace at least once
        
        "remove_arcs_heuristics": False, #"dependency_score", # Remove arcs exclusive to tDFG before applying fall throughs ("dependency_score"), set to False to disable
                                        # "exclusive_choice_frequency" to use the choice frequency for filtering
                                        # 'confidence' to use confidence of translucent df relation for filtering
                                        # 'support' to use support of translucent df relation for filtering
        
        "add_arcs_heuristics": False, # "dependency_score" # Add arcs from the tDFG to the DFG before applying fall throughs
                                    # "parallel_relationship_frequency" to use the choice frequency for filtering
                                        # 'confidence' to use confidence of translucent df relation for filtering
                                        # 'support' to use support of translucent df relation for filtering
        
        "parallel_end_activities_heuristic": True, # If two activities are in translucent parallel relation and one is an end activity, the other is also considered an end activity in the (frequent) tDFG
        
        "translucent_self_loops": True # Keep translucent self loops when projecting onto single activities, by projecting these on traces where activities follow themselves directly.

    }
    net, im, fm = discover_petri_net(log,{"translucent_variant": "IMtf", "tDFG_fall_through": True} | algo_parameters, noise_threshold=0.8)
    alignments = get_alignment_fitness_with_processtree(log, net, im, fm)
    print(alignments)