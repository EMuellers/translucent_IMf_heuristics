from pm4py.objects.log.obj import EventLog
from pm4py.objects.conversion.log import converter as log_converter
import pandas as pd
import pm4py

from translucent_discovery.translucent_inductive_miner.utils import get_translucent_trace_variants
from translucent_discovery.translucent_inductive_miner.translucent_datatype import TCL


def get_start_activities(log, executed_activities, enabled_activities_key="enabled_activities"):
    start_activities = set()
    variants = get_translucent_trace_variants(log)
    for variant in variants:
        trace = variants[variant][0]
        if len(trace) > 0:
            start_activities_strings = trace[0][enabled_activities_key].split(",")
            for el in start_activities_strings:
                if el.strip() in executed_activities:
                    start_activities.add(el.strip())
    return start_activities

# start activities for tcl logs
def get_start_activities_tcl(log: TCL, executed_activities):
    start_activities = set()
    for trace in log:
        if len(trace) > 0:
            start_activities_current = trace[0][1]
            for el in start_activities_current:
                if el in executed_activities:
                    start_activities.add(el)
    return start_activities


def get_end_activities(log, executed_activities, enabled_activities_key="enabled_activities", strict_end_activities=False):
    end_activities = set()
    variants = get_translucent_trace_variants(log)
    #Change:
    #Elias: Strict end activities only count those that actually appeared at the end of a trace at least once
    if strict_end_activities:
        at_least_once_end_activities = { variants[v][0][-1]["concept:name"] for v in variants if len(variants[v][0]) > 0 }
    for variant in variants:
        trace = variants[variant][0]
        if len(trace) > 0:
            end_activities_strings = trace[-1][enabled_activities_key].split(",")
            for el in end_activities_strings:
                el_s = el.strip()
                # include activity only if it's an executed activity and,
                # when strict_end_activities is True, also only if it actually appeared
                # as the final executed activity in at least one trace
                if el_s in executed_activities and (not strict_end_activities or el_s in at_least_once_end_activities):
                    end_activities.add(el_s)
    return end_activities

# end activities for tcl logs
def get_end_activities_tcl(log: TCL, executed_activities, strict_end_activities=False):
    end_activities = set()
    #Change:
    #Elias: Strict end activities only count those that actually appeared at the end of a trace at least once
    if strict_end_activities:
        at_least_once_end_activities = { trace[-1][0] for trace in log if trace }
    for trace in log:
        if len(trace) > 0:
            end_activities_current = trace[-1][1]
            for el in end_activities_current:
                # include activity only if it's an executed activity and,
                # when strict_end_activities is True, also only if it actually appeared
                # as the final executed activity in at least one trace
                if el in executed_activities and (not strict_end_activities or el in at_least_once_end_activities):
                    end_activities.add(el)
    return end_activities


def get_directly_follow_relationships(log, executed_activities, enabled_activities_key="enabled_activities") -> dict:
    activity_follow = {}
    variants = get_translucent_trace_variants(log)
    for variant in variants:
        trace = variants[variant][0]
        for index, current_event in enumerate(trace):
            if index < len(trace)-1:
                executed_activity = current_event["concept:name"]
                enabled_activities_next = [el.strip() for el in trace[index+1][enabled_activities_key].split(",") if el.strip() in executed_activities]
                if executed_activity not in activity_follow:
                    activity_follow[executed_activity] = set()
                for next_activity in enabled_activities_next:
                    activity_follow[executed_activity].add(next_activity)
    return activity_follow

# translucent directly follows for tcl
def get_directly_follow_relationships_tcl(log: TCL, executed_activities) -> dict:
    activity_follow = {}
    for trace in log:
        for index, current_event in enumerate(trace):
            if index < len(trace)-1:
                executed_activity = current_event[0]
                enabled_activities_next = trace[index+1][1].intersection(executed_activities)
                if executed_activity not in activity_follow:
                    activity_follow[executed_activity] = set()
                for next_activity in enabled_activities_next:
                    activity_follow[executed_activity].add(next_activity)
    return activity_follow


def get_choice_relationships(log, executed_activities, enabled_activities_key="enabled_activities") -> dict:
    activity_choice = {}
    variants = get_translucent_trace_variants(log)
    for variant in variants:
        trace = variants[variant][0]
        for index, current_event in enumerate(trace):
            if index < len(trace)-1:
                executed_activity = current_event["concept:name"]
                enabled_activities_current = set(
                    [el.strip() for el in current_event[enabled_activities_key].split(",") if
                     el.strip() in executed_activities])
                enabled_activities_next = set(
                    [el.strip() for el in trace[index + 1][enabled_activities_key].split(",") if
                     el.strip() in executed_activities])
                removed_activities = enabled_activities_current.difference(enabled_activities_next)
                for activity in removed_activities:
                    if executed_activity not in activity_choice:
                        activity_choice[executed_activity] = set()
                    activity_choice[executed_activity].add(activity)
                    if activity not in activity_choice:
                        activity_choice[activity] = set()
                    activity_choice[activity].add(executed_activity)
    return activity_choice


def get_parallel_relationships(log, executed_activities, enabled_activities_key="enabled_activities") -> dict:
    activity_parallel = {}
    variants = get_translucent_trace_variants(log)
    for variant in variants:
        trace = variants[variant][0]
        for index, current_event in enumerate(trace):
            if index < len(trace)-1:
                executed_activity = current_event["concept:name"]
                enabled_activities_current = set([el.strip() for el in current_event[enabled_activities_key].split(",") if el.strip() in executed_activities])
                enabled_activities_next = set([el.strip() for el in trace[index+1][enabled_activities_key].split(",") if el.strip() in executed_activities])
                still_enabled = enabled_activities_current.intersection(enabled_activities_next)
                for activity in still_enabled:
                    if executed_activity not in activity_parallel:
                        activity_parallel[executed_activity] = set()
                    activity_parallel[executed_activity].add(activity)
                    if activity not in activity_parallel:
                        activity_parallel[activity] = set()
                    activity_parallel[activity].add(executed_activity)
    return activity_parallel

# get_parallel_relationships for tcl logs
def get_parallel_relationships_tcl(log: TCL, executed_activities) -> dict:
    activity_parallel = {}
    for trace in log:
        for index, current_event in enumerate(trace):
            if index < len(trace)-1:
                executed_activity = current_event[0]
                enabled_activities_current = current_event[1].intersection(executed_activities)
                enabled_activities_next = trace[index+1][1].intersection(executed_activities)
                still_enabled = enabled_activities_current.intersection(enabled_activities_next)
                for activity in still_enabled:
                    if activity != executed_activity:
                        if executed_activity not in activity_parallel:
                            activity_parallel[executed_activity] = set()
                        activity_parallel[executed_activity].add(activity)
                        if activity not in activity_parallel:
                            activity_parallel[activity] = set()
                        activity_parallel[activity].add(executed_activity)
    return activity_parallel


def get_parallel_relationships_frequent(log, executed_activities, enabled_activities_key="enabled_activities") -> dict:
    activity_parallel = {}
    variants = get_translucent_trace_variants(log)
    for variant in variants:
        trace = variants[variant][0]
        number_of_occurrence = len(variants[variant][1])
        for index, current_event in enumerate(trace):
            if index < len(trace)-1:
                executed_activity = current_event["concept:name"]
                enabled_activities_current = set([el.strip() for el in current_event[enabled_activities_key].split(",") if el.strip() in executed_activities])
                enabled_activities_next = set([el.strip() for el in trace[index+1][enabled_activities_key].split(",") if el.strip() in executed_activities])
                still_enabled = enabled_activities_current.intersection(enabled_activities_next)
                for activity in still_enabled:
                    if (executed_activity, activity) not in activity_parallel:
                        activity_parallel[(executed_activity, activity)] = 0
                    activity_parallel[(executed_activity, activity)] += number_of_occurrence
                    if (activity, executed_activity) not in activity_parallel:
                        activity_parallel[(activity, executed_activity)] = 0
                    activity_parallel[(activity, executed_activity)] += number_of_occurrence
    return activity_parallel

# get_parallel_relationships_frequent for tcl logs
def get_parallel_relationships_frequent_tcl(log: TCL, executed_activities) -> dict:
    activity_parallel = {}
    for trace in log:
        number_of_occurrence = log[trace]
        for index, current_event in enumerate(trace):
            if index < len(trace)-1:
                executed_activity = current_event[0]
                enabled_activities_current = current_event[1].intersection(executed_activities)
                enabled_activities_next = trace[index+1][1].intersection(executed_activities)
                still_enabled = enabled_activities_current.intersection(enabled_activities_next)
                for activity in still_enabled:
                    if activity != executed_activity:
                        if (executed_activity, activity) not in activity_parallel:
                            activity_parallel[(executed_activity, activity)] = 0
                        activity_parallel[(executed_activity, activity)] += number_of_occurrence
                        if (activity, executed_activity) not in activity_parallel:
                            activity_parallel[(activity, executed_activity)] = 0
                        activity_parallel[(activity, executed_activity)] += number_of_occurrence
    return activity_parallel


def get_start_activities_frequent(log, executed_activities, enabled_activities_key="enabled_activities"):
    start_activities = {}
    variants = get_translucent_trace_variants(log)
    for variant in variants:
        number_of_occurrence = len(variants[variant][1])
        trace = variants[variant][0]
        if len(trace) > 0:
            start_activities_strings = trace[0][enabled_activities_key].split(",")
            for el in start_activities_strings:
                if el.strip() in executed_activities:
                    if el.strip() not in start_activities:
                        start_activities[el.strip()] = 0
                    start_activities[el.strip()] += number_of_occurrence
    return start_activities

# get_start_activities_frequent for tcl logs
def get_start_activities_frequent_tcl(log: TCL, executed_activities):
    start_activities = {}
    for trace in log:
        number_of_occurrence = log[trace]
        if len(trace) > 0:
            start_activities_current = trace[0][1]
            for el in start_activities_current:
                if el in executed_activities:
                    if el not in start_activities:
                        start_activities[el] = 0
                    start_activities[el] += number_of_occurrence
    return start_activities


def get_end_activities_frequent(log, executed_activities, enabled_activities_key="enabled_activities", strict_end_activities=False):
    end_activities = {}
    variants = get_translucent_trace_variants(log)
    #Change:
    #Elias: Strict end activities only count those that actually appeared at the end of a trace at least once
    if strict_end_activities:
        at_least_once_end_activities = { variants[v][0][-1]["concept:name"] for v in variants if len(variants[v][0]) > 0 }
    for variant in variants:
        number_of_occurrence = len(variants[variant][1])
        trace = variants[variant][0]
        if len(trace) > 0:
            end_activities_strings = trace[-1][enabled_activities_key].split(",")
            for el in end_activities_strings:
                el_s = el.strip()
                # include activity only if it's an executed activity and,
                # when strict_end_activities is True, also only if it actually appeared
                # as the final executed activity in at least one trace
                if el_s in executed_activities and (not strict_end_activities or el_s in at_least_once_end_activities):
                    if el_s not in end_activities:
                        end_activities[el_s] = 0
                    end_activities[el_s] += number_of_occurrence
    return end_activities

# get_end_activities_frequent for tcl logs
def get_end_activities_frequent_tcl(log: TCL, executed_activities, strict_end_activities=False):
    end_activities = {}
    #Change:
    #Elias: Strict end activities only count those that actually appeared at the end of a trace at least once
    if strict_end_activities:
        at_least_once_end_activities = { variant[-1][0] for variant in log }
    for trace in log:
        number_of_occurrence = log[trace]
        if len(trace) > 0:
            end_activities_current = trace[-1][1]
            for el in end_activities_current:
                # include activity only if it's an executed activity and,
                # when strict_end_activities is True, also only if it actually appeared
                # as the final executed activity in at least one trace
                if el in executed_activities and (not strict_end_activities or el in at_least_once_end_activities):
                    if el not in end_activities:
                        end_activities[el] = 0
                    end_activities[el] += number_of_occurrence
    return end_activities


def get_choice_relationships_frequent(log, executed_activities, enabled_activities_key="enabled_activities") -> dict:
    activity_choice = {}
    variants = get_translucent_trace_variants(log)
    for variant in variants:
        number_of_occurrence = len(variants[variant][1])
        trace = variants[variant][0]
        for index, current_event in enumerate(trace):
            if index < len(trace)-1:
                executed_activity = current_event["concept:name"]
                enabled_activities_current = set(
                    [el.strip() for el in current_event[enabled_activities_key].split(",") if
                     el.strip() in executed_activities])
                enabled_activities_next = set(
                    [el.strip() for el in trace[index + 1][enabled_activities_key].split(",") if
                     el.strip() in executed_activities])
                removed_activities = enabled_activities_current.difference(enabled_activities_next)
                for activity in removed_activities:
                    if (executed_activity, activity) not in activity_choice:
                        activity_choice[(executed_activity, activity)] = 0
                    activity_choice[(executed_activity, activity)] += number_of_occurrence
                    if (activity, executed_activity) not in activity_choice:
                        activity_choice[(activity, executed_activity)] = 0
                    activity_choice[(activity, executed_activity)] += number_of_occurrence
    return activity_choice

# get_choice_relationships_frequent for tcl logs
def get_choice_relationships_frequent_tcl(log: TCL, executed_activities) -> dict:
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
    return activity_choice


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
