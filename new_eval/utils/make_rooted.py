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