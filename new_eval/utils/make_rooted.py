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