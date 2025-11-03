import pm4py


def add_start_and_end_translucent(log):
    log = pm4py.insert_artificial_start_end(log, artificial_start="artificial_start_activity", artificial_end="artificial_end_activity")
    for trace in log:
        trace[0]["enabled_activities"] = 'artificial_start_activity'
        trace[-1]["enabled_activities"] = 'artificial_end_activity'
    return log

def add_start_and_end_evaluation(log):
    log = pm4py.insert_artificial_start_end(log, artificial_start="artificial_start_activity_2",
                                            artificial_end="artificial_end_activity_2")
    for trace in log:
        trace[0]["enabled_activities"] = 'artificial_start_activity_2'
        trace[-1]["enabled_activities"] = 'artificial_end_activity_2'
    return log