from typing import Counter, Tuple, FrozenSet
from pm4py.objects.log.obj import EventLog
from pm4py.util.compression.dtypes import UVCL

# 1. A single Event is a pair: (Executed Activity, Set of Enabled Activities)
# We use FrozenSet because the set of enabled activities must be hashable.
TranslucentEvent = Tuple[str, FrozenSet[str]]

# 2. A Variant is a sequence of Translucent Events
TranslucentVariant = Tuple[TranslucentEvent, ...]

# 3. The Compressed Log maps the Translucent Variant to its frequency
TranslucentCompressedLog = Counter[TranslucentVariant]

TCL = TranslucentCompressedLog

def translucent_log_to_tcl(log: EventLog, enabled_activities_key="enabled_activities") -> TCL:
    variants = Counter()
    for trace in log:
        variant = tuple((event["concept:name"], frozenset(el.strip() for el in event[enabled_activities_key].split(","))) for event in trace)
        variants[variant] += 1
    return variants

def tcl_to_uvcl(tcl: TCL) -> UVCL:
    uvcl = Counter()
    for variant, count in tcl.items():
        ref_trace = tuple(event[0] for event in variant)
        uvcl[ref_trace] += count
    return uvcl

def get_executed_activities(tcl: TCL) -> set[str]:
    executed_activities = set()
    for variant in tcl.keys():
        for event in variant:
            executed_activities.add(event[0])
    return set(sorted(executed_activities))

def get_executed_events(tcl: TCL) -> set[TranslucentEvent]:
    executed_events = set()
    for variant in tcl.keys():
        for event in variant:
            executed_events.add(event)
    return set(sorted(executed_events))

def get_start_activities_tcl(tcl: TCL) -> Counter[str]:
    starts = Counter()
    for variant, count in tcl.items():
        if len(variant) > 0:
            starts[variant[0][0]] += count
    return starts

def get_end_activities_tcl(tcl: TCL) -> Counter[str]:
    ends = Counter()
    for variant, count in tcl.items():
        if len(variant) > 0:
            ends[variant[-1][0]] += count
    return ends

def get_executed_activity_frequencies_tcl(tcl: TCL) -> Counter[str]:
    activity_frequencies = Counter()
    for variant, count in tcl.items():
        for event in variant:
            activity_frequencies[event[0]] += count
    return activity_frequencies

def get_translucent_self_loops_old(tcl: TCL) -> set[str]:
    excecuted_activities = get_executed_activities(tcl)
    loop_dict = {activity: 0 for activity in excecuted_activities}
    for variant, count in tcl.items():
        for i in range(len(variant) - 1):
            current_event = variant[i]
            next_event = variant[i + 1]
            #if current_event[0] != next_event[0] and current_event[0] in next_event[1]: # count only translucent self-loops (enabled in next event but not executed)
            if current_event[0] in next_event[1]:
                loop_dict[current_event[0]] += count
    return loop_dict

def get_translucent_self_loops(tcl: TCL, threshold: float) -> set[str]:
    excecuted_activities_frequencies = get_executed_activity_frequencies_tcl(tcl)
    loop_dict = {activity: 0 for activity in excecuted_activities_frequencies.keys()}
    for variant, count in tcl.items():
        for i in range(len(variant) - 1):
            current_event = variant[i]
            next_event = variant[i + 1]
            #if current_event[0] != next_event[0] and current_event[0] in next_event[1]: # count only translucent self-loops (enabled in next event but not executed)
            if current_event[0] in next_event[1]:
                loop_dict[current_event[0]] += count
    end_frequencies = get_end_activities_tcl(tcl)
    excecuted_activities_frequencies -= end_frequencies
    for activity in loop_dict.keys():
        if loop_dict[activity] > excecuted_activities_frequencies[activity] * threshold:
            loop_dict[activity] = 1
        else:
            loop_dict[activity] = 0
    return loop_dict

if __name__ == "__main__":
    # Testing some functionality
    import pandas as pd
    from pm4py.objects.conversion.log import converter as log_converter
    csv_file = r"C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\TranslucentActivityRelationships-main\\evaluation\\sepsis\\04\\4.csv"
    df = pd.read_csv(csv_file)
    log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)
    tcl_log = translucent_log_to_tcl(log)
    uvcl_log = tcl_to_uvcl(tcl_log)
    print("done")