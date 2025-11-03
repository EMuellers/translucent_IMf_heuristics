from pm4py.objects.log.obj import EventLog, Trace

def get_translucent_trace_variants(event_log: EventLog, enabled_activities_key="enabled_activities"
                                   ) -> dict[tuple[tuple[str, frozenset[str]], ...], tuple[Trace, list[int]]]:
    variants: dict[tuple[tuple[str, str], ...], tuple[Trace, list[int]]] = {}
    for idx, trace in enumerate(event_log):
        variant = tuple((event["concept:name"], event[enabled_activities_key]) for event in trace)
        if variant not in variants:
            variants[variant] = (trace, [idx])
        else:
            variants[variant][1].append(idx)
    return variants