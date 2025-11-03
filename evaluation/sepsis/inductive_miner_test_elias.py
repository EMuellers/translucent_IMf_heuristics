import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization

initial_log = pm4py.read_xes(r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\evaluation\sepsis\Sepsis Cases - Event Log.xes.gz")  # or construct event log
initial_log = log_converter.apply(initial_log, variant=log_converter.Variants.TO_EVENT_LOG)


# Direct pm4py convenience call (if available in your pm4py build)
net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(initial_log, noise_threshold=0.4)

pm4py.view_petri_net(net, initial_marking, final_marking)