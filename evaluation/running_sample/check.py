from synthetical_log_generation.log_generator import create_translucent_event_log
from synthetical_log_generation.log_generator import create_logs
from translucent_discovery.translucent_inductive_miner.translucent_base import discover_petri_net
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from translucent_precision.main import translucent_precision_score
from evaluation.translucent_f_1_score import compute_f_1_scores
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataframe = pd.read_csv(r"C:\Users\beyel\Documents\ProcessScience\2024\RoadTraffic\logs\04\ground_truth.csv", sep=",")
dataframe = pm4py.format_dataframe(dataframe, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')
#dataframe = pm4py.format_dataframe(dataframe, case_id='CaseID', activity_key='Activity', timestamp_key='Timestamp')
log = pm4py.convert_to_event_log(dataframe)

#model, i_m, f_m = discover_petri_net(log, {"translucent_variant": "IMto", "tDFG_fall_through": True})
#pm4py.view_petri_net(model, i_m, f_m)

#pm4py.save_vis_petri_net(model, i_m, f_m, "petri_net_without_frequency.png")



model, i_m, f_m = discover_petri_net(log,{"translucent_variant": "IMts", "tDFG_fall_through": False}, 0.2)
pm4py.view_petri_net(model, i_m, f_m)
#print(model)
pm4py.save_vis_petri_net(model, i_m, f_m, "petri_net_2.png")
