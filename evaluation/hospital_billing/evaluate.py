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
import os
import glob



def evaluate(base, tDFG, threshold):
    if base == 0.8:
        base = "08"
    elif base == 0.6:
        base = "06"
    elif base == 0.4:
        base = "04"
    elif base == 0.3:
        base = "03"
    elif base == 0.2:
        base = "02"
    else:
        base = "None"

    if threshold == 0.8:
        threshold_str = "08"
    elif threshold == 0.6:
        threshold_str = "06"
    elif threshold == 0.4:
        threshold_str = "04"
    elif threshold == 0.3:
        threshold_str = "03"
    elif threshold == 0.2:
        threshold_str = "02"
    elif threshold == 0:
        threshold_str = "00"
    else:
        threshold_str = "None"
    # Bedeutung: base --> Mit welchem IMf wurde der log erstellt
    # threshold --> Mit welchem noise filtering werden die Modelle erstellt
    log = pd.read_csv('logs/'+base+'/ground_truth.csv')
    log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)

    im_fitness = []
    im_precision = []

    imto_fitness = []
    imto_precision = []

    imtf_fitness = []
    imtf_precision = []

    imts_fitness = []
    imts_precision = []
    i = 0
    subfolder_path = 'logs/'+base

    # Get a list of all CSV files in the subfolder except 'ground_truth.csv'
    csv_files = glob.glob(os.path.join(subfolder_path, '*.csv'))
    csv_files = [file for file in csv_files if 'ground_truth.csv' not in file]
    i = 0
    for file in csv_files:
        print(i)
        i += 1
        df = pd.read_csv(file)
        sub_log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)
        if not tDFG:
            model, i_m, f_m = discover_petri_net(sub_log, {"translucent_variant": "IM"}, threshold)
        else:
            model, i_m, f_m = discover_petri_net(sub_log, {"translucent_variant": "IM", "tDFG_fall_through": True}, threshold)
        im_fitness.append(pm4py.conformance.fitness_alignments(log, model, i_m, f_m)["log_fitness"])
        im_precision.append(translucent_precision_score(log, model, i_m, f_m))

        if not tDFG:
            model, i_m, f_m = discover_petri_net(sub_log, {"translucent_variant": "IMto"}, threshold)
        else:
            model, i_m, f_m = discover_petri_net(sub_log, {"translucent_variant": "IMto", "tDFG_fall_through": True},
                                                 threshold)
        imto_fitness.append(pm4py.conformance.fitness_alignments(log, model, i_m, f_m)["log_fitness"])
        imto_precision.append(translucent_precision_score(log, model, i_m, f_m))

        if not tDFG:
            model, i_m, f_m = discover_petri_net(sub_log, {"translucent_variant": "IMtf"}, threshold)
        else:
            model, i_m, f_m = discover_petri_net(sub_log, {"translucent_variant": "IMtf", "tDFG_fall_through": True},
                                                 threshold)
        imtf_fitness.append(pm4py.conformance.fitness_alignments(log, model, i_m, f_m)["log_fitness"])
        imtf_precision.append(translucent_precision_score(log, model, i_m, f_m))

        if not tDFG:
            model, i_m, f_m = discover_petri_net(sub_log, {"translucent_variant": "IMts"}, threshold)
        else:
            model, i_m, f_m = discover_petri_net(sub_log, {"translucent_variant": "IMts", "tDFG_fall_through": True},
                                                 threshold)
        imts_fitness.append(pm4py.conformance.fitness_alignments(log, model, i_m, f_m)["log_fitness"])
        imts_precision.append(translucent_precision_score(log, model, i_m, f_m))

    im_f1 = compute_f_1_scores(im_fitness, im_precision)
    imto_f1 = compute_f_1_scores(imto_fitness, imto_precision)
    imtf_f1 = compute_f_1_scores(imtf_fitness, imtf_precision)
    imts_f1 = compute_f_1_scores(imts_fitness, imts_precision)

    result_im = pd.DataFrame({"Fitness": im_fitness, "Precision": im_precision, "F1_Score": im_f1})
    if not tDFG:
        result_im.to_csv('results/DFG_fall/'+base+"/"+threshold_str+"/im_result.csv", sep=",", index=False)
    else:
        result_im.to_csv('results/tDFG_fall/'+base+"/"+threshold_str+"/im_result.csv", sep=",", index=False)

    result_imto = pd.DataFrame({"Fitness": imto_fitness, "Precision": imto_precision, "F1_Score": imto_f1})
    if not tDFG:
        result_imto.to_csv('results/DFG_fall/'+base+"/"+threshold_str+"/imto_result.csv", sep=",", index=False)
    else:
        result_imto.to_csv('results/tDFG_fall/'+base+"/"+threshold_str+"/imto_result.csv", sep=",", index=False)

    result_imtf = pd.DataFrame({"Fitness": imtf_fitness, "Precision": imtf_precision, "F1_Score": imtf_f1})
    if not tDFG:
        result_imtf.to_csv('results/DFG_fall/'+base+"/"+threshold_str+"/imtf_result.csv", sep=",", index=False)
    else:
        result_imtf.to_csv('results/tDFG_fall/'+base+"/"+threshold_str+"/imtf_result.csv", sep=",", index=False)


    result_imts = pd.DataFrame({"Fitness": imts_fitness, "Precision": imts_precision, "F1_Score": imts_f1})
    if not tDFG:
        result_imts.to_csv('results/DFG_fall/'+base+"/"+threshold_str+"/imts_result.csv", sep=",", index=False)
    else:
        result_imts.to_csv('results/tDFG_fall/'+base+"/"+threshold_str+"/imts_result.csv", sep=",", index=False)

    algorithms = ['IM', 'IMto', 'IMtf', 'IMts']
    criteria = ['Fitness', 'Precision', 'F1-Score']
    algorithm_colors = ['blue', 'green', 'red', 'purple']
    data = {
        'IM': {
            'Fitness': im_fitness,
            'Precision': im_precision,
            'F1-Score': im_f1
        },
        'IMto': {
            'Fitness': imto_fitness,
            'Precision': imto_precision,
            'F1-Score': imto_f1
        },
        'IMtf': {
            'Fitness': imtf_fitness,
            'Precision': imtf_precision,
            'F1-Score': imtf_f1
        },
        'IMts': {
            'Fitness': imts_fitness,
            'Precision': imts_precision,
            'F1-Score': imts_f1
        }
    }

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot data for each criterion
    for i, criterion in enumerate(criteria):
        ax = axs[i]
        ax.set_title(criterion, fontsize=14)
        ax.set_xlabel('Number of Variants', fontsize=12)
        ax.set_xticks(np.insert(np.arange(4, len(data["IM"]["Fitness"]), 5), 0, 0))
        ax.set_xticklabels([str(i) for i in np.insert(np.arange(5, len(data["IM"]["Fitness"]) + 1, 5), 0, 1)])
        ax.tick_params(axis='x', which='major', labelsize=10, size=5)
        for algo, color in zip(algorithms, algorithm_colors):
            ax.plot(data[algo][criterion], label=algo, marker='o', color=color)
    # Add legend below subplots
    fig.legend(algorithms, loc='lower center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True,
               ncol=len(algorithms))

    # Adjust layout
    plt.tight_layout()

    # Save plot as a tight PDF
    if not tDFG:
        plt.savefig('results/DFG_fall/'+base+"/"+threshold_str+"/result_5.pdf", bbox_inches="tight")
    else:
        plt.savefig('results/tDFG_fall/'+base+"/"+threshold_str+"/result_5.pdf", bbox_inches="tight")

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot data for each criterion
    for i, criterion in enumerate(criteria):
        ax = axs[i]
        ax.set_title(criterion, fontsize=14)
        ax.set_xlabel('Number of Variants', fontsize=12)
        ax.set_xticks(np.insert(np.arange(4, len(data["IM"]["Fitness"]), 10), 0, 0))
        ax.set_xticklabels([str(i) for i in np.insert(np.arange(5, len(data["IM"]["Fitness"]) + 1, 10), 0, 1)])
        ax.tick_params(axis='x', which='major', labelsize=10, size=5)
        for algo, color in zip(algorithms, algorithm_colors):
            ax.plot(data[algo][criterion], label=algo, marker='o', color=color)
    # Add legend below subplots
    fig.legend(algorithms, loc='lower center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True,
               ncol=len(algorithms))

    # Adjust layout
    plt.tight_layout()

    # Save plot as a tight PDF
    if not tDFG:
        plt.savefig('results/DFG_fall/'+base+"/"+threshold_str+"/result_10.pdf", bbox_inches="tight")
    else:
        plt.savefig('results/tDFG_fall/'+base+"/"+threshold_str+"/result_10.pdf", bbox_inches="tight")


logs = [0.8, 0.6, 0.4, 0.3, 0.2]
model = [0.8, 0.6, 0.4, 0.3, 0.2, 0]

tDFGs = [True, False]

for log in logs:
    for s2 in model:
        for tdfg in tDFGs:
            evaluate(log, tdfg, s2)
