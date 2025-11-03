import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from translucent_precision.main import translucent_precision_score
from evaluation.modify_logs import add_start_and_end_translucent
from translucent_discovery.translucent_inductive_miner.translucent_base import discover_petri_net
from evaluation.translucent_f_1_score import compute_f_1_scores
import matplotlib.pyplot as plt
import numpy as np

numbers = range(1, 25, 1)
ground_truth = pd.read_csv("logs/case_24.csv", sep=",")
ground_truth = pm4py.format_dataframe(ground_truth, case_id='case', activity_key='activity', timestamp_key='timestamp')
ground_truth = log_converter.apply(ground_truth, variant=log_converter.Variants.TO_EVENT_LOG)
# ground_truth = add_start_and_end_translucent(ground_truth)
im_fitness = []
im_precision = []

imto_fitness = []
imto_precision = []

imtf_fitness = []
imtf_precision = []

imts_fitness = []
imts_precision =[]

for number in numbers:
    print(number)
    log = pd.read_csv("logs/case_" + str(number) + ".csv", sep=",")
    log = pm4py.format_dataframe(log, case_id='case', activity_key='activity', timestamp_key='timestamp')
    log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
    # log = add_start_and_end_translucent(log)
    model, i_m, f_m = discover_petri_net(log, {"translucent_variant": "IM"})
    pm4py.write.write_pnml(model, i_m, f_m, "models/IM/"+str(number))
    im_fitness.append(pm4py.conformance.fitness_alignments(ground_truth, model, i_m, f_m)["log_fitness"])
    im_precision.append(translucent_precision_score(ground_truth, model, i_m, f_m))

    model, i_m, f_m = discover_petri_net(log, {"translucent_variant": "IMto"})
    pm4py.write.write_pnml(model, i_m, f_m, "models/IMto/"+str(number))
    imto_fitness.append(pm4py.conformance.fitness_alignments(ground_truth, model, i_m, f_m)["log_fitness"])
    imto_precision.append(translucent_precision_score(ground_truth, model, i_m, f_m))

    model, i_m, f_m = discover_petri_net(log, {"translucent_variant": "IMtf"})
    pm4py.write.write_pnml(model, i_m, f_m, "models/IMtf/"+str(number))
    imtf_fitness.append(pm4py.conformance.fitness_alignments(ground_truth, model, i_m, f_m)["log_fitness"])
    imtf_precision.append(translucent_precision_score(ground_truth, model, i_m, f_m))

    model, i_m, f_m = discover_petri_net(log, {"translucent_variant": "IMts"})
    pm4py.write.write_pnml(model, i_m, f_m, "models/IMts/"+str(number))
    imts_fitness.append(pm4py.conformance.fitness_alignments(ground_truth, model, i_m, f_m)["log_fitness"])
    imts_precision.append(translucent_precision_score(ground_truth, model, i_m, f_m))

im_f1 = compute_f_1_scores(im_fitness, im_precision)
imto_f1 = compute_f_1_scores(imto_fitness, imto_precision)
imtf_f1 = compute_f_1_scores(imtf_fitness, imtf_precision)
imts_f1 = compute_f_1_scores(imts_fitness, imts_precision)

result_im = pd.DataFrame({"Fitness": im_fitness, "Precision": im_precision, "F1_Score": im_f1})
result_im.to_csv("im_result.csv", sep=",", index=False)

result_imto = pd.DataFrame({"Fitness": imto_fitness, "Precision": imto_precision, "F1_Score": imto_f1})
result_imto.to_csv("imto_result.csv", sep=",", index=False)

result_imtf = pd.DataFrame({"Fitness": imtf_fitness, "Precision": imtf_precision, "F1_Score": imtf_f1})
result_imtf.to_csv("imtf_result.csv", sep=",", index=False)

result_imts = pd.DataFrame({"Fitness": imts_fitness, "Precision": imts_precision, "F1_Score": imts_f1})
result_imts.to_csv("imts_result.csv", sep=",", index=False)


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
plt.savefig("artificial_log_result_5.pdf", bbox_inches="tight")

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
plt.savefig("artificial_log_result_10.pdf", bbox_inches="tight")
