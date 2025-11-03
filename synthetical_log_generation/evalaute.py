import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from translucent_precision.main import translucent_precision_score
from evaluation.modify_logs import add_start_and_end_translucent
from evaluation.modify_logs import add_start_and_end_evaluation
from translucent_discovery.translucent_inductive_miner.translucent_base import discover_petri_net
from evaluation.translucent_f_1_score import compute_f_1_scores
import matplotlib.pyplot as plt


def create_logs():
    ground_truth = pd.read_csv("../evaluation/output_logs/road_traffic/ground_truth.csv", sep=",")
    ground_truth = pm4py.format_dataframe(ground_truth, case_id='case:concept:name', activity_key='concept:name',timestamp_key='time:timestamp')
    ground_truth = log_converter.apply(ground_truth, variant=log_converter.Variants.TO_EVENT_LOG)
    variants = pm4py.statistics.variants.log.get.get_variants(ground_truth)
    i = 1
    while i <= len(variants):
        variants_to_consider = pm4py.filter_variants_top_k(ground_truth, i)
        dataframe = pm4py.convert_to_dataframe(variants_to_consider)
        dataframe.to_csv("output_logs/road_traffic/sub_logs/exported_" + str(i) + ".csv", sep=",")
        i += 1
    print(len(variants))


def evaluate(path_dir, number_of_sublogs):
    ground_truth = pd.read_csv(path_dir + "ground_truth.csv", sep=",")
    ground_truth = pm4py.format_dataframe(ground_truth, case_id='case:concept:name', activity_key='concept:name',timestamp_key='time:timestamp')
    ground_truth = log_converter.apply(ground_truth, variant=log_converter.Variants.TO_EVENT_LOG)
    im_fitness = []
    im_precision = []

    imto_fitness = []
    imto_precision = []

    imtf_fitness = []
    imtf_precision = []

    imts_fitness = []
    imts_precision = []

    for number in range(3, number_of_sublogs + 1, 1):
        print(number)
        log = pd.read_csv(path_dir + "sub_logs/exported_" + str(number) + ".csv", sep=",")
        log = pm4py.format_dataframe(log, case_id='case:concept:name', activity_key='concept:name',timestamp_key='time:timestamp')
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
        model, i_m, f_m = pm4py.discover_petri_net_inductive(log)
        im_fitness.append(pm4py.conformance.fitness_alignments(ground_truth, model, i_m, f_m)["log_fitness"])
        im_precision.append(translucent_precision_score(ground_truth, model, i_m, f_m))
        print("im finished")
        model, i_m, f_m = discover_petri_net(log, {"translucent_variant": "IMto"})
        pm4py.view_petri_net(model, i_m, f_m)
        imto_fitness.append(pm4py.conformance.fitness_alignments(ground_truth, model, i_m, f_m)["log_fitness"])
        imto_precision.append(translucent_precision_score(ground_truth, model, i_m, f_m))
        print("imto finished")
        model, i_m, f_m = discover_petri_net(log, {"translucent_variant": "IMtf"})
        imtf_fitness.append(pm4py.conformance.fitness_alignments(ground_truth, model, i_m, f_m)["log_fitness"])
        imtf_precision.append(translucent_precision_score(ground_truth, model, i_m, f_m))

        model, i_m, f_m = discover_petri_net(log, {"translucent_variant": "IMts"})
        imts_fitness.append(pm4py.conformance.fitness_alignments(ground_truth, model, i_m, f_m)["log_fitness"])
        imts_precision.append(translucent_precision_score(ground_truth, model, i_m, f_m))

    im_f1 = compute_f_1_scores(im_fitness, im_precision)
    imto_f1 = compute_f_1_scores(imto_fitness, imto_precision)
    imtf_f1 = compute_f_1_scores(imtf_fitness, imtf_precision)
    imts_f1 = compute_f_1_scores(imts_fitness, imts_precision)

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
        for algo, color in zip(algorithms, algorithm_colors):
            ax.plot(data[algo][criterion], label=algo, marker='o', color=color)

    # Add legend below subplots
    fig.legend(algorithms, loc='lower center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True,
               ncol=len(algorithms))

    # Adjust layout
    plt.tight_layout()

    # Save plot as a tight PDF
    plt.savefig("road_traffic.pdf", bbox_inches="tight")


evaluate("../evaluation/output_logs/road_traffic/", 142)
