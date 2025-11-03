import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

im_results = pd.read_csv("im_result.csv")
im_fitness = im_results["Fitness"]
im_precision = im_results["Precision"]
im_f1 = im_results["F1_Score"]

imto_results = pd.read_csv("imto_result.csv")
imto_fitness = imto_results["Fitness"]
imto_precision = imto_results["Precision"]
imto_f1 = imto_results["F1_Score"]

imtf_results = pd.read_csv("imtf_result.csv")
imtf_fitness = imtf_results["Fitness"]
imtf_precision = imtf_results["Precision"]
imtf_f1 = imtf_results["F1_Score"]

imts_results = pd.read_csv("imts_result.csv")
imts_fitness = imts_results["Fitness"]
imts_precision = imts_results["Precision"]
imts_f1 = imts_results["F1_Score"]

algorithms = ['IM', 'IMto', 'IMtf', 'IMts']
criteria = ['Fitness', 'Precision', 'F1 Score']
algorithm_colors = ['blue', 'green', 'red', 'purple']
data = {
    'IM': {
        'Fitness': im_fitness,
        'Precision': im_precision,
        'F1 Score': im_f1
    },
    'IMto': {
        'Fitness': imto_fitness,
        'Precision': imto_precision,
        'F1 Score': imto_f1
    },
    'IMtf': {
        'Fitness': imtf_fitness,
        'Precision': imtf_precision,
        'F1 Score': imtf_f1
    },
    'IMts': {
        'Fitness': imts_fitness,
        'Precision': imts_precision,
        'F1 Score': imts_f1
    }
}

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot data for each criterion
for i, criterion in enumerate(criteria):
    ax = axs[i]
    ax.set_title(criterion, fontsize=16)
    ax.set_xlabel('Number of Variants', fontsize=14)
    ax.set_xticks(np.insert(np.arange(4, len(data["IM"]["Fitness"]), 5), 0, 0))
    ax.set_xticklabels([str(i) for i in np.insert(np.arange(5, len(data["IM"]["Fitness"]) + 1, 5), 0, 1)])
    ax.tick_params(axis='x', which='major', labelsize=12, size=5)
    ax.tick_params(axis='y', which='major', labelsize=12, size=5)
    for algo, color in zip(algorithms, algorithm_colors):
        ax.plot(data[algo][criterion], label=algo, marker='o', color=color)
# Add legend below subplots
fig.legend(algorithms, loc='lower center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True,
           ncol=len(algorithms))

# Adjust layout
plt.tight_layout()

# Save plot as a tight PDF
plt.savefig("artificial_log_result_5.pdf", bbox_inches="tight")