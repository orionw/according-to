from datasets import load_dataset
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


nq_grounded_path = "according-to-generations/popularity_files/results_2023-05-18-23-14-54.json"
tqa_grounded_path = "according-to-generations/popularity_files/results_2023-05-18-23-19-32.json"
nq_null_path = "according-to-generations/popularity_files/results_2023-05-18-23-49-57.json"
tqa_null_path = "according-to-generations/popularity_files/results_2023-05-18-23-54-07.json"

with open(nq_grounded_path) as f:
    nq_grounded = json.load(f)

with open(tqa_grounded_path) as f:
    tqa_grounded = json.load(f)

with open(nq_null_path) as f:
    nq_null = json.load(f)

with open(tqa_null_path) as f:
    tqa_null = json.load(f)

base_nq_df = pd.read_csv("according-to-generations/popularity_files/nq_popularity.csv", index_col=0)
base_tqa_df = pd.read_csv("according-to-generations/popularity_files/tqa_popularity.csv", index_col=0)

tqa_popularity = base_tqa_df["popularity"]
nq_popularity = base_nq_df["popularity"]

ranges = [(-1, 0), (0, 10), (10, 100), (100, 1000), (1000, 10000)]
interval = pd.IntervalIndex.from_tuples(ranges)

# bin the data by popularity according to the ranges above
nq_popularity_cut = pd.cut(nq_popularity, interval)
tqa_popularity_cut = pd.cut(tqa_popularity, interval)


for dataset_name, json_obj, cut_list in [("nq", nq_grounded, nq_popularity_cut), ("nq_null", nq_null, nq_popularity_cut), ("tqa", tqa_grounded, tqa_popularity_cut), ("tqa_null", tqa_null, tqa_popularity_cut)]:
    percent_overlap_list = json_obj["results"]["datasketch"]["percent_overlap_list"]

    palette = sns.color_palette("rocket")
    reversed_palette = list(reversed(palette))
    sns.set_palette(reversed_palette)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=cut_list, y=percent_overlap_list, errorbar=('se', 1), dodge=False, ax=ax)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = '0'
    ax.set_xticklabels(labels)

    plt.xlabel('Frequency in Pre-training Data', size=18, labelpad=10, fontname="Times New Roman")
    plt.ylabel(f"QUIP-Score", labelpad=5, size=18, fontname="Times New Roman")
    pretty_name = "Natural Questions" if 'nq' in dataset_name else "TriviaQA"
    plt.title(pretty_name, fontname="Times New Roman", size=20)
    plt.tight_layout()
    plt.savefig(f"popularity_scatter_{dataset_name}_overlap.pdf")
    plt.savefig(f"popularity_scatter_{dataset_name}_overlap.png")
    plt.close()
    print(f"popularity_scatter_{dataset_name}_overlap.png")


