import argparse
import pandas as pd
import os
import io
import glob

import seaborn as sns
import matplotlib.pyplot as plt

def remove_boilerplate(x: str):
    return x.replace("You are a highly intelligent & complex question-answer generative model. You take a question as an input and answer it by imitating the way a human gives short answers with a corresponding explanation. You answer should be short - only a few words.\\n\\nYour output format should be the answer, then a semicolon, then the explanation.\\n", "").replace("}}?", "}}").replace("}} ", "}}")

MODEL_MAP = {
    "gpt3-text-ada-001": "base",
    "gpt3-text-babbage-001": "large",
    "gpt3-text-curie-001": "xl",
    "gpt3-text-davinci-003": "xxl",
    "t5-google/flan-t5-small": "small",
    "t5-google/flan-t5-base": "base",
    "t5-google/flan-t5-large": "large",
    "t5-google/flan-t5-xl": "xl",
    "t5-google/flan-t5-xxl": "xxl",
    "t5-t5-base": "base",
    "t5-t5-large": "large",
    "t5-t5-3b": "xl",
    "t5-t5-11b": "xxl",
    "t5-google/t5-xxl-lm-adapt": "xxl"
}


def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    # read in two arguments for file paths that contain csv folders
    parser.add_argument('csv_folder_path1', help='The CSV folders to read in.')
    parser.add_argument('csv_folder_path2', help='The CSV folders to read in.')
    args = parser.parse_args()

    palette = sns.color_palette("rocket")
    reversed_palette = list(reversed(palette))

    # Read in the TSV files using Pandas
    dfs = []
    for csv_file in [args.csv_folder_path2,args.csv_folder_path1]:
        df = pd.read_csv(csv_file, sep='\t')
        if "flan-t5" in csv_file:
            df["Model Type"] = "FLAN-T5"
        else:
            df["Model Type"] = "T5"
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    df["Overlap"] = df["ds-25-overlap-mean"]
    df["Rouge-L"] = df["rougel-mean"]
    df["Model Size"] = df.model.apply(lambda x: MODEL_MAP[x])
    df = df.sort_values(by=["Model Size"])
    df["Prompt"] = df.prompt.apply(lambda x: "Null" if x == "{{ QUESTION }}" else "AccordingTo")
    print(df["Model Size"].value_counts())

    df = df[df["Model Size"] == "xxl"]

    order = ["small", "base", "large", "xl", "xxl"]
    # set df in the correct order
    df["Model Size"] = pd.Categorical(df["Model Size"], order)
    df["Prompt"] = pd.Categorical(df["Prompt"], ["Null", "AccordingTo"])


    # set the size of the plot
    plt.figure(figsize=(8, 6))
    COLOR = 'black'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    plt.rcParams['hatch.color'] = COLOR
    sns.set_theme(style="white")
    # to_use_palette = sns.color_palette(list(reversed(["#DAE8FC".lower(), "#CE77FF".lower()])))
    to_use_palette = [sns.color_palette("pastel")[0], sns.color_palette("pastel")[4]]
    sns.set_palette(to_use_palette)
    plt.rcParams["font.family"] = "Times New Roman"
    df["Prompt"] = df.prompt.apply(lambda x: "Grounded" if "Respond to this question using" in x else "Null")
    df = df.sort_values(by=["Prompt"], ascending=False)
    plt.rcParams['hatch.color'] = COLOR
    ax = sns.barplot(data=df, x="Model Type", y="Overlap", hue="Prompt", errorbar=('se', 1))
    patterns = ['/', '/', '', '']
    for i, bar in enumerate(ax.patches):
        bar.set_hatch(patterns[i])

    leg = ax.get_legend()
    leg.legend_handles[1].set_hatch('/')
    plt.xticks(fontsize=22, fontname="Times New Roman")
    plt.yticks(fontsize=20, fontname="Times New Roman")
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.xlabel('Model Size', size=22, labelpad=10, fontname="Times New Roman")
    plt.ylabel("QUIP-Score", labelpad=10, size=22, fontname="Times New Roman")
    plt.legend(title="Prompt Type", fontsize=18, loc='upper left', title_fontsize=20)
    # plt.errorbar(df['Model Size'], df['Overlap'], yerr=df['ds-25-overlap-std']*2, fmt='o')
    # set legend outside of plot
    plt.tight_layout()
    plt.savefig("results/size_exp_t5_comp.png")
    plt.savefig("results/size_exp_t5_comp.pdf")
    print("results/size_exp_t5_comp.png")
    df.to_csv("results/size_exp_t5_comp.tsv")

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    ax = sns.barplot(data=df, x="Model Type", y="Rouge-L", hue="Prompt", palette=[reversed_palette[0], reversed_palette[-2]], ci=None)
    
    patterns = ['', '', '/', '/']
    for i, bar in enumerate(ax.patches):
        bar.set_hatch(patterns[i])

    leg = ax.get_legend()
    leg.legend_handles[1].set_hatch('/')
    
    # plt.errorbar(df['Model Size'], df['Overlap'], yerr=df['ds-25-overlap-std']*2, fmt='o')
    # set legend outside of plot
    plt.xticks(fontsize=22, fontname="Times New Roman")
    plt.yticks(fontsize=20, fontname="Times New Roman")
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.xlabel('Model Size', size=22, labelpad=10, fontname="Times New Roman")
    plt.ylabel("Rouge-L", labelpad=10, size=22, fontname="Times New Roman")
    # plt.legend(title="Prompt Type", fontsize=18, loc='upper right', title_fontsize=20)
    ax.get_legend().remove()

    plt.tight_layout()
    plt.savefig("results/size_exp_t5_comp_rouge.png")
    plt.savefig("results/size_exp_t5_comp_rouge.pdf")
    df.to_csv("results/size_exp_t5_comp_rouge.tsv")

    print("Wrote tsv file to {} and figure to {}".format("according_to_wikipedia/results/size_exp_t5_comp_rouge.csv", "according_to_wikipedia/results/size_exp_t5_comp_rouge.png"))

if __name__ == '__main__':
    main()

    # python src/results_compare.py results/flan-t5-xxl/t5-google-flan-t5-xxl_ELI5_dev.csv results/t5-xxl-real/t5-t5-11b_ELI5_dev.csv
    # python src/results_compare.py results/flan-t5-xxl/t5-google-flan-t5-xxl_ELI5_dev.csv results/t5-adapt/t5-google-t5-xxl-lm-adapt_ELI5_dev.csv 
