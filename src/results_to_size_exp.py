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
    "gpt3-text-ada-001": "ada",
    "gpt3-text-babbage-001": "babbage",
    "gpt3-text-curie-001": "curie",
    "gpt3-text-davinci-003": "davinci",
    "t5-google/flan-t5-small": "small",
    "t5-google/flan-t5-base": "base",
    "t5-google/flan-t5-large": "large",
    "t5-google/flan-t5-xl": "xl",
    "t5-google/flan-t5-xxl": "xxl",
}


def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_folders', nargs='+', help='The CSV folders to read in.')
    args = parser.parse_args()

    # palette = sns.color_palette("rocket")
    # reversed_palette = list(reversed(palette))
    # sns.set_palette([reversed_palette[1], reversed_palette[-1]])

    # Read in the TSV files using Pandas
    dfs = []
    print(args.csv_folders)
    for csv_folder in args.csv_folders:
        for csv_file in glob.glob(os.path.join(csv_folder, '*.csv')):
            if "results.csv" == csv_file.split("/")[-1]:
                continue
            print(f"Loading", csv_file)
            df = pd.read_csv(csv_file, sep='\t')
            if "openai" in csv_file:
                df["Model Type"] = "OpenAI"
                is_flan = False
            else:
                df["Model Type"] = "FLAN-T5"
                is_flan = True
            # print(df)
            dfs.append(df)


    df = pd.concat(dfs, ignore_index=True)
    df["Overlap"] = df["ds-25-overlap-mean"]
    df["Model Size"] = df.model.apply(lambda x: MODEL_MAP[x])
    df = df.sort_values(by=["Model Size"])
    df["Prompt"] = df.prompt.apply(lambda x: "Null" if x == "{{ QUESTION }}" else "Grounded")
    print(df["Model Size"].value_counts())

    order = ["small", "base", "large", "xl", "xxl", "ada", "babbage", "curie", "davinci"]
    # set df in the correct order
    df["Model Size"] = pd.Categorical(df["Model Size"], order)
    df["Prompt"] = pd.Categorical(df["Prompt"], ["Null", "Grounded"])

    # set the size of the plot
    # plt.figure(figsize=(8, 6)
    COLOR = 'black'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    sns.set_theme(style="white")
    # to_use_palette = sns.color_palette(list(reversed(["#4D79D1".lower(), "#CE77FF".lower()])))
    to_use_palette = [sns.color_palette("pastel")[0], sns.color_palette("pastel")[4]]
    sns.set_palette(to_use_palette)
    plt.rcParams["font.family"] = "Times New Roman"
    # make figure size wider and shorter
    fig, ax = plt.subplots(figsize=(8, 4))
    # df = df.sort_values(by=["Prompt"], ascending=False)
    sns.lineplot(data=df, x="Model Size", y="Overlap", hue="Prompt", errorbar=("se", 1), linewidth=3.0, style="Prompt", style_order=["Grounded", "Null"], ax=ax)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.errorbar(df['Model Size'], df['Overlap'], yerr=df['ds-25-overlap-std']*2, fmt='o', color="black")
    # set legend outside of plot
    ax.set_yticks([0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    ax.set_ylim(0.11, 0.34)
    plt.xticks(fontsize=22, fontname="Times New Roman")
    plt.yticks(fontsize=20, fontname="Times New Roman")
    if not is_flan:
        ax.get_legend().remove()
    # plt.rcParams["font.family"] = "Times New Roman"
    if is_flan:
        plt.legend(title="Prompt Type", fontsize=17, loc='upper left', title_fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels, title="Prompt Type", fontsize=17, loc='upper left', title_fontsize=18)
    plt.xlabel('Model Size', size=22, labelpad=10, fontname="Times New Roman")
    plt.ylabel("QUIP-Score", labelpad=5, size=22, fontname="Times New Roman")
    plt.tight_layout()
    model_type = "_flan" if is_flan else "_openai"
    plt.savefig(f"results/size_exp{model_type}.png")
    plt.savefig(f"results/size_exp{model_type}.pdf")
    df.to_csv(f"results/size_exp{model_type}.tsv")

    print("Wrote tsv file to {} and figure to {}".format("according_to_wikipedia/results/size_exp.csv", f"according_to_wikipedia/results/size_exp{model_type}.png"))

if __name__ == '__main__':
    main()
    # python src/results_to_size_exp.py results/size_exp_flan_t5/
    # python src/results_to_size_exp.py results/size_exp_openai/
