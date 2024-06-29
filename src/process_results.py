import os
import glob
import json
import argparse
import pandas as pd

from utils import load_jsonl

def replace_newline(s):
    return s.replace("\n", "\\n")


def read_in(folder):
    all_results = []
    if folder.endswith(".json"):
        with open(folder, 'r') as f:
            all_results.append(json.load(f))
        return pd.DataFrame(all_results)
    
    recursive_paths = list(glob.glob(folder + '/**/*.json', recursive=True))
    for file_path in recursive_paths:
            with open(file_path, 'r') as f:
                all_results.append(json.load(f))
    df = pd.DataFrame(all_results)
    return df

def process_results(args):
    df = read_in(args.folder)

    # # TODO: param and results were switched, change later
    # if "dataset" in df.iloc[0]["results"]:
    #      results_key = "params"
    #      param_key = "results"
    # else:
    results_key = "results"
    param_key = "params"

    df["dataset"] = df.apply(lambda x: x[param_key]["dataset"], axis=1)
    df["model"] = df.apply(lambda x: f'{x[param_key]["model_name"]}-{x[param_key]["specific_name"]}', axis=1)
    df["prompt"] = df.apply(lambda x: replace_newline(x[param_key]["prompt_before"]) + "{{ QUESTION }}" + replace_newline(x[param_key]["prompt_after"]), axis=1)
    
    ## without quotes
    df["ds-25-overlap"] = df.apply(lambda x: x[results_key]["datasketch"]["avg_percent_overlap"], axis=1)

    df["em"] = df.apply(lambda x: x[results_key]["answers"]["em"], axis=1)
    df["f1"] = df.apply(lambda x: x[results_key]["answers"]["f1"], axis=1)
    df["rougel"] = df.apply(lambda x: x[results_key]["answers"]["rougel"], axis=1)
    df["bleu"] = df.apply(lambda x: x[results_key]["answers"]["bleu"], axis=1)

    df["average_tokens"] = df.apply(lambda x: x[results_key]["total_tokens_used"] / len(x[results_key]["answers"]["f1_list"]), axis=1)
    
    
    KEEP_COLS = ["prompt", "dataset", "model", "ds-25-overlap", "em", "f1", "rougel", "bleu", "average_tokens"]
    df_only = df[KEEP_COLS]
    for model, model_df in df_only.groupby("model"):
        model_name = model.replace("/", "-")
        for dataset, dataset_df in model_df.groupby("dataset"):
            data_for_dataset = []
            for (prompt, prompt_df) in dataset_df.groupby("prompt"):
                # create mean and std dev of columns
                for column in KEEP_COLS[3:]:
                    prompt_df[f"{column}-mean"] = prompt_df[column].mean()
                    prompt_df[f"{column}-std"] = prompt_df[column].std()
                data_for_dataset.append(prompt_df.iloc[[0]])
            KEEP_AGG_COLS = [col for col in prompt_df.columns if "std" in col or "mean" in col or col in KEEP_COLS[:3]]
            pd.concat(data_for_dataset)[KEEP_AGG_COLS].sort_values("ds-25-overlap-mean", ascending=False).to_csv(os.path.join(args.folder, f"{model_name}_{dataset}.csv"), index=False, sep="\t")
    df_only.to_csv(os.path.join(args.folder, "results.csv"), index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='path to folder containing results (can be recursive)', type=str)
    args = parser.parse_args()
    process_results(args)