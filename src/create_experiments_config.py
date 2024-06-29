import json
import pandas as pd
import copy
import argparse

def create_config(args):
    data = []
    with open(args.config_file) as fin:
        for line in fin:
            data.append(json.loads(line))

    prompts = pd.read_json(args.prompt_file, lines=True)

    all_exps = []
    for item in data:
        cur_copy = copy.deepcopy(item)
        cur_copy["batch_size"] = 4
        if args.debug:
            cur_copy["num_examples"] = 10
        all_exps.append(cur_copy) # null prompt

        for (idx, row) in prompts.iterrows():
            cur_copy = copy.deepcopy(item)
            cur_copy["batch_size"] = 4

            if args.debug:
                cur_copy["num_examples"] = 10

            if row["location"] == "before":
                if row["location_specific"] == "before":
                    cur_copy["prompt_before"] = row["prompt"] + cur_copy["prompt_before"]
                else:
                    cur_copy["prompt_before"] = cur_copy["prompt_before"] + row["prompt"]
            elif row["location"] == "after":
                if row["location_specific"] == "after":
                    cur_copy["prompt_after"] = cur_copy["prompt_after"] + row["prompt"]
                else:
                    cur_copy["prompt_after"] = row["prompt"] + cur_copy["prompt_after"]

            all_exps.append(cur_copy) # null prompt

    pd.DataFrame(all_exps).to_json(args.output_file, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, required=True)
    parser.add_argument("-p", "--prompt_file", type=str, required=True)
    parser.add_argument("-o", "--output_file", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    create_config(args)
    # config is the datasets and model used
    # prompt is the prompts used
    # example `python src/create_experiments_config.py -c configs/initial_experiments_chatgpt_pubmed.jsonl -p prompts/prompts_all.jsonl -o to_run.jsonl --debug`