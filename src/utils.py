from datasets import load_dataset
import pandas as pd
import time
import json
import datetime
import os
import nltk

models = ["gpt3", "chatgpt"]
specific_models = ["text-davinci-003", "text-davinci-002", "code-davinci-002", "text-curie-001", "text-ada-001", "text-babbage-001", "davinci", "curie", "babbage", "ada"]

DATASET_MAP = {
  "ELI5_train": "according-to-data/eli5-kilt-train.jsonl",
  "ELI5_dev": "according-to-data/eli5-kilt-dev.jsonl",
  "HotpotQA_train": "according-to-data/hotpot-kilt-train.jsonl",
  "HotpotQA_dev": "according-to-data/hotpot-kilt-dev.jsonl",
  "NQ_KILT_train": "according-to-data/nq-kilt-train.jsonl",
  "NQ_KILT_dev": "according-to-data/nq-kilt-dev.jsonl",
  "TriviaQA_KILT_train": "according-to-data/triviaqa-kilt-train.jsonl",
  "TriviaQA_KILT_dev": "according-to-data/triviaqa-kilt-dev.jsonl",
  "ELI5_HF": "eli5",
  "HumanEval": "openai_humaneval",
  "nq_open": "nq_open",
  "trivia_qa": "trivia_qa",
  "hotpot": "hotpot_qa",
  "tqa_popularity": "according-to-data/tqa_popularity.csv",
  "nq_popularity": "according-to-data/nq_popularity.csv",
  "pubmedqa": "according-to-data/pubmedqa-test.json",
  "GBaker/MedQA-USMLE-4-options": "GBaker/MedQA-USMLE-4-options",
  "aldrinc/medicationqa": "aldrinc/medicationqa",
  "jhu-clsp/SARA": "jhu-clsp/SARA",
}

# load_dataset("trivia_qa", "unfiltered.nocontext", "test")
# load_dataset("nq_open", "validation") 


dataset_list = list(DATASET_MAP.keys())
NOT_KILT_DATASETS = ["ELI5_HF", "HumanEval", "pubmedqa", "GBaker/MedQA-USMLE-4-options", "aldrinc/medicationqa", "jhu-clsp/SARA"]


def load_jsonl(file_name):
    with open(file_name, 'r') as file:
        return [json.loads(line.strip()) for line in file]
    

def load_dataset_fn(location: str = "huggingface"):
  if location in ["openai_humaneval", "eli5", "nq_open", "NQ_KILT", "trivia_qa", "TriviaQA_KILT", "hotpot_qa", "pubmedqa", "GBaker/MedQA-USMLE-4-options", "aldrinc/medicationqa", "jhu-clsp/SARA"]:
    if location not in ["hotpot_qa", "trivia_qa"]:
      dataset = load_dataset(location) 

    if location == "eli5":
      return dataset["train_eli5"]
    elif location == "openai_humaneval":
      dataset = dataset["test"].to_pandas()
      dataset.columns = ["id", "title", "answers", "meta", "test"]
      return dataset.to_dict('records')
    elif location == "nq_open":
      dataset = dataset["validation"].to_pandas()
      dataset["id"] = "n/a"
      dataset["provenance"] = "empty n/a"
      dataset["title"] = dataset["question"]
      dataset["answers"] = dataset["answer"].apply(lambda x: x.tolist()) # NOTE list of answers
      dataset = dataset[["id", "title", "answers", "provenance"]]
      return dataset.to_dict('records')
    elif location == "trivia_qa":
      dataset = load_dataset("trivia_qa", "rc.wikipedia.nocontext", "validation")["validation"].to_pandas()
      dataset["provenance"] = "empty n/a"
      dataset["id"] = dataset["question_id"]
      dataset["title"] = dataset["question"]
      dataset["answers"] = dataset["answer"].apply(lambda x: x["aliases"].tolist())
      dataset = dataset[["id", "title", "answers", "provenance"]]
      return dataset.to_dict('records')
    elif location == "GBaker/MedQA-USMLE-4-options":
      dataset = load_dataset("GBaker/MedQA-USMLE-4-options", "test")["test"].to_pandas()
      dataset["provenance"] = "empty n/a"
      dataset['id'] = dataset.index
      def combine_mc_dict(x):
        formatted = ""
        for option in ["A", "B", "C", "D"]:
          formatted += option + ": " + x[option] + "\n"
        return formatted
      dataset["options_formatted"] = dataset["options"].apply(lambda x: combine_mc_dict(x))
      dataset["title"] = dataset.apply(lambda x: x["question"] + "\nOptions:\n" + x["options_formatted"], axis=1)
      dataset["answers"] = dataset["answer_idx"].apply(lambda x: [x])
      dataset = dataset[["id", "title", "answers", "provenance"]]
      return dataset.to_dict('records')
    elif location == "aldrinc/medicationqa":
      dataset = load_dataset(location, "train")["train"].to_pandas()
      dataset["title"] = dataset["Question"]
      dataset["answers"] = dataset["Answer"].apply(lambda x: [x])
      dataset["provenance"] = "empty n/a"
      dataset["id"] = dataset.index
      dataset = dataset[["id", "title", "answers", "provenance"]]
      return dataset.to_dict('records')
    elif location == "jhu-clsp/SARA":
      dataset = load_dataset(location, data_files={"test": "nli/test.jsonl"})["test"].to_pandas()
      dataset["title"] = dataset.apply(lambda x: "Premise: " + x["text"] + "\nHypothesis: " + x["question"], axis=1)
      dataset["answers"] = dataset["answer"].apply(lambda x: [x])
      dataset["provenance"] = "empty n/a"
      dataset["id"] = dataset["id"]
      dataset = dataset[["id", "title", "answers", "provenance"]]
      return dataset.to_dict('records')
    elif location == "hotpot_qa":
      dataset = load_dataset("hotpot_qa", "fullwiki")["validation"].to_pandas()
      dataset["title"] = dataset["question"]
      dataset["answers"] = dataset["answer"].apply(lambda x: [x])

      def reverse_lookup(title, titles):
        for t in titles:
          if t in title:
            return True
        return False

      def get_provenance_hotpot(row: dict):
        # get the support by the "sent_id" key of the "supporting_facts" key
        all_contexts = []
        titles_to_get = row["supporting_facts"]["title"]
        for idx in range(len(row["context"]["title"])):
          if row["context"]["title"][idx] in titles_to_get or reverse_lookup(row["context"]["title"][idx], titles_to_get):
            all_contexts.append(" ".join(row["context"]["sentences"][idx]))
        if not len(all_contexts): # fullwiki settings does not always have the right provenance
          all_contexts.append("empty n/a")
        return " ".join(all_contexts)
      
      dataset["provenance"] = dataset.apply(lambda x: get_provenance_hotpot(x), axis=1)
      dataset = dataset[["id", "title", "answers", "provenance"]]
      return dataset.to_dict('records')
  else:
    if ".jsonl" in location or ".json" in location:
      df = pd.DataFrame(load_jsonl(location))
    else:
      df = pd.read_csv(location, index_col=0)
    

    if len(df.columns) > 4:
      df.columns = ["id", "title", "answers", "provenance", "popularity"]
    else:
      df.columns = ["id", "title", "answers", "provenance"]
    return df.to_dict('records')
  

def write_results(params, results, to_look_at, output_path: str = "results"):
    all_output = {
      "params": params,
      "results": results,
      "pred": [item["pred"] for item in to_look_at],
      "refs": [item["references"] for item in to_look_at],
      "questions": [item["question"] for item in to_look_at],
      "provenances": [item["provenances"] for item in to_look_at] if "provenances" in to_look_at[0] else None,
    }

    if ".json" in output_path:
      with open(output_path, "w") as f:
        json.dump(all_output, f, indent=2)
    else:
      pretty_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
      if not os.path.isdir(output_path):
        os.makedirs(output_path)
      with open(os.path.join(output_path, f"results_{pretty_time}.json"), "w") as f:
        json.dump(all_output, f, indent=2)
      print(f"Wrote to {os.path.join(output_path, f'results_{pretty_time}.json')}")