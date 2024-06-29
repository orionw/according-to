import os
import json
import argparse
import pandas as pd

from kilt.knowledge_source import KnowledgeSource

base_path = "KILT/"

with open(f"{base_path}/kilt/configs/all_data.json", "r") as fin:
    mapping = json.load(fin)

ks = KnowledgeSource()
print(ks.get_num_pages())

def get_with_id(wiki_dict: dict):
    inst = ks.get_page_by_id(int(wiki_dict["wikipedia_id"]))

    try:
        assert inst["wikipedia_title"] == wiki_dict["title"]
    except Exception as e:
        print("titles mismatch", inst["wikipedia_title"], wiki_dict["title"])

    assert wiki_dict["start_paragraph_id"] == wiki_dict["end_paragraph_id"]
    if "start_character" in wiki_dict:
        return inst["text"][wiki_dict["start_paragraph_id"]][wiki_dict["start_character"]:wiki_dict["end_character"]]
    else:
        return inst["text"][wiki_dict["start_paragraph_id"]]

def process_answer_and_provenance(x: list, type: str) -> list:
    provenances = []
    answers = []
    for item in x:
        if "provenance" in item:
            provenances.append(" ".join([get_with_id(i) for i in item["provenance"]]))
        if "answer" in item:
            answers.append(item["answer"])
    if "wiki" == type:
        return provenances
    elif "reference" == type:
        return answers
    else:
        raise NotImplementedError(type)

def parse():
    all_dict_items = {**mapping["Open Domain QA Dev"], **mapping["Dialogue Dev"]}
    for dataset_name, dataset_path in all_dict_items.items():
        # if dataset_name == "ELI5":
        #     dataset_path = mapping["Open Domain QA Dev"]["ELI5"] # ELI5 does not have provenances in the train set
        print(dataset_name, dataset_path)

        data = []
        with open(os.path.join(base_path, dataset_path), "r") as fin:
            for line in fin:
                data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        df["references"] = df.output.apply(lambda x: process_answer_and_provenance(x, "reference"))
        df["wiki_references"] = df.output.apply(lambda x: process_answer_and_provenance(x, "wiki"))

        df = df[df["wiki_references"].apply(lambda x: x != [])]

        if "meta" in df.columns:
            df = df.drop(["output", "meta"], axis=1)
        else:
            df = df.drop(["output"], axis=1)

        df.columns = ["id", "question", "references", "wiki_references"]
        print(len(df))
        id2idx = {i: idx for idx, i in enumerate(df.id)}
        with open(os.path.join(base_path, dataset_path).replace(".jsonl", "_final_id2idx.json"), "w") as fout:
            json.dump(id2idx, fout)
        df.to_json(os.path.join(base_path, dataset_path).replace(".jsonl", "_final_eval.jsonl"), orient="records", lines=True)



if __name__ == "__main__":
    parse()