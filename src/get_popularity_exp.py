from datasets import load_dataset
import os
import json
import pandas as pd
import numpy as np
import random

random.seed(12)
np.random.seed(12)

nq_path_train = "/home/hltcoe/oweller/my_exps/datasketches-dev/according_to_wikipedia/long_tail_knowledge/output/qa_co_occurrence_split=the_pile_entity_map_nq_train_entities_qa_co_occurrence.json"
tqa_path_dev = "/home/hltcoe/oweller/my_exps/datasketches-dev/according_to_wikipedia/long_tail_knowledge/output/qa_co_occurrence_split=the_pile_entity_map_trivia_qa_unfiltered.nocontext_validation_entities.json"
tqa_path_train = "/home/hltcoe/oweller/my_exps/datasketches-dev/according_to_wikipedia/long_tail_knowledge/output/qa_co_occurrence_split=the_pile_entity_map_trivia_qa_unfiltered.nocontext_train_entities_qa_co_occurrence.json"
nq_path_dev = "/home/hltcoe/oweller/my_exps/datasketches-dev/according_to_wikipedia/long_tail_knowledge/output/qa_co_occurrence_split=the_pile_entity_map_nq_validation_entities.json"

with open(tqa_path_train) as f:
    tqa_train = json.load(f)

with open(nq_path_train) as f:
    nq_train = json.load(f)

with open(tqa_path_dev) as f:
    tqa_dev = json.load(f)

with open(nq_path_dev) as f:
    nq_dev = json.load(f)

tqa_full_train = load_dataset("trivia_qa", "unfiltered.nocontext", "train")["train"].to_pandas()
tqa_full_dev = load_dataset("trivia_qa", "unfiltered.nocontext", "validation")["validation"].to_pandas()
nq_full_train = load_dataset("nq_open")["train"].to_pandas()
nq_full_dev = load_dataset("nq_open")["validation"].to_pandas()

tqa_full_train["popularity"] = tqa_train
tqa_full_dev["popularity"] = tqa_dev
nq_full_train["popularity"] = nq_train
nq_full_dev["popularity"] = nq_dev

nq_full = pd.concat([nq_full_train, nq_full_dev])
tqa_full = pd.concat([tqa_full_train, tqa_full_dev])

ranges = [(-1, 0), (1, 10), (10, 100), (100, 1000), (1000, 10000)]
interval = pd.IntervalIndex.from_tuples(ranges)

# bin the data by popularity according to the ranges above
nq_full["popularity_bin"] = pd.cut(nq_full["popularity"], interval)
tqa_full["popularity_bin"] = pd.cut(tqa_full["popularity"], interval)

n_to_get = 400
nq_popularity = nq_full.groupby("popularity_bin").apply(lambda x: x.sample(n=min(n_to_get, len(x)))).reset_index(drop=True)
tqa_popularity = tqa_full.groupby("popularity_bin").apply(lambda x: x.sample(n=min(n_to_get, len(x)))).reset_index(drop=True)



# ["id", "title", "answers", "provenance", "popularity"]
nq_popularity["id"] = range(len(nq_popularity))
nq_popularity["title"] = nq_popularity["question"]
nq_popularity["answers"] = nq_popularity["answer"]
nq_popularity["provenance"] = ""
nq_popularity["popularity"] = nq_popularity["popularity"].astype(int)

tqa_popularity["id"] = tqa_popularity["question_id"]
tqa_popularity["title"] = tqa_popularity["question"]
tqa_popularity["answers"] = tqa_popularity["answer"].apply(lambda x: x["aliases"])
tqa_popularity["provenance"] = ""

# drop all other columns
nq_popularity = nq_popularity[["id", "title", "answers", "provenance", "popularity"]]
tqa_popularity = tqa_popularity[["id", "title", "answers", "provenance", "popularity"]]

nq_popularity.to_csv("nq_popularity.csv")
tqa_popularity.to_csv("tqa_popularity.csv")



