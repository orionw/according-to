import os
import pandas as pd
import random
import time
import asyncio

from run_gpt3 import predict as predict_gpt3
from run_huggingface import predict as predict_huggingface
from evaluation import do_evaluation, get_quotes
from utils import DATASET_MAP, NOT_KILT_DATASETS, load_dataset_fn, models, specific_models, write_results

random.seed(42)

def choose_predict_func(model_name: str):
  if model_name in ["gpt4", "gpt3", "chatgpt"]:
    return predict_gpt3
  else:
    return predict_huggingface


async def run_examples(prompt_before: str, prompt_after: str, model_name: str = "gpt3", specific_name: str = "text-davinci-003", num_examples: int = 2, temperature: float = 0.3, dataset_name: str = "ELI5", uses_gradio: bool = True, system_prompt: str = None, output_format: str = "semicolon", batch_size: int = 1):
    params = {
      "prompt_before": prompt_before,
      "prompt_after": prompt_after,
      "system_prompt": system_prompt,
      "model_name": model_name,
      "specific_name": specific_name,
      "num_examples": num_examples,
      "temperature": temperature,
      "dataset": dataset_name,
      "output_format": output_format,
      "batch_size": batch_size,
    }
    questions = []
    refs = []
    provenances = []
    base_dataset = load_dataset_fn(DATASET_MAP[dataset_name])
    num_examples = min(num_examples, len(base_dataset))
    print(f"Predicting for {num_examples} examples")
    for inst in range(num_examples):
      question = base_dataset[inst]["title"]
      ref = base_dataset[inst]["answers"]
      if dataset_name == "huggingface":
        ref = ref["text"]
      questions.append(question)
      refs.append(ref)
      if dataset_name not in NOT_KILT_DATASETS and "popularity" not in dataset_name.lower():
        provenances.append(base_dataset[inst]["provenance"])

    if pd.isnull(prompt_before):
      prompt_before = ""
    if pd.isnull(prompt_after):
      prompt_after = ""
    if "eli5" in dataset_name.lower() or model_name.lower() in ["gpt3", "gpt4", "chatgpt"]:
      full_prompts = [prompt_before + q.capitalize() + prompt_after for q in questions]
      full_prompts = [p.strip() for p in full_prompts] # matters more for smaller models
    else: 
      ending = "?\nAnswer string only:" if "nq" in dataset_name.lower() else "\nAnswer string only:"
      full_prompts_1 = ["Output the answer only. " + q.capitalize() + ending for q in questions]
      full_prompts_2 = ["Question: {q}\nAnswer: ANSWER\n\n".format(q=q) + "\n\nGive a detailed explanation for why this is true." + prompt_after + "\nExplanation:" for q in questions]
      full_prompts = [full_prompts_1, full_prompts_2]


    start_time = time.time()
    predict = choose_predict_func(model_name)
    outputs = await predict(full_prompts, model_type=model_name, temperature=temperature, specific_name=specific_name, batch_size=batch_size)

    model_time = round(time.time() - start_time, 2)
    
    if dataset_name not in NOT_KILT_DATASETS and "popularity" not in dataset_name.lower():
      assert len(provenances) == len(refs) == len(outputs), f"{len(provenances)} {len(refs)} {len(outputs)}"
    prompt_ends_with_quote = (prompt_before[-1] == '"') if len(prompt_before) > 0 else False
    results_dicts = do_evaluation(dataset_name, outputs, refs, provenances, prompt_ends_with_quote, output_format)

    combined = {
      "datasketch": results_dicts["datasketch"],
      "answers": results_dicts["base"],
      "model_generate_time": model_time,
    }
    if dataset_name not in NOT_KILT_DATASETS and "popularity" not in dataset_name:
      combined["provenance"] = results_dicts["provenance"] 

    
    to_look_at = []
    for idx in range(num_examples):
      cur_inst = {
        "question": questions[idx],
        "pred": outputs[idx],
        "references": refs[idx],
      }
      if dataset_name not in NOT_KILT_DATASETS and "popularity" not in dataset_name:
        cur_inst["provenances"] = provenances[idx]
      to_look_at.append(cur_inst)

    write_results(params, combined, to_look_at, "results")
    

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--file', help='The JSON file to load what to run', type=str, required=True)
  parser.add_argument('-o', '--output_path', help='Where to write the results to', type=str, default="results")
  args = parser.parse_args()

  if not os.path.isdir(args.output_path):
    os.makedirs(args.output_path)

  df = pd.read_json(args.file, lines=True)
  for idx, row in df.iterrows():
    # no need to use them, they're saved
    print(row)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_examples(row["prompt_before"], row["prompt_after"], row["model_name"], row["specific_name"], row["num_examples"], row["temperature"], row["dataset_name"], uses_gradio=False, system_prompt=row["system_prompt"] if "system_prompt" in row else None, output_format=row["output_format"] if "output_format" in row else "semicolon", batch_size=row["batch_size"] if "batch_size" in row else 1))
    