from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM, pipeline, T5ForConditionalGeneration, T5Tokenizer, set_seed, LlamaForCausalLM, LlamaTokenizer

import torch
import pandas as pd
import os
import json
import argparse
import tqdm
from torch.utils.data import Dataset              
import asyncio  
import time
from transformers.pipelines import pipeline


set_seed(2)


model_loaded = None
tokenizer = None

class PromptDataset(Dataset):  
    def __init__(self, list_of_examples):
        super().__init__()
        self.examples = list_of_examples

    def __len__(self):                                                              
        return len(self.examples)                                                                 

    def __getitem__(self, i):                                                       
        return self.examples[i]                                                


async def predict(input_prompts, model_type="t5", temperature: float = 0.3, specific_name: str = "google/flan-t5-xxl", batch_size: int = 1, **kwargs):
    global model_loaded
    global tokenizer

    if type(input_prompts[0]) == list:
        is_qa_two_step = True
    else:
        is_qa_two_step = False

    # dataset = PromptDataset(input_prompts)
    if model_type == "t5":
        # load examples to classify
        print("Loading T5 based model..", specific_name)
        if model_loaded is None:
            tokenizer = T5Tokenizer.from_pretrained(specific_name)
            model = T5ForConditionalGeneration.from_pretrained(specific_name, torch_dtype=torch.bfloat16).cuda()
            text2text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)
            model_loaded = model
        else:
            model = model_loaded
            text2text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)
    
    elif model_type in ["llama"]:
        if model_loaded is None:
            tokenizer = LlamaTokenizer.from_pretrained(specific_name)
            model = LlamaForCausalLM.from_pretrained(specific_name, torch_dtype=torch.float16).cuda()
            text2text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
            text2text_generator.tokenizer.pad_token_id = model.config.eos_token_id
            model_loaded = model
        else:
            model = model_loaded.cuda()
            text2text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
            text2text_generator.tokenizer.pad_token_id = model.config.eos_token_id
    elif model_type in ["mpt"]:
        config = AutoConfig.from_pretrained(
            'mosaicml/mpt-7b-instruct',
            trust_remote_code=True
        )
        # config.attn_config['attn_impl'] = 'triton'

        model = AutoModelForCausalLM.from_pretrained(
            'mosaicml/mpt-7b-instruct',
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        text2text_generator = TextGenerationPipeline(model=model, device=0, tokenizer=tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(specific_name, torch_dtype=torch.float16)
        # if "gpt-j" in specific_name:
        tokenizer = AutoTokenizer.from_pretrained(specific_name, padding_side='left')
        # else:
            # tokenizer = AutoTokenizer.from_pretrained(specific_name)
        text2text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, batch_size=batch_size)
        text2text_generator.tokenizer.pad_token_id = model.config.eos_token_id

    # print(f"Num Parameters: {model.num_parameters()}")
    if is_qa_two_step:
        generated_strings = []
        for out in tqdm.tqdm(text2text_generator(input_prompts[0], batch_size=batch_size, max_new_tokens=512), total=len(input_prompts)):
            generated_strings.append(item["generated_text"] for item in out)
        generated_strings = [next(item).replace(input_prompts[0][i], "").strip() for i, item in enumerate(generated_strings)]
        prompts2 = [item.replace("ANSWER", generated_strings[i]) for i, item in enumerate(input_prompts[1])]
        
        second_round = []
        for out in tqdm.tqdm(text2text_generator(prompts2, batch_size=batch_size, max_length=2000), total=len(input_prompts)):
            second_round.append(item["generated_text"] for item in out)
        second_round = [next(item).replace(prompts2[i], "").strip() for i, item in enumerate(second_round)]

        generated_strings = [answer + ";" + explain for (answer, explain) in zip(generated_strings, second_round)]

    else:
        start_time = time.time()
        print(start_time)
        generated_strings = []
        for out in tqdm.tqdm(text2text_generator(input_prompts, batch_size=batch_size, max_length=1024), total=len(input_prompts)):
            generated_strings.append(item["generated_text"] for item in out)
        assert len(generated_strings) == len(input_prompts)
        print(time.time() - start_time)
        generated_strings = [next(item).replace(input_prompts[i], "") for i, item in enumerate(generated_strings)]
    return list(generated_strings), 0



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    print(predict(["What is capitol of France?"]))
