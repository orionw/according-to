from argparse import ArgumentParser

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import logging
import sys
import time

# need OPENAI_API_KEY eviron variable set


logger = logging.getLogger(__name__)


async def predict(input_prompts, model_type='chatgpt', temperature: float = 0.3, specific_name: str = "text-davinci-003", system_prompt: str = None, **kwargs):
    # batch over prompts in batches of 3000
    all_generations = []
    batch_num = 200 if model_type == "gpt4" else 350
    for i in range(0, len(input_prompts), batch_num):
        print("Batching for prompt", i, "to", i+batch_num, "of", len(input_prompts), file=sys.stderr, flush=True)
        batch = input_prompts[i:i+batch_num]
        generations = await predict_batch(batch, model_type=model_type, temperature=temperature, specific_name=specific_name, system_prompt=system_prompt, **kwargs)
        all_generations.extend(generations)
        print("Done, sleeping...")
        if model_type == "gpt4":
            time.sleep(60)
        elif batch_num + i < len(input_prompts):
            time.sleep(30)

    return all_generations


async def predict_batch(input_prompts, model_type='chatgpt', temperature: float = 0.3, specific_name: str = "text-davinci-003", system_prompt: str = None, **kwargs):
    if "batch_size" in kwargs:
        del kwargs["batch_size"]

    if model_type in ["gpt4", 'chatgpt']:
        llm = ChatOpenAI(model_name=specific_name, temperature=temperature, request_timeout=600, max_retries=10)
        if system_prompt is None:
            batch_messages = [[HumanMessage(content=prompt)] for prompt in input_prompts]
        else:
            batch_messages = [[HumanMessage(content=prompt), SystemMessage(content=system_prompt)] for prompt in input_prompts]
        llm_result = await llm.agenerate(batch_messages, **kwargs)

    elif model_type in ['gpt3']:
        llm = OpenAI(model_name=specific_name, temperature=temperature, request_timeout=600, max_retries=10)
        llm_result = await llm.agenerate(input_prompts, **kwargs)
    else:
        raise NotImplementedError()

    outputs = [[gen.text.strip() for gen in result] for result in llm_result.generations]
    logging.info(llm_result.llm_output)
    generations = [" ".join(entry) for entry in outputs]
    return generations




if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument('')
    # # TODO add prompts
    # args = parser.parse_args()

    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )
    predict(["What is capitol of France?"])

