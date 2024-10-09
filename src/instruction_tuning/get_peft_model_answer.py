import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import os
import sys
import json

from torch import Tensor
from tqdm import tqdm
import shortuuid
# import ray
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, Trainer, GenerationConfig, \
    LlamaForCausalLM, HfArgumentParser
from peft import PeftModel

from src.instruction_tuning.data import iso2lang

from src.instruction_tuning.prompter import Prompter
import gensim.downloader as gensim_api

from src.utils import get_typological_language_features


@dataclass
class GetModelAnswerConfig:
    base_model: str = field(default="bigscience/bloom-560m", metadata={"help": "Path to pretrained model"})
    load_8bit: bool = field(default=True, metadata={"help": "Whether to load the model in 8-bit"})
    peft_weights: Optional[str] = field(default=None, metadata={"help": "Path to PEFT weights"})
    template_dir: str = field(default="./templates", metadata={"help": "Directory containing prompt templates"})
    prompt_template_name: str = field(default="bactrian", metadata={"help": "Name of the prompt template to use"})
    question_file: str = field(default=None, metadata={"help": "Path to the question file", "required": True})
    answer_file: str = field(default="answer.jsonl", metadata={"help": "Path to the output answer file"})
    num_gpus: int = field(default=1, metadata={"help": "Number of GPUs to use"})


@torch.inference_mode()
def get_model_answers(args, question_jsons):
    prompter = Prompter(args.prompt_template_name, args.template_dir)
    tokenizer_class = LlamaTokenizer if 'llama' in args.base_model else AutoTokenizer
    model_class = LlamaForCausalLM if 'llama' in args.base_model else AutoModelForCausalLM

    tokenizer = tokenizer_class.from_pretrained(args.base_model)
    model = model_class.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if args.peft_weights:
        model = PeftModel.from_pretrained(
            model,
            args.peft_weights,
            torch_dtype=torch.float16,
        )

    # unwind broken decapoda-research config
    if 'llama' in args.base_model:
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

    if not args.load_8bit:
        model.half()

    peft_config = model.peft_config.get('default', {})

    ans_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = json.loads(line)
        idx = ques_json["question_id"]
        qs = ques_json["text"]
        lang_code = ques_json["lang"]

        prompt = prompter.generate_prompt(instruction=qs, input='', )  # language=lang)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()

        generation_config = GenerationConfig(
            temperature=0.1,
            no_repeat_ngram_size=3,
        )

        additional_kwargs = {}

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=1024,
                **additional_kwargs
            )
        s = generation_output.sequences[0]
        outputs = tokenizer.decode(s)

        ans_id = shortuuid.uuid()
        # print(outputs)
        ans_jsons.append(
            {
                "question_id": idx,
                "text": outputs,
                "answer_id": ans_id,
                "metadata": {},
            }
        )
    return ans_jsons


def run_eval(args: GetModelAnswerConfig):
    # split question file into num_gpus files
    ques_jsons = []
    with open(args.question_file, "r") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)

    chunk_size = len(ques_jsons) // args.num_gpus
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        ans_handles.append(
            # get_model_answers.remote(args, ques_jsons[i : i + chunk_size]
            get_model_answers(args, ques_jsons[i: i + chunk_size])
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ans_handle)

    answer_directory = os.path.dirname(args.answer_file)
    if not os.path.exists(answer_directory):
        os.makedirs(answer_directory)

    with open(args.answer_file, "w", encoding='utf-8') as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = HfArgumentParser(GetModelAnswerConfig)
    parsed_args: Tuple[GetModelAnswerConfig] = parser.parse_args_into_dataclasses()
    config = parsed_args[0]

    # ray.init()
    run_eval(config)
