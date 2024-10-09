import logging
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Tuple, Callable, Union

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, concatenate_datasets, DatasetDict
from peft import (
    prepare_model_for_kbit_training,
)
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, TrainingArguments, PreTrainedTokenizer, \
    PreTrainedModel, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM

from src.instruction_tuning.prompter import Prompter


@dataclass
class ModelWithAdaptionArguments:
    # base model arguments
    model_name_or_path: Optional[str] = field(default=None)
    load_in_8bit: bool = field(
        default=True, metadata={"help": "Whether to convert the loaded model into mixed-8bit quantized model."}
    )
    # PEFT parameters
    peft_method: str = field(default="adaption_prompt",
                             metadata={"help": "peft method for fine-tuning, e.g. adaption_prompt, "
                                               "p_tuning, prompt_tuning, prefix_tuning"})
    adapter_len: int = field(default=10, metadata={"help": "num adapter tokens"})
    # llama-adapter params...
    use_tanh_for_adaption_gate: bool = field(default=False, metadata={"help": "enables tanh for adaption gates"})
    neg_adapter_layers: int = field(default=2, metadata={"help": "minus from total blocks"})
    # additional modality configs
    use_additional_modality: bool = field(default=False, metadata={"help": "flag for additional modality usage"})
    additional_modality_processor_alias: Optional[str] = field(default=None,
                                                               metadata={"help": "implies additional modality type"})
    additional_modality_adapter_spans: Optional[str] = field(default=None,
                                                             metadata={"help": "use it as follows 0-5 or"
                                                                               " 0-5-10-15 for complementary modality"})


@dataclass
class BactrianDataArguments:
    data_path: Optional[str] = field(default='MBZUAI/Bactrian-X', metadata={"help": "Path to the training file."})
    source_max_length: Optional[int] = field(
        default=256, metadata={"help": "only for seq2seq: Maximum length of source. "
                                       "Sequences will be right padded (and possibly truncated)."}
    )
    model_max_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    val_set_size: Optional[int] = field(default=2000, metadata={"help": "The validation set size. For loss checking."})


@dataclass
class BactrianTrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer to use."})
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training."}
    )
    lang: str = field(default="tr", metadata={
        "help": "The language or language list separated by `,`, dataset will be downlaoded from HF Hub."})
    template_dir: str = field(default="./templates", metadata={"help": "Prompt template dir."})
    prompt_template_name: str = field(default="bactrian", metadata={"help": "Prompt template file to use."})
    evaluation_strategy: str = field(default="steps", metadata={"help": ""})
    save_strategy: str = field(default="steps", metadata={"help": ""})
    wandb_project: str = field(default="bactrian", metadata={"help": "Weight & Bias (W&B) project name."})
    budget_constraint: bool = field(
        default=False,
        metadata={"help": "This makes number of iterations constant "
                          "for epoch by dividing training data with number of languages"}
    )


@dataclass
class BactrianSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer to use."})
    fp16: bool = field(
        default=False, metadata={"help": "Whether to use fp16 16-bit"
                                         " (mixed) precision training instead "
                                         "of 32-bit training. Not recommand for mT5"}
    )
    lang: str = field(default="tr", metadata={
        "help": "The language or language list separated by `,`, dataset will be downlaoded from HF Hub."})
    template_dir: str = field(default="./templates", metadata={"help": "Prompt template dir."})
    prompt_template_name: str = field(default="bactrian_seq2seq", metadata={"help": "Prompt template file to use."})
    evaluation_strategy: str = field(default="steps", metadata={"help": ""})
    save_strategy: str = field(default="steps", metadata={"help": ""})
    wandb_project: str = field(default="bactrian", metadata={"help": "Weight & Bias (W&B) project name."})
    budget_constraint: bool = field(
        default=False,
        metadata={"help": "This makes number of iterations constant "
                          "for epoch by dividing training data with number of languages"}
    )

def setup_logging(logger,
                  model_args,
                  training_args: BactrianTrainingArguments):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")


def get_seq2seq_model_and_tokenizer(model_args,
                                    training_args: BactrianSeq2SeqTrainingArguments,
                                    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        # device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        from accelerate import Accelerator

        device_index = Accelerator().process_index
        print('accelerate device index =>', device_index)

        device_map = {"": device_index}

    if 'wandb' in training_args.report_to:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        # torch_dtype=torch.float16,  # Result in collapse
        load_in_8bit=model_args.load_in_8bit,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=True,
    )

    if model_args.load_in_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    else:
        model.enable_input_require_grads()

    return model, tokenizer


def get_model_and_tokenizer(model_args,
                            training_args: BactrianTrainingArguments,
                            default_pad_token: str,
                            default_eos_token: str,
                            default_bos_token: str,
                            default_unk_token: str
                            ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        # device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        from accelerate import Accelerator

        device_index = Accelerator().process_index
        print('accelerate device index =>', device_index)

        device_map = {"": device_index}

    if 'wandb' in training_args.report_to:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        load_in_8bit=model_args.load_in_8bit,
        device_map=device_map,
    )

    tokenizer_class = LlamaTokenizer if "llama" in model_args.model_name_or_path else AutoTokenizer
    tokenizer: PreTrainedTokenizer = tokenizer_class.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )

    # llama has no pad_token
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(dict(pad_token=default_pad_token),
                                             tokenizer=tokenizer,
                                             model=model)
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": default_eos_token,
                "bos_token": default_bos_token,
                "unk_token": default_unk_token,
            }
        )

    if model_args.load_in_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    else:
        model.enable_input_require_grads()

    return model, tokenizer


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel
):
    """
    Copied from https://github.com/bofenghuang/stanford_alpaca/blob/eb5b171d9b103a12a8e14e0edca9cbc45fe1d512/train.py#L75-L95
    Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def generate_and_tokenize_prompt(data_point,
                                 prompter,
                                 tokenizer,
                                 ignore_index,
                                 model_max_length,
                                 small_lm_tokenizer,
                                 sentence2vec,
                                 source_max_length: Optional[int] = None,
                                 is_seq2seq: bool = False):
    """
    @gsoykan to @mbzuai - this is so smart, esp. setting up the labels...
    """
    if not is_seq2seq:
        full_prompt = prompter.generate_prompt(data_point['instruction'],
                                               data_point['input'],
                                               data_point['output'])
        user_prompt = prompter.generate_prompt(data_point['instruction'], data_point['input'])
        user_prompt_len = len(tokenizer(user_prompt, truncation=True, max_length=model_max_length)["input_ids"])
        tokenized_full_prompt = tokenizer(full_prompt + tokenizer.eos_token, truncation=True,
                                          max_length=model_max_length)
        tokenized_full_prompt['labels'] = [ignore_index] * user_prompt_len + tokenized_full_prompt["input_ids"].copy()[
                                                                             user_prompt_len:]
        tokenized_full_prompt.pop('attention_mask')
    else:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        source_ids = tokenizer(text=user_prompt, truncation=True, max_length=source_max_length)["input_ids"]
        target_ids = \
            tokenizer(text_target=data_point["output"], truncation=True,
                      max_length=model_max_length - source_max_length)[
                "input_ids"]
        tokenized_full_prompt = {"input_ids": source_ids, "labels": target_ids}

    # apply small lm tokenizer just to use 'instruction' + 'input'
    # there is an option to use with the prompt template or without
    # however, since we won't be training the small lm
    # we should not get the encodings of the template structure
    if small_lm_tokenizer is not None:
        raw_user_prompt = (data_point['instruction'] if data_point['instruction'] else '') + ' ' + (
            data_point['input'] if data_point['input'] else '')
        raw_user_prompt = raw_user_prompt.strip()
        if 'intfloat/multilingual-e5' in small_lm_tokenizer.name_or_path:
            raw_user_prompt = f'query: {raw_user_prompt}'
        small_lm_input = {f'small_lm_{k}': v for (k, v) in small_lm_tokenizer(
            raw_user_prompt,
            padding="max_length",
            truncation=True,
        ).items()}
        tokenized_full_prompt.update(small_lm_input)

    if sentence2vec is not None:
        raw_user_prompt = (data_point['instruction'] if data_point['instruction'] else '') + ' ' + (
            data_point['input'] if data_point['input'] else '')
        raw_user_prompt = raw_user_prompt.strip()
        tokenized_full_prompt.update({
            'modality_input': sentence2vec(raw_user_prompt)
        })

    return tokenized_full_prompt


def setup_dataset(data_args: BactrianDataArguments,
                  langs: str,
                  prompter: Prompter,
                  tokenizer: PreTrainedTokenizer,
                  ignore_index: int,
                  is_seq2seq: bool = False
                  ):
    # load dataset from HF Hub
    if langs != '':
        all_dataset = [load_dataset(data_args.data_path, lang) for lang in langs.split(',')]
    merged_dataset = concatenate_datasets(list(map(lambda x: x['train'], all_dataset)))
    raw_datasets = DatasetDict({'train': merged_dataset})

    if data_args.val_set_size > 0:
        train_val_data = raw_datasets['train'].train_test_split(test_size=data_args.val_set_size,
                                                                shuffle=True,
                                                                seed=42)
    else:
        raise ValueError("val_set_size must be larger than 0.")

    # Determine model_max_length for truncation
    source_max_length = data_args.source_max_length
    model_max_length = data_args.model_max_length

    generate_and_tokenize_prompt_fn = partial(
        generate_and_tokenize_prompt,
        prompter=prompter,
        tokenizer=tokenizer,
        ignore_index=ignore_index,
        model_max_length=model_max_length,
        source_max_length=source_max_length,
        is_seq2seq=is_seq2seq,
        small_lm_tokenizer=None,
        sentence2vec=None,
    )

    # @gsoykan - below is commented out by @mbzuai
    # with training_args.main_process_first(desc="dataset map tokenization"):
    train_data = train_val_data["train"].map(
        generate_and_tokenize_prompt_fn,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=next(iter(raw_datasets.values())).column_names,
        desc="preprocess train data set",
    )
    val_data = train_val_data["test"].map(
        generate_and_tokenize_prompt_fn,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=next(iter(raw_datasets.values())).column_names,
        desc="preprocess val data set",
    )

    return train_data, val_data


def setup_dataset_with_lang(data_args: BactrianDataArguments,
                            langs: str,
                            prompter: Prompter,
                            tokenizer: PreTrainedTokenizer,
                            ignore_index: int,
                            lang2vec: Optional[Callable[[str], Union[int, np.ndarray, Tensor]]],
                            small_lm_tokenizer: Optional[PreTrainedTokenizer],
                            sentence2vec: Optional[Callable[[str], Union[int, np.ndarray, Tensor]]],
                            is_seq2seq: bool = False,
                            budget_constraint: bool = False
                            ):
    if langs != '':
        all_dataset = {lang: load_dataset(data_args.data_path, lang) for lang in langs.split(',')}

    def add_lang_column(example, lang_vec_or_idx):
        example['modality_input'] = lang_vec_or_idx
        return example

    if lang2vec is not None:
        for lang_code, dataset in all_dataset.items():
            lang_vec_or_idx = lang2vec(lang_code)
            all_dataset[lang_code] = dataset.map(lambda x: add_lang_column(x, lang_vec_or_idx))

    all_dataset = list(all_dataset.values())

    if budget_constraint:
        # get parallel samples in budget constraint setting
        # and they will be same instances (same id) across experiments too
        all_dataset = list(map(lambda x: x.sort('id'), all_dataset))
        num_langs = len(langs.split(','))
        dataset_size = len(all_dataset[0]['train'])
        single_dataset_size = dataset_size // num_langs
        random.seed(42)
        dataset_indices = random.sample(range(len(all_dataset[0]['train'])),
                                        k=single_dataset_size)
        merged_dataset = concatenate_datasets(list(map(
            lambda x: x['train'].select(indices=dataset_indices),
            all_dataset)))
    else:
        merged_dataset = concatenate_datasets(list(map(lambda x: x['train'], all_dataset)))

    raw_datasets = DatasetDict({'train': merged_dataset})

    if data_args.val_set_size > 0:
        train_val_data = raw_datasets['train'].train_test_split(test_size=data_args.val_set_size,
                                                                shuffle=True,
                                                                seed=42)
    else:
        raise ValueError("val_set_size must be larger than 0.")

    # Determine model_max_length for truncation
    source_max_length = data_args.source_max_length
    model_max_length = data_args.model_max_length

    generate_and_tokenize_prompt_fn = partial(
        generate_and_tokenize_prompt,
        prompter=prompter,
        tokenizer=tokenizer,
        ignore_index=ignore_index,
        model_max_length=model_max_length,
        small_lm_tokenizer=small_lm_tokenizer,
        source_max_length=source_max_length,
        is_seq2seq=is_seq2seq
    )

    columns_to_remove = set(next(iter(raw_datasets.values())).column_names) - {'modality_input',
                                                                               'small_lm_input_ids',
                                                                               'small_lm_attention_mask'}

    if os.getenv("ON_CLUSTER") == "False":
        print('setup dataset with lang - not on cluster...')
        train_val_data['train'] = train_val_data['train'].select(range(1000))

    # @gsoykan - below is commented out by @mbzuai
    # with training_args.main_process_first(desc="dataset map tokenization"):
    train_data = train_val_data["train"].map(
        generate_and_tokenize_prompt_fn,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=columns_to_remove,
        desc="preprocess train data set",
        fn_kwargs={
            'sentence2vec': sentence2vec
        }
    )
    val_data = train_val_data["test"].map(
        generate_and_tokenize_prompt_fn,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=columns_to_remove,
        desc="preprocess val data set",
        fn_kwargs={
            'sentence2vec': sentence2vec
        }
    )

    return train_data, val_data
