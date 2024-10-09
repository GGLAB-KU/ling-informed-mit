import logging
import sys

import gensim.downloader as gensim_api
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import HfArgumentParser, \
    DataCollatorForSeq2Seq, AutoTokenizer, Seq2SeqTrainer

from peft import (
    get_peft_model, AdaptionPromptConfig, TaskType, PromptTuningConfig, PromptTuningInit, PrefixTuningConfig,
    PromptEncoderConfig, PromptEncoderReparameterizationType, LoraConfig,
)
from src.instruction_tuning.data import iso2lang
from src.instruction_tuning.prompter import Prompter
from src.instruction_tuning.shared import BactrianDataArguments, BactrianTrainingArguments, \
    setup_dataset_with_lang, ModelWithAdaptionArguments, BactrianSeq2SeqTrainingArguments, \
    get_seq2seq_model_and_tokenizer
from src.utils import get_typological_language_features, load_fasttext_model

load_dotenv()

# code inspired from => https://github.dev/mbzuai-nlp/bactrian-x

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def train(model_args: ModelWithAdaptionArguments,
          data_args: BactrianDataArguments,
          training_args: BactrianSeq2SeqTrainingArguments):
    if isinstance(training_args.lang, tuple):
        training_args.lang = training_args.lang[0]
    model, tokenizer = get_seq2seq_model_and_tokenizer(model_args,
                                                       training_args)

    prompter = Prompter(training_args.prompt_template_name, training_args.template_dir)

    spans = list(map(int, model_args.additional_modality_adapter_spans.split('-'))) \
        if model_args.additional_modality_adapter_spans is not None \
        else None
    if model_args.peft_method == 'adaption_prompt':
        config = AdaptionPromptConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            adapter_len=model_args.adapter_len,
            use_tanh_for_adaption_gate=model_args.use_tanh_for_adaption_gate,
            encoder_adapter_layers=model.encoder.config.num_hidden_layers - model_args.neg_adapter_layers,
            decoder_adapter_layers=model.decoder.config.num_hidden_layers,
            use_additional_modality=model_args.use_additional_modality,
            additional_modality_processor_alias=model_args.additional_modality_processor_alias,
            additional_modality_adapter_spans=spans
        )
    elif model_args.peft_method == 'prompt_tuning':
        config = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=model_args.adapter_len,
            prompt_tuning_init_text=None,
            tokenizer_name_or_path=None,
            inference_mode=False,
            use_additional_modality=model_args.use_additional_modality,
            additional_modality_processor_alias=model_args.additional_modality_processor_alias,
            additional_modality_adapter_spans=spans
        )
    elif model_args.peft_method == 'prefix_tuning':
        config = PrefixTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=model_args.adapter_len,
            prefix_projection=True,
            use_additional_modality=model_args.use_additional_modality,
            additional_modality_processor_alias=model_args.additional_modality_processor_alias,
            additional_modality_adapter_spans=spans
        )
    elif model_args.peft_method == 'p_tuning':
        config = PromptEncoderConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                                     num_virtual_tokens=model_args.adapter_len,
                                     # default params - begin
                                     encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
                                     encoder_hidden_size=None,
                                     encoder_num_layers=2,
                                     encoder_dropout=0.0,
                                     # default params - end
                                     use_additional_modality=model_args.use_additional_modality,
                                     additional_modality_processor_alias=model_args.additional_modality_processor_alias,
                                     additional_modality_adapter_spans=spans)
    elif model_args.peft_method == 'lora':
        config = LoraConfig(
            r=64,
            lora_alpha=16,
            # target_modules=model_args.lora_target_modules.split(','),
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
    else:
        raise ValueError(f'Unsupported peft method: {model_args.peft_method}')

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    train_data, val_data = setup_dataset_with_lang(data_args,
                                                   training_args.lang,
                                                   prompter,
                                                   tokenizer,
                                                   IGNORE_INDEX,
                                                   None,
                                                   None,
                                                   None,
                                                   is_seq2seq=True,
                                                   budget_constraint=training_args.budget_constraint)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=data_collator
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelWithAdaptionArguments, BactrianDataArguments, BactrianSeq2SeqTrainingArguments))
    parsed_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args = parsed_args
