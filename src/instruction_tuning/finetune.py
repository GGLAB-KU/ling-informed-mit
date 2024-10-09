import sys
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from transformers import Trainer, HfArgumentParser, \
    DataCollatorForSeq2Seq

from peft import (
    LoraConfig,
    get_peft_model,
)
from src.instruction_tuning.prompter import Prompter
from src.instruction_tuning.shared import BactrianDataArguments, BactrianTrainingArguments, \
    get_model_and_tokenizer, setup_dataset

# code inspired from => https://github.dev/mbzuai-nlp/bactrian-x

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    # base model arguments
    model_name_or_path: Optional[str] = field(default=None)
    load_in_8bit: bool = field(
        default=True, metadata={"help": "Whether to convert the loaded model into mixed-8bit quantized model."}
    )
    # LoRA parameters
    lora_r: int = field(default=8, metadata={"help": "Lora rank."})
    lora_alpha: int = field(default=16, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout."})
    lora_target_modules: str = field(default="query_key_value",
                                     metadata={"help": "Names of the modules to apply Lora to."})


def train(model_args: ModelArguments,
          data_args: BactrianDataArguments,
          training_args: BactrianTrainingArguments):
    model, tokenizer = get_model_and_tokenizer(model_args,
                                               training_args,
                                               DEFAULT_PAD_TOKEN,
                                               DEFAULT_EOS_TOKEN,
                                               DEFAULT_BOS_TOKEN,
                                               DEFAULT_UNK_TOKEN)

    prompter = Prompter(training_args.prompt_template_name, training_args.template_dir)

    config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.lora_target_modules.split(','),
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    train_data, val_data = setup_dataset(data_args,
                                         training_args.lang,
                                         prompter,
                                         tokenizer,
                                         IGNORE_INDEX)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    trainer = Trainer(
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
    parser = HfArgumentParser((ModelArguments, BactrianDataArguments, BactrianTrainingArguments))
    parsed_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args = parsed_args
