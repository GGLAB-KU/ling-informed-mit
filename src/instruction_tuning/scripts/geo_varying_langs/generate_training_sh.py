import argparse
import os


def generate_sh_file(lang: str,
                     model_alias: str,
                     lang_selection: str,
                     model_name: str,  # 'ai-forever/mGPT' | 'bigscience/bloom-7b1'
                     batch_size: int,  # 4 | 2
                     grad_accumulation_steps: int,  # 8 | 16
                     output_dir: str
                     ):
    sh_content = f"""\
#!/bin/bash

lang='{lang}'
WORLD_SIZE=1
CUDA_VISIBLE_DEVICES=0

current_user=$(whoami)
if [ "$current_user" = "gsoykan20" ]; then
    peft_root='/scratch/users/gsoykan20/projects/multilingual-adapters/src/instruction_tuning/output/'
elif [ "$current_user" = "gosahin" ]; then
    peft_root='/scratch/users/gosahin/hpc_run/multilingual-adapters/src/instruction_tuning/output/'
else
    echo "Unknown user: $current_user"
    exit 1
fi

model_name='{model_name}'
model_alias='{model_alias}'
peft_method='lora'
lang_selection='{lang_selection}'

# use this for multigpu setting instead of python => torchrun --nproc_per_node=1 --master_port=1234

# python -m pdb -c continue 
python finetune_adapter_with_lang_priors.py \\
--model_name_or_path ${{model_name}} \\
--lang ${{lang}} \\
--output_dir output/${{model_alias}}-X-${{lang_selection}}-${{peft_method}} \\
--overwrite_output_dir \\
--template_dir ./templates \\
--learning_rate 3e-4 \\
--load_in_8bit \\
--fp16 \\
--budget_constraint \\
--per_device_train_batch_size {str(batch_size)} \\
--per_device_eval_batch_size {str(batch_size)} \\
--gradient_accumulation_steps {str(grad_accumulation_steps)} \\
--num_train_epochs 4 \\
--model_max_length 768 \\
--val_set_size 5000 \\
--save_steps 2000 \\
--eval_steps 2000 \\
--logging_steps 100 \\
--preprocessing_num_workers 5 \\
--dataloader_num_workers 5 \\
--peft_method ${{peft_method}} \\
--remove_unused_columns True \\
--ddp_find_unused_parameters False \\
--save_total_limit 1 \\
--group_by_length \\
--report_to wandb \\
--wandb_project bactrian-X \\
--run_name ${{model_alias}}-X-${{lang_selection}}-${{peft_method}}

# evaluate
lm_eval --model hf \\
--model_args pretrained=${{model_name}},peft=${{peft_root}}${{model_alias}}-X-${{lang_selection}}-${{peft_method}}  \\
--tasks xcopa,xnli,xstorycloze,pawsx,xwinograd \\
--device cuda:0 \\
--batch_size {str(batch_size * 2)}
"""
    os.makedirs(output_dir, exist_ok=True)
    num_langs, exp_id = model_alias.split('_')[-2:]
    file_name = f"{output_dir}/finetune_lang_selection_geo_langs_{num_langs}_{exp_id}.sh"
    with open(file_name, 'w') as file:
        file.write(sh_content)

    os.chmod(file_name, 0o755)  # Make the file executable
    print(f"Generated {file_name}")


def read_geo_selections(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    selections = []
    current_num_langs = None
    current_selections = []

    for line in lines:
        line = line.strip()
        if line.startswith('**** LANGUAGE COUNT UPDATED'):
            if current_selections:
                selections.append((current_num_langs, current_selections))
                current_selections = []
            current_num_langs = int(line.split(',')[1].strip().split()[0])
        else:
            current_selections.append(line)

    if current_selections:
        selections.append((current_num_langs, current_selections))

    return selections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate shell scripts for varying size experiments training')
    parser.add_argument('--model_name',
                        choices=['ai-forever/mGPT',
                                 'bigscience/bloom-7b1'],
                        # required=True,
                        help='Name of the model',
                        default='bigscience/bloom-7b1')
    args = parser.parse_args()

    model_name = args.model_name
    if model_name == 'ai-forever/mGPT':
        batch_size = 4
        grad_accumulation_steps = 8
        model_short_name = 'mgpt'
        output_dir = './mgpt'
    elif model_name == 'bigscience/bloom-7b1':
        batch_size = 2
        grad_accumulation_steps = 16
        model_short_name = 'bloom-7b1'
        output_dir = './bloom7b'
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    selections = read_geo_selections('geo_language_selections.txt')

    for num_langs, langs in selections:
        for i, lang in enumerate(langs):
            model_alias = f'{model_short_name}_{num_langs}_{i + 1}'
            lang_selection = f'ls_geo_size'
            generate_sh_file(lang,
                             model_alias,
                             lang_selection,
                             model_name=model_name,
                             batch_size=batch_size,
                             grad_accumulation_steps=grad_accumulation_steps,
                             output_dir=output_dir)
