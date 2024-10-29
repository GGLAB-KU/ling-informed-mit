lang='cs,fi,hr,km,ko,lt,lv,my,nl,pt,sl,ta,uk,vi'
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

model_name='bigscience/bloom-7b1'
model_alias='bloom-7b1_2'
peft_method='lora'
lang_selection='ls_learned'

# use this for multigpu setting instead of python => torchrun --nproc_per_node=1 --master_port=1234

# python -m pdb -c continue 
python finetune_adapter_with_lang_priors.py \
--model_name_or_path ${model_name} \
--lang ${lang} \
--output_dir output/${model_alias}-X-${lang_selection}-${peft_method} \
--overwrite_output_dir \
--template_dir ./templates \
--learning_rate 3e-4 \
--load_in_8bit \
--fp16 \
--budget_constraint \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 16 \
--num_train_epochs 4 \
--model_max_length 768 \
--val_set_size 5000 \
--save_steps 2000 \
--eval_steps 2000 \
--logging_steps 100 \
--preprocessing_num_workers 5 \
--dataloader_num_workers 5 \
--peft_method ${peft_method} \
--remove_unused_columns True \
--ddp_find_unused_parameters False \
--save_total_limit 1 \
--group_by_length \
--report_to wandb \
--wandb_project bactrian-X \
--run_name ${model_alias}-X-${lang_selection}-${peft_method}

# evaluate
lm_eval --model hf \
--model_args pretrained=${model_name},peft=${peft_root}${model_alias}-X-${lang_selection}-${peft_method}  \
--tasks xcopa,xnli,xstorycloze,pawsx,xwinograd \
--device cuda:0 \
--batch_size 4