BASE_MODEL='bigscience/bloom-3b'
# PEFT_WEIGHTS='./output/bloom-3b-X-adapter_cluster'
PEFT_WEIGHTS='./output/bloom-3b-X-adapter-lang_cluster'

python get_peft_model_answer.py \
--base_model ${BASE_MODEL} \
--load_8bit True \
--template_dir ./templates \
--peft_weights ${PEFT_WEIGHTS} \
--prompt_template_name bactrian \
--question_file ./data/tr_demo_questions.jsonl \
--answer_file ./output/answer.jsonl \
--num_gpus 1