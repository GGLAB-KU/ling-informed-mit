import os
import re

root_directory = '/home/gsoykan/Desktop/dev/multilingual-adapters/src/instruction_tuning/scripts/analysis'
new_files_directory = '/path/to/where/you/want/new/scripts/'  # Update this with the correct path


def create_new_scripts(filename, content, script_dir):
    base_name = os.path.splitext(filename)[0]
    for i in range(1, 5):
        new_model_alias = f"{model_alias}_add_{i}"
        new_content = re.sub(r"model_alias='[^']+'", f"model_alias='{new_model_alias}'", content)
        new_file_name = f"{base_name}_{new_model_alias}.sh"
        with open(os.path.join(script_dir, new_file_name), 'w') as new_file:
            new_file.write(new_content)


# python run_experiment_with_template.py --exp_name mgpt_ls_random --script_name finetune_lang_selection_random_mgpt_cluster --use_a40

if __name__ == '__main__':
    script_directories = list(
        map(lambda x: os.path.join(root_directory, x), ['bloom1_7b', 'bloom3b', 'bloom7b', 'mgpt', 'mt5']))
    for script_directory in script_directories:
        for script_file in os.listdir(script_directory):
            if script_file.endswith('.sh'):
                print(script_file)
                with open(os.path.join(script_directory, script_file), 'r') as file:
                    file_content = file.read()
                    model_alias_match = re.search(r"model_alias='([^']+)'", file_content)
                    if model_alias_match:
                        model_alias = model_alias_match.group(1)
                        if 'selection_resource' not in script_file or '_add_' in model_alias:
                            continue
                        create_new_scripts(script_file, file_content, script_directory)
                    else:
                        raise Exception(f"No model alias for {script_file}")
