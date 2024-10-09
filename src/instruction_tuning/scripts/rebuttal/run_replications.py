import os
import argparse
import subprocess
import time

from dotenv import load_dotenv

load_dotenv()


def generate_commands(script_dir, model_id):
    count = 0
    for script_file in os.listdir(script_dir):
        if '_add_' not in script_file:
            continue

        if script_file.endswith('.sh'):
            type_match = script_file.split('_')[3]
            add_x_part = script_file.split('_')[-2] + '_' + script_file.split('_')[-1].split('.')[0]
            exp_name = f"{model_id}_ls_{type_match}_{add_x_part}"

            # skip already completed experiments...
            if type_match != "resource":
                continue

            script_name = script_file[:-3]

            command = f"python run_experiment_with_template_rebuttal.py --exp_name {exp_name} --script_name {script_name} --rebuttal_model_alias {model_id}"

            if model_id not in ['bloom7b', 'mt5']:
                command += " --use_a40"

            print(command)
            count += 1

            if os.getenv("ON_CLUSTER") == "True":
                time.sleep(2)
                subprocess.run(command, shell=True, check=True)

    print('num exp ', count)


# you should run this from src/instruction_tuning/scripts
# python ./rebuttal/run_replications.py --model bloom3b
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments for a given model.')
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        # default='bloom1_7b',
                        choices=['mgpt', 'mt5', 'bloom7b', 'bloom3b', 'bloom1_7b'],
                        help='Model identifier (e.g., bloom3b, mgpt) to filter the scripts.')

    args = parser.parse_args()
    model_identifier = args.model

    if os.getenv("ON_CLUSTER") == "False":
        script_directory = f'/home/gsoykan/Desktop/dev/multilingual-adapters/src/instruction_tuning/scripts/rebuttal/{model_identifier}'
    else:
        script_directory = f'./rebuttal/{model_identifier}'

    generate_commands(script_directory, model_identifier)
