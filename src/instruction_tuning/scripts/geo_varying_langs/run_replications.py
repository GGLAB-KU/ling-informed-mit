import os
import argparse
import subprocess
import time

from dotenv import load_dotenv

load_dotenv()


def generate_commands(script_dir, model_id):
    count = 0
    for script_file in os.listdir(script_dir):
        if 'finetune_lang_selection' not in script_file:
            continue

        if script_file.endswith('.sh'):
            type_match, _, lang_count, exp_id = script_file.split('_')[-4:]
            exp_id = exp_id.split('.')[0]
            exp_name = f"{model_id}_ls_{type_match}_size_{lang_count}_{exp_id}"

            script_name = script_file[:-3]  # without .sh extension

            incomplete_logs = ['bloom7b_ls_geo_size_4_3.log',
                               'bloom7b_ls_geo_size_6_1.log',
                               'bloom7b_ls_geo_size_6_2.log',
                               'bloom7b_ls_geo_size_7_3.log',
                               'bloom7b_ls_geo_size_8_2.log',
                               'bloom7b_ls_geo_size_8_3.log',
                               'bloom7b_ls_geo_size_9_2.log']

            incomplete_log_components = list(map(lambda x: x[:-4].split('_'), incomplete_logs))

            if not any(map(lambda x: model_id == x[0] and type_match == x[2] and int(lang_count) == int(x[-2]) and int(
                    exp_id) == int(x[-1]),
                           incomplete_log_components)):
                continue

            command = f"python run_experiment_with_template_geo_varying_langs.py --exp_name {exp_name} --script_name {script_name} --model_alias {model_id}"

            if model_id not in ['bloom7b', 'mt5']:
                command += " --use_a40"

            print(command)
            count += 1

            if os.getenv("ON_CLUSTER") == "True":
                time.sleep(2)
                subprocess.run(command, shell=True, check=True)

    print('num exp ', count)


# you should run this from src/instruction_tuning/scripts
# python ./geo_varying_langs/run_replications.py --model bloom7b
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments for a given model.')
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        # default='mgpt',
                        # choices=['mgpt', 'mt5', 'bloom7b', 'bloom3b', 'bloom1_7b'],
                        choices=['mgpt', 'bloom7b', ],
                        help='Model identifier (e.g., bloom7b, mgpt) to filter the scripts.')

    args = parser.parse_args()
    model_identifier = args.model

    if os.getenv("ON_CLUSTER") == "False":
        script_directory = f'/home/gsoykan/Desktop/dev/multilingual-adapters/src/instruction_tuning/scripts/geo_varying_langs/{model_identifier}'
    else:
        script_directory = f'./geo_varying_langs/{model_identifier}'

    generate_commands(script_directory, model_identifier)
