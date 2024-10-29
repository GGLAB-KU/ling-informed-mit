import argparse
import os
import subprocess
import tempfile


def submit_job_for_experiment(template_path,
                              exp_name,
                              script_name,
                              model_alias: str,
                              num_gpus: int,
                              use_a40: bool = False):
    with open(template_path, 'r') as file:
        template = file.read()

    modified_script = template.replace('--job-name=<tmp>', f'--job-name={exp_name}')
    modified_script = modified_script.replace('--output=<tmp>.log', f'--output={exp_name}.log')
    modified_script = modified_script.replace('sh ./scripts/<tmp>.sh',
                                              f'sh ./scripts/geo_varying_langs/{model_alias}/{script_name}.sh')
    modified_script = modified_script.replace('--gres=gpu:1', f'--gres=gpu:{str(num_gpus)}')

    if use_a40:
        modified_script = modified_script.replace('--constraint=rtx_a6000', f'--constraint=nvidia_a40')

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Create a temporary file in the same directory as the script
    fd, temp_file_path = tempfile.mkstemp(suffix='.sh', dir=script_dir)
    with os.fdopen(fd, 'w') as temp_file:
        temp_file.write(modified_script)

    # Submit the job
    subprocess.run(['sbatch', temp_file_path])

    # Delete the temporary file
    os.remove(temp_file_path)


# sample command => python
# run_experiment_with_template_main_results.py
# --exp_name mgpt_ls_random
# --script_name finetune_lang_selection_random_mgpt_cluster
# --model_alias bloom1_7b
# --use_a40
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit a job for an experiment.')
    parser.add_argument('--exp_name',
                        type=str,
                        required=True,
                        help='Experiment name')
    parser.add_argument('--script_name',
                        type=str,
                        required=True,
                        help='Script name')
    parser.add_argument('--model_alias',
                        type=str,
                        required=True,
                        help='Model Alias (as in directory paths)')
    parser.add_argument('--use_a40',
                        action='store_true',  # This makes it a boolean flag
                        help='Use this flag to indicate the use of A40 GPU')
    parser.add_argument('--num_gpus',
                        type=int,
                        default=1,  # Default to 1 GPU if not specified
                        help='Number of GPUs to use')

    args = parser.parse_args()

    template_path = './sbatch_finetune_template.sh'
    submit_job_for_experiment(template_path,
                              args.exp_name,
                              args.script_name,
                              args.model_alias,
                              args.num_gpus,
                              args.use_a40)
