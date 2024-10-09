#!/bin/bash
#
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# TODO:
#   - Set name of the job below changing "Test" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch examle_submit.sh

# -= Resources =-

#SBATCH --job-name=<tmp>
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gsoykan20@ku.edu.tr
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=5
#SBATCH --partition ai
#SBATCH --account=ai
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ai26,ai18
# #SBATCH --constraint=tesla_t4
#SBATCH --constraint=rtx_a6000
#SBATCH --time=7-0
#SBATCH --mem=96G
#SBATCH --output=<tmp>.log

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

## Module Load
echo "Loading Anaconda Module..."
module load anaconda/2022.10
module load cuda/12.3

echo ""
echo "======================================================================================"
env

current_user=$(whoami)

if [ "$current_user" = "gsoykan20" ]; then
    export PYTHONPATH=$PYTHONPATH:/scratch/users/gsoykan20/projects/multilingual-adapters/
elif [ "$current_user" = "gosahin" ]; then
    export PYTHONPATH=$PYTHONPATH:/scratch/users/gosahin/hpc_run/multilingual-adapters/
else
    echo "Unknown user: $current_user"
    exit 1
fi

echo "======================================================================================"
echo ""
source activate lit-gpt

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Running Python Job (Multilingual Adapters)!"
echo "==============================================================================="
# Command  for running
# Put Python script command below

if [ "$current_user" = "gsoykan20" ]; then
    cd /scratch/users/gsoykan20/projects/multilingual-adapters/src/instruction_tuning || exit
elif [ "$current_user" = "gosahin" ]; then
    cd /scratch/users/gosahin/hpc_run/multilingual-adapters/src/instruction_tuning || exit
else
    echo "Unknown user: $current_user"
    exit 1
fi

sh ./scripts/<tmp>.sh
RET=$?
echo ""
echo "RTC exited with return code: $RET"
exit $RET