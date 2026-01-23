#!/bin/bash

#SBATCH --job-name="culture_agent_eval"
#SBATCH --account="p_lv_hpgpu_2526"
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --output=/data/horse/ws/s4610340-culture_agent/culture_questions_agent/logs/job-%j.out

module load release/24.10 GCCcore/13.2.0 GCC/13.2.0 Python/3.11.5

cd /data/horse/ws/s4610340-culture_agent/culture_questions_agent

source /data/horse/ws/s4610340-virtualenvs/culture-questions-agent/bin/activate

python -m culture_questions_agent.evaluate task_type="saq" evaluation.max_questions=100
python -m culture_questions_agent.evaluate task_type="mcq" evaluation.max_questions=100
