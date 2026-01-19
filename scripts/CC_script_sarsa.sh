#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=def-mtaylor3
#SBATCH --output=/home/lwoudstr/links/scratch/slurm_out/%A.out


export results=$SLURM_TMPDIR/results
export data=$SLURM_TMPDIR/data

module load python/3.10
module load cuda
module load gcc opencv/4.9.0
source /home/lwoudstr/links/projects/def-mtaylor3/lwoudstr/MARL/.venv/bin/activate 

echo $1  # layout
echo $2  # num_agents
echo $3  # num_episodes
echo $4  # seed
echo $5  # lr
echo $6  # gamma
echo $7  # epsilon_start
echo $8  # epsilon_end
echo $9  # epsilon_decay
echo ${10} # target_update_freq
echo ${11} # hidden_dim
echo ${12} # data_path
echo ${13} # feature

python3 main.py --algorithm sarsa --save-path models --num-agents $2 --num-envs 1 --layout $1 \
--num-episodes $3 --seed $4 --lr $5 --gamma $6 \
--epsilon-start $7 --epsilon-end $8 --epsilon-decay $9 --target-update-freq ${10} \
--hidden-dim ${11} \
--data-path ${12} --feature ${13} --save
