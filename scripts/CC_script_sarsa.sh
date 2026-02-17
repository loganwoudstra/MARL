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

echo "layout: $1"
echo "total_steps: $2"
echo "seed: $3"
echo "lr: $4"
echo "gamma: $5"
echo "epsilon_start: $6" 
echo "epsilon_end :$7"
echo "epsilon_decay: $8" 
echo "target_update_freq: $9" 
echo "hidden_dim: ${10}" 
echo "data_path: ${11}"
echo "feature: ${12}"

python3 main.py --algorithm sarsa --save-path models --num-agents 1 --num-envs 1 --layout $1 \
--total-steps $2 --seed $3 --lr $4 --gamma $5 \
--epsilon-start $6 --epsilon-end $7 --epsilon-decay $8 --target-update-freq $9 \
--hidden-dim ${10} \
--data-path ${11} --feature ${12} --save
