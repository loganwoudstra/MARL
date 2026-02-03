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

echo "batch_size: $1"
echo "total_steps: $2"
echo "seed: $3"
echo "gamma: $4"
echo "lr: $5"
echo "buffer_size: $6"
echo "hidden_dim: $7"
echo "tau: $8"
echo "data_path: $9"
echo "layout: ${10}"
echo "feature: ${11}"

python3 main.py --algorithm sac --save-path models --num-agents 1 --num-envs 1 --num-steps 1 \
--batch-size-sac $1 --total-steps $2 --seed $3 --log --gamma $4 --lr $5 --buffer-size $6 --hidden-dim $7 --tau $8 \
--data-path $9 --layout ${10} --feature ${11}
