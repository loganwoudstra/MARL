#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --account=def-mtaylor3
#SBATCH --output=/home/lwoudstr/links/scratch/slurm_out/%A.out

export results=$SLURM_TMPDIR/results
export data=$SLURM_TMPDIR/data

module load python/3.10
module load gcc opencv/4.9.0
source /home/lwoudstr/links/projects/def-mtaylor3/lwoudstr/MARL/.venv/bin/activate 

echo "num_episodes: $1"
echo "alpha: $2"
echo "lambda: $3"
echo "gamma: $4"
echo "epsilon: $5"
echo "seed: $6"
echo "data_path: $7"
echo "layout: $8"

python3 sarsalambda.py --num_episodes $1 --alpha $2 --lambda_ $3 --gamma $4 --epsilon $5 --seed $6 --data_path $7 --layout $8 
