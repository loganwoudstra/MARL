#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --account=aip-mtaylor3
#SBATCH --output=/home/lwoudstr/scratch/slurm_out/%A.out


export results=$SLURM_TMPDIR/results
export data=$SLURM_TMPDIR/data
PROJECT=/project/aip-mtaylor3/lwoudstr

module load python/3.10
module load cuda
module load gcc opencv/4.9.0
source /home/lwoudstr/projects/aip-mtaylor3/lwoudstr/MARL/.venv/bin/activate 

echo "num envs $1"
echo "num_steps $2"
echo "num_minibatches $3"
echo "total_steps $4"
echo "seed $5" 
echo "ppo_epoch $6"
echo "clip_param $7"
echo "value_loss_coef $8"
echo "entropy_coef $9"
echo "gamma ${10}"
echo "lam ${11}"
echo "lr ${12}"

echo "data_path ${13}"
echo "layout ${14}"
echo "feature ${15}"
echo "fixed_bfs: ${16}"

# Check if $14 is 1 (enable fixed-bfs)
if [ "${16}" -eq 1 ]; then
    FIXED_BFS_FLAG="--fixed-bfs"
    NUM_AGENTS=2
else
    FIXED_BFS_FLAG=""
    NUM_AGENTS=1
fi

python3 main.py --save-path models --num-agents $NUM_AGENTS --num-envs $1 --num-steps $2 --num-minibatches $3 \
--total-steps $4 --seed $5 --log --centralised --ppo-epoch $6 --clip-param $7 \
--value-loss-coef $8 --entropy-coef $9 --gamma ${10} --lam ${11} --max-grad-norm 0.5 --lr ${12} --data-path ${13} --layout ${14} \
--feature ${15} $FIXED_BFS_FLAG

mkdir -p $PROJECT/MARL/logs
cp -r $SLURM_TMPDIR/logs $PROJECT/MARL/logs/$SLURM_JOB_ID