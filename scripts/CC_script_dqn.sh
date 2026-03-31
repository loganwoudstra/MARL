#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=12G
#SBATCH --time=14:00:00
#SBATCH --account=aip-mtaylor3
#SBATCH --output=/home/lwoudstr/scratch/slurm_out/%A.out

export results=$SLURM_TMPDIR/results
export data=$SLURM_TMPDIR/data
PROJECT=/project/aip-mtaylor3/lwoudstr

module load python/3.10
module load cuda
module load gcc opencv/4.9.0
source /home/lwoudstr/projects/aip-mtaylor3/lwoudstr/MARL/.venv/bin/activate 

echo "batch_size: $1"
echo "total_steps: $2"
echo "seed: $3"
echo "gamma: $4"
echo "lr: $5"
echo "buffer_size: $6"
echo "hidden_dim: $7"
echo "tau: $8"
echo "grad_norm_clip: $9"
echo "anneal_scale: ${10}"
echo "data_path: ${11}"
echo "layout: ${12}"
echo "feature: ${13}"
echo "fixed_bfs: ${14}"

# Check if $14 is 1 (enable fixed-bfs)
if [ "${14}" -eq 1 ]; then
    FIXED_BFS_FLAG="--fixed-bfs"
    NUM_AGENTS=2
else
    FIXED_BFS_FLAG=""
    NUM_AGENTS=1
fi

python3 main.py --algorithm dqn --save-path models --save --num-agents $NUM_AGENTS --num-envs 1 --num-steps 1 \
--batch-size-sac $1 --total-steps $2 --seed $3 --log --gamma $4 --lr $5 --buffer-size $6 --hidden-dim $7 --tau $8 \
--grad-norm-clip $9 --anneal-scale ${10} \
--data-path ${11} --layout ${12} --feature ${13} $FIXED_BFS_FLAG

mkdir -p $PROJECT/MARL/logs
cp -r $SLURM_TMPDIR/logs $PROJECT/MARL/logs/$SLURM_JOB_ID
