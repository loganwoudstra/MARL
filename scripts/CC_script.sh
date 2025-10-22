#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --account=aip-mtaylor3
#SBATCH --output=/home/truonggi/scratch/slurm_out/%A.out
#SBATCH --mail-user=truonggi@ualberta.ca
#SBATCH --mail-type=ALL


export results=$SLURM_TMPDIR/results
export data=$SLURM_TMPDIR/data

module load python/3.10
module load cuda
module load gcc opencv/4.9.0
source /home/truonggi/projects/aip-mtaylor3/truonggi/MARL/env/bin/activate 

echo $1 # num_envs
echo $2 # num_steps
echo $3 # num_minibatches
echo $4 # total_steps
echo $5 # seed
echo $6 # ppo_epoch
echo $7 # clip_param
echo $8 # value_loss_coef
echo $9 # entropy_coef
echo ${10} # gamma
echo ${11} # lam
echo ${12} # lr

echo ${13} # data_path
echo ${14} # layout
echo ${15} # feature

python3 ../main.py --save-path models --num-agents 2 --num-envs $1 --num-steps $2 --num-minibatches $3 \
--total-steps $4 --seed $5  --ppo-epoch $6 --clip-param $7 \
--value-loss-coef $8 --entropy-coef $9 --gamma ${10} --lam ${11} --max-grad-norm 0.5 --lr ${12} --data-path ${13} --layout ${14} \
--feature ${15}

python3 ../main.py --save-path models --num-agents 2 --num-envs $1 --num-steps $2 --num-minibatches $3 \
--total-steps $4 --seed $5  --ppo-epoch $6 --clip-param $7 \
--value-loss-coef $8 --entropy-coef $9 --gamma ${10} --lam ${11} --max-grad-norm 0.5 --lr ${12} --data-path ${13} --layout ${14} \
--feature ${15} --algorithm open_loop_mappo

#python3 main.py --save-path models --num-agents 2 --num-envs 16 --layout overcooked_cramped_room_v0  --num-steps 256 --num-minibatches 4 \
#--total-steps 20000000 --seed 3 --log --ppo-epoch 5 --clip-param 0.05 \
#--value-loss-coef 0.1 --entropy-coef 0.01 --gamma 0.99 --lam 0.95 --max-grad-norm 0.5 --lr 3e-4 --data-path data \
#--feature global_obs --algorithm open_loop_mappo