## PPO

1. CC
```
sbatch scripts/cc_script.sh \
  16 \
  128 \
  4 \
  20000000 \
  2 \
  5 \
  0.2 \
  0.5 \
  0.01 \
  0.99 \
  0.95 \
  3e-4 \
  data \
  overcooked_cramped_room_v0 \
  global_obs
```
## SARSA
1. local
```
python main.py --save-path models --num-agents 1 --num-envs 1 --layout overcooked_cramped_room_v0 --total-steps 20000000 --seed 2 --log --gamma 0.99 --lr 3e-4 --data-path data --feature global_obs --algorithm sarsa
```

2. CC
```
sbatch scripts/CC_script_sarsa.sh \
  overcooked_cramped_room_v0 \
  5_000_000 \
  2 \
  3e-4 \
  0.99 \
  1.0 \
  0.05 \
  0.995 \
  200 \
  256 \
  data \
  global_obs
```

## SAC
1. local
```
python main.py --save-path models --num-agents 1 --num-envs 1 --layout overcooked_cramped_room_v0 --total-steps 10000000 --seed 2 --log --gamma 0.99 --lr 3e-4 --data-path data --feature global_obs --algorithm sac --batch-size-sac 128
```

2. CC
```
sbatch scripts/CC_script_sac.sh \
  128 \
  10_000_000 \
  2 \
  0.99 \
  3e-5 \
  500_000 \
  256 \
  0.01 \
  10.0 \
  0.1 \
  data/sac \
  overcooked_cramped_room_v0 \
  global_obs
```


tensorboard --logdir logs/run__1770182592 --port 6006 --load_fast=false

python -m tests.test_load --model-path .\models\dqn_bfs_2_agents_overcooked_counter_circuit_v0_seed_5\dqn_overcooked_counter_circuit_v0_seed5.pth --layout overcooked_counter_circuit_v0 --num-agents 2 --algorithm dqn --fixed-bfs

python main.py --num-agents 2 --num-envs 1 --layout overcooked_cramped_room_v0 --total-steps 10000000 --seed 2 --feature global_obs --algorithm dqn --batch-size-sac 1 --fixed-bfs --render



python main.py --save-path models --num-agents 2 --num-envs 1 --num-steps 1 --num-minibatches 1 --centralised --ppo-epoch 6 --layout overcooked_cramped_room_v0 --feature global_obs --fixed-bfs --render



python3 plot.py --folders .\data\_final\ppo_bfs_coordination_ring_global_obs\combined\ .\data\_final\dqn_bfs_coordination_ring_global_obs_long\combined\ .\data\_final\sac_bfs_coordination_ring_global_obs\combined\ .\data\_final\ppo_bfs_shared_coordination_ring_global_obs\combined\ .\data\_final\dqn_bfs_shared_coordination_ring_global_obs\combined\  --keyword returns