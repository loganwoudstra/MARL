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
  5_000_000 \
  2 \
  0.99 \
  1e-4 \
  100_000 \
  256 \
  0.01 \
  data/sac \
  overcooked_cramped_room_v0 \
  global_obs
```
