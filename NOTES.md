## PPO

1. CC run ppo
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
1. local run sarsa
```
python main.py --save-path models --num-agents 1 --num-envs 1 --layout overcooked_cramped_room_v0 --total-steps 20000000 --seed 2 --log --gamma 0.99 --lr 3e-4 --data-path data --feature global_obs --algorithm sarsa
```

2. CC run sarsa
```
sbatch scripts/CC_script_sarsa.sh \
  overcooked_cramped_room_v0 \
  1 \
  6000 \
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
1. local run sac
```
python main.py --save-path models --num-agents 1 --num-envs 1 --layout overcooked_cramped_room_v0 --total-steps 20000000 --seed 2 --log --gamma 0.99 --lr 3e-4 --batch-size-sac 256 --data-path data --feature global_obs --algorithm sac
```
