```
sbatch run_marl.sh \
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

```
python3 main.py --save-path models --num-agents 1 --num-envs 16 --layout overcooked_cramped_room_v0 --total-steps 20000000 --seed 2 --log  --ppo-epoch 5 --gamma 0.99 --lr 3e-4 --data-path data --feature global_obs --algorithim sarsa
```