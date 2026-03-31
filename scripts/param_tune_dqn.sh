#!/bin/bash
# MAP="cramped_room"
# MAP="coordination_ring"
MAP="counter_circuit"

# default
sbatch scripts/CC_script_dqn.sh   128   3_000_000   2   0.99   3e-5   500_000   256   0.01   10.0 1.0  data/dqn_${MAP}_tune/0   overcooked_${MAP}_v0   global_obs 0

# LR (default 3e-5)
sbatch scripts/CC_script_dqn.sh   128   3_000_000   2   0.99   5e-4   500_000   256   0.01   10.0 1.0  data/dqn_${MAP}_tune/1   overcooked_${MAP}_v0   global_obs 0
sbatch scripts/CC_script_dqn.sh   128   3_000_000   2   0.99   1e-4   500_000   256   0.01   10.0 1.0  data/dqn_${MAP}_tune/2   overcooked_${MAP}_v0   global_obs 0
sbatch scripts/CC_script_dqn.sh   128   3_000_000   2   0.99   1e-5   500_000   256   0.01   10.0 1.0  data/dqn_${MAP}_tune/3   overcooked_${MAP}_v0   global_obs 0

# buffer size (default 500_000)
# sbatch scripts/CC_script_dqn.sh   128   3_000_000   2   0.99   3e-5   1_000_000   256   0.01  10.0 1.0 data/dqn_${MAP}_tune/4   overcooked_${MAP}_v0   global_obs 0
sbatch scripts/CC_script_dqn.sh   128   3_000_000   2   0.99   3e-5   250_000   256   0.01  10.0 1.0 data/dqn_${MAP}_tune/5   overcooked_${MAP}_v0   global_obs 0
# sbatch scripts/CC_script_dqn.sh   128   3_000_000   2   0.99   3e-5   100_000   256   0.01 10.0 1.0  data/dqn_${MAP}_tune/6   overcooked_${MAP}_v0   global_obs 0

# batch size (default 128)
sbatch scripts/CC_script_dqn.sh   256   3_000_000   2   0.99   3e-5   500_000   256   0.01  10.0 1.0 data/dqn_${MAP}_tune/7   overcooked_${MAP}_v0   global_obs 0
sbatch scripts/CC_script_dqn.sh   64   3_000_000   2   0.99   3e-5   500_000   256   0.01  10.0 1.0 data/dqn_${MAP}_tune/8   overcooked_${MAP}_v0   global_obs 0

# network size (default 256)
sbatch scripts/CC_script_dqn.sh   128   3_000_000   2   0.99   3e-5   500_000   128   0.01  10.0 1.0 data/dqn_${MAP}_tune/9   overcooked_${MAP}_v0   global_obs 0
sbatch scripts/CC_script_dqn.sh   128   3_000_000   2   0.99   3e-5   500_000   512   0.01  10.0 1.0 data/dqn_${MAP}_tune/10   overcooked_${MAP}_v0   global_obs 0