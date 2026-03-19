#!bin/bash

sbatch scripts/CC_script.sh   16   128   4   20000000   1   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_cramped_room/1  overcooked_cramped_room_v0   global_obs
# sbatch scripts/CC_script.sh   16   128   4   20000000   2   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_cramped_room/2  overcooked_cramped_room_v0   global_obs
sbatch scripts/CC_script.sh   16   128   4   20000000   3   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_cramped_room/3  overcooked_cramped_room_v0   global_obs
sbatch scripts/CC_script.sh   16   128   4   20000000   4   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_cramped_room/4  overcooked_cramped_room_v0   global_obs
sbatch scripts/CC_script.sh   16   128   4   20000000   5   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_cramped_room/5  overcooked_cramped_room_v0   global_obs

sbatch scripts/CC_script.sh   16   128   4   20000000   1   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_coord_ring/1  overcooked_coordination_ring_v0   global_obs
# sbatch scripts/CC_script.sh   16   128   4   20000000   2   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_coord_ring/2  overcooked_coordination_ring_v0   global_obs
sbatch scripts/CC_script.sh   16   128   4   20000000   3   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_coord_ring/3  overcooked_coordination_ring_v0   global_obs
sbatch scripts/CC_script.sh   16   128   4   20000000   4   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_coord_ring/4  overcooked_coordination_ring_v0   global_obs
sbatch scripts/CC_script.sh   16   128   4   20000000   5   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_coord_ring/5  overcooked_coordination_ring_v0   global_obs

sbatch scripts/CC_script.sh   16   128   4   20000000   1   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_ccircuit/1  overcooked_counter_circuit_v0   global_obs
# sbatch scripts/CC_script.sh   16   128   4   20000000   2   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_ccircuit/2  overcooked_counter_circuit_v0   global_obs
sbatch scripts/CC_script.sh   16   128   4   20000000   3   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_ccircuit/3  overcooked_counter_circuit_v0   global_obs
sbatch scripts/CC_script.sh   16   128   4   20000000   4   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_ccircuit/4  overcooked_counter_circuit_v0   global_obs
sbatch scripts/CC_script.sh   16   128   4   20000000   5   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo_ccircuit/5  overcooked_counter_circuit_v0   global_obs