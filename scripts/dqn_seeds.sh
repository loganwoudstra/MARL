#!/bin/bash

# # cramped room
# sbatch scripts/CC_script_dqn.sh   128   3_000_000   1   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_cramped_room/1   overcooked_cramped_room_v0   global_obs
# sbatch scripts/CC_script_dqn.sh   128   3_000_000   2   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_cramped_room/2   overcooked_cramped_room_v0   global_obs
# sbatch scripts/CC_script_dqn.sh   128   3_000_000   3   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_cramped_room/3   overcooked_cramped_room_v0   global_obs
# sbatch scripts/CC_script_dqn.sh   128   3_000_000   4   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_cramped_room/4   overcooked_cramped_room_v0   global_obs
# sbatch scripts/CC_script_dqn.sh   128   3_000_000   5   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_cramped_room/5   overcooked_cramped_room_v0   global_obs

# # coordination ring
# sbatch scripts/CC_script_dqn.sh   128   3_000_000   1   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_coordination_ring/1   overcooked_coordination_ring_v0   global_obs
# sbatch scripts/CC_script_dqn.sh   128   3_000_000   2   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_coordination_ring/2   overcooked_coordination_ring_v0   global_obs
# sbatch scripts/CC_script_dqn.sh   128   3_000_000   3   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_coordination_ring/3   overcooked_coordination_ring_v0   global_obs
# sbatch scripts/CC_script_dqn.sh   128   3_000_000   4   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_coordination_ring/4   overcooked_coordination_ring_v0   global_obs
# sbatch scripts/CC_script_dqn.sh   128   3_000_000   5   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_coordination_ring/5   overcooked_coordination_ring_v0   global_obs



# cramped room (lfa)
sbatch scripts/CC_script_dqn.sh   128   5_000_000   1   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_cramped_room_lfa/1   overcooked_cramped_room_v0   Binary_feature
sbatch scripts/CC_script_dqn.sh   128   5_000_000   2   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_cramped_room_lfa/2   overcooked_cramped_room_v0   Binary_feature
sbatch scripts/CC_script_dqn.sh   128   5_000_000   3   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_cramped_room_lfa/3   overcooked_cramped_room_v0   Binary_feature
sbatch scripts/CC_script_dqn.sh   128   5_000_000   4   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_cramped_room_lfa/4   overcooked_cramped_room_v0   Binary_feature
sbatch scripts/CC_script_dqn.sh   128   5_000_000   5   0.99   3e-5   500_000   256   0.01   10.0 0.1  data/dqn_cramped_room_lfa/5   overcooked_cramped_room_v0   Binary_feature