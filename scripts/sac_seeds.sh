#!/bin/bash

sbatch scripts/CC_script_sac.sh   128   10_000_000   1   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac_seeds/1 overcooked_cramped_room_v0   global_obs
# sbatch scripts/CC_script_sac.sh   128   10_000_000   2   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac_seeds/2 overcooked_cramped_room_v0   global_obs
sbatch scripts/CC_script_sac.sh   128   10_000_000   3   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac_seeds/3 overcooked_cramped_room_v0   global_obs
sbatch scripts/CC_script_sac.sh   128   10_000_000   4   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac_seeds/4 overcooked_cramped_room_v0   global_obs
sbatch scripts/CC_script_sac.sh   128   10_000_000   5   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac_seeds/5 overcooked_cramped_room_v0   global_obs