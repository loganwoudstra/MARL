#!/bin/bash

# LR (default 3e-5)
sbatch scripts/CC_script_sac.sh   256   10_000_000   2   0.99   5e-4   500_000   256   0.01   data/sac_tune1   overcooked_cramped_room_v0   global_obs
sbatch scripts/CC_script_sac.sh   256   10_000_000   2   0.99   1e-4   500_000   256   0.01   data/sac_tune2   overcooked_cramped_room_v0   global_obs
sbatch scripts/CC_script_sac.sh   256   10_000_000   2   0.99   1e-5   500_000   256   0.01   data/sac_tune3   overcooked_cramped_room_v0   global_obs

# buffer size (default 500_000)
# sbatch scripts/CC_script_sac.sh   256   10_000_000   2   0.99   3e-5   1_000_000   256   0.01   data/sac_tune4   overcooked_cramped_room_v0   global_obs
sbatch scripts/CC_script_sac.sh   256   10_000_000   2   0.99   3e-5   250_000   256   0.01   data/sac_tune5   overcooked_cramped_room_v0   global_obs
sbatch scripts/CC_script_sac.sh   256   10_000_000   2   0.99   3e-5   100_000   256   0.01   data/sac_tune6   overcooked_cramped_room_v0   global_obs

# batch size (default 256)
sbatch scripts/CC_script_sac.sh   128   10_000_000   2   0.99   3e-5   500_000   256   0.01   data/sac_tune7   overcooked_cramped_room_v0   global_obs
sbatch scripts/CC_script_sac.sh   64   10_000_000   2   0.99   3e-5   500_000   256   0.01   data/sac_tune8   overcooked_cramped_room_v0   global_obs

# network size (default 256)
sbatch scripts/CC_script_sac.sh   256   10_000_000   2   0.99   3e-5   500_000   64   0.01   data/sac_tune9   overcooked_cramped_room_v0   global_obs
sbatch scripts/CC_script_sac.sh   256   10_000_000   2   0.99   3e-5   500_000   128   0.01   data/sac_tune10   overcooked_cramped_room_v0   global_obs