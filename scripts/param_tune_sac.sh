#!/bin/bash

# LR (default 3e-5)
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   5e-4   500_000   256   0.01   10.0 1.0  data/sac_coordring_tune1   overcooked_coordination_ring_v0   global_obs
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   1e-4   500_000   256   0.01   10.0 1.0  data/sac_coordring_tune2   overcooked_coordination_ring_v0   global_obs
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   1e-5   500_000   256   0.01   10.0 1.0  data/sac_coordring_tune3   overcooked_coordination_ring_v0   global_obs

# buffer size (default 500_000)
# sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   3e-5   1_000_000   256   0.01  10.0 1.0 data/sac_coordring_tune4   overcooked_coordination_ring_v0   global_obs
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   3e-5   250_000   256   0.01  10.0 1.0 data/sac_coordring_tune5   overcooked_coordination_ring_v0   global_obs
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   3e-5   100_000   256   0.01 10.0 1.0  data/sac_coordring_tune6   overcooked_coordination_ring_v0   global_obs

# batch size (default 128)
sbatch scripts/CC_script_sac.sh   256   3_000_000   2   0.99   3e-5   500_000   256   0.01  10.0 1.0 data/sac_coordring_tune7   overcooked_coordination_ring_v0   global_obs
sbatch scripts/CC_script_sac.sh   64   3_000_000   2   0.99   3e-5   500_000   256   0.01  10.0 1.0 data/sac_coordring_tune8   overcooked_coordination_ring_v0   global_obs

# network size (default 256)
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   3e-5   500_000   128   0.01  10.0 1.0 data/sac_coordring_tune9   overcooked_coordination_ring_v0   global_obs
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   3e-5   500_000   512   0.01  10.0 1.0 data/sac_coordring_tune10   overcooked_coordination_ring_v0   global_obs