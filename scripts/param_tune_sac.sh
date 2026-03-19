#!/bin/bash
# overcooked_coordination_ring_v0
# overcooked_counter_circuit_v0

# LR (default 3e-5)
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   5e-4   500_000   256   0.01   10.0 1.0  data/sac_ccircuit_tune/1   overcooked_counter_circuit_v0   global_obs
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   1e-4   500_000   256   0.01   10.0 1.0  data/sac_ccircuit_tune/2   overcooked_counter_circuit_v0   global_obs
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   1e-5   500_000   256   0.01   10.0 1.0  data/sac_ccircuit_tune/3   overcooked_counter_circuit_v0   global_obs

# buffer size (default 500_000)
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   3e-5   1_000_000   256   0.01  10.0 1.0 data/sac_ccircuit_tune/4   overcooked_counter_circuit_v0   global_obs
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   3e-5   250_000   256   0.01  10.0 1.0 data/sac_ccircuit_tune/5   overcooked_counter_circuit_v0   global_obs
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   3e-5   100_000   256   0.01 10.0 1.0  data/sac_ccircuit_tune/6   overcooked_counter_circuit_v0   global_obs

# batch size (default 128)
sbatch scripts/CC_script_sac.sh   256   3_000_000   2   0.99   3e-5   500_000   256   0.01  10.0 1.0 data/sac_ccircuit_tune/7   overcooked_counter_circuit_v0   global_obs
sbatch scripts/CC_script_sac.sh   64   3_000_000   2   0.99   3e-5   500_000   256   0.01  10.0 1.0 data/sac_ccircuit_tune/8   overcooked_counter_circuit_v0   global_obs

# network size (default 256)
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   3e-5   500_000   128   0.01  10.0 1.0 data/sac_ccircuit_tune/9   overcooked_counter_circuit_v0   global_obs
sbatch scripts/CC_script_sac.sh   128   3_000_000   2   0.99   3e-5   500_000   512   0.01  10.0 1.0 data/sac_ccircuit_tune/10   overcooked_counter_circuit_v0   global_obs