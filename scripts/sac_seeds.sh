#!/bin/bash
FEATURE='local_obs'
FIXED_BFS=1
if [ "$FIXED_BFS" -eq 1 ]; then
    BFS_SUFFIX="_bfs"
else
    BFS_SUFFIX=""
fi

# cramped room
sbatch scripts/CC_script_sac.sh   128   10_000_000   1   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_cramped_room_${FEATURE}/1 overcooked_cramped_room_v0   $FEATURE $FIXED_BFS
# sbatch scripts/CC_script_sac.sh   128   10_000_000   2   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_cramped_room_${FEATURE}/2 overcooked_cramped_room_v0   $FEATURE $FIXED_BFS
# sbatch scripts/CC_script_sac.sh   128   10_000_000   3   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_cramped_room_${FEATURE}/3 overcooked_cramped_room_v0   $FEATURE $FIXED_BFS
# sbatch scripts/CC_script_sac.sh   128   10_000_000   4   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_cramped_room_${FEATURE}/4 overcooked_cramped_room_v0   $FEATURE $FIXED_BFS
# sbatch scripts/CC_script_sac.sh   128   10_000_000   5   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_cramped_room_${FEATURE}/5 overcooked_cramped_room_v0   $FEATURE $FIXED_BFS

# # coord ring
# sbatch scripts/CC_script_sac.sh   128   10_000_000   1   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_coordination_ring_${FEATURE}/1 overcooked_coordination_ring_v0   $FEATURE $FIXED_BFS
# sbatch scripts/CC_script_sac.sh   128   10_000_000   2   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_coordination_ring_${FEATURE}/2 overcooked_coordination_ring_v0   $FEATURE $FIXED_BFS
# sbatch scripts/CC_script_sac.sh   128   10_000_000   3   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_coordination_ring_${FEATURE}/3 overcooked_coordination_ring_v0   $FEATURE $FIXED_BFS
# sbatch scripts/CC_script_sac.sh   128   10_000_000   4   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_coordination_ring_${FEATURE}/4 overcooked_coordination_ring_v0   $FEATURE $FIXED_BFS
# sbatch scripts/CC_script_sac.sh   128   10_000_000   5   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_coordination_ring_${FEATURE}/5 overcooked_coordination_ring_v0   $FEATURE $FIXED_BFS

# # counter circuit
# sbatch scripts/CC_script_sac.sh   128   10_000_000   1   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_counter_circuit_${FEATURE}/1 overcooked_counter_circuit_v0   $FEATURE $FIXED_BFS
# sbatch scripts/CC_script_sac.sh   128   10_000_000   2   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_counter_circuit_${FEATURE}/2 overcooked_counter_circuit_v0   $FEATURE $FIXED_BFS
# sbatch scripts/CC_script_sac.sh   128   10_000_000   3   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_counter_circuit_${FEATURE}/3 overcooked_counter_circuit_v0   $FEATURE $FIXED_BFS
# sbatch scripts/CC_script_sac.sh   128   10_000_000   4   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_counter_circuit_${FEATURE}/4 overcooked_counter_circuit_v0   $FEATURE $FIXED_BFS
# sbatch scripts/CC_script_sac.sh   128   10_000_000   5   0.99   3e-5   500_000   256   0.01 10.0 0.1  data/sac${BFS_SUFFIX}_counter_circuit_${FEATURE}/5 overcooked_counter_circuit_v0   $FEATURE $FIXED_BFS