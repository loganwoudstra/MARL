#!bin/bash
FEATURE='local_obs'
FIXED_BFS=1
if [ "$FIXED_BFS" -eq 1 ]; then
    BFS_SUFFIX="_bfs"
else
    BFS_SUFFIX=""
fi

#crmaped room
sbatch scripts/CC_script.sh   16   128   4   20_000_000   1   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_cramped_room_${FEATURE}/1  overcooked_cramped_room_v0   $FEATURE $FIXED_BFS
sbatch scripts/CC_script.sh   16   128   4   20_000_000   2   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_cramped_room_${FEATURE}/2  overcooked_cramped_room_v0   $FEATURE $FIXED_BFS
sbatch scripts/CC_script.sh   16   128   4   20_000_000   3   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_cramped_room_${FEATURE}/3  overcooked_cramped_room_v0   $FEATURE $FIXED_BFS
sbatch scripts/CC_script.sh   16   128   4   20_000_000   4   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_cramped_room_${FEATURE}/4  overcooked_cramped_room_v0   $FEATURE $FIXED_BFS
sbatch scripts/CC_script.sh   16   128   4   20_000_000   5   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_cramped_room_${FEATURE}/5  overcooked_cramped_room_v0   $FEATURE $FIXED_BFS

# coordiantion ring
sbatch scripts/CC_script.sh   16   128   4   20_000_000   1   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_coordination_ring_${FEATURE}/1  overcooked_coordination_ring_v0   $FEATURE $FIXED_BFS
sbatch scripts/CC_script.sh   16   128   4   20_000_000   2   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_coordination_ring_${FEATURE}/2  overcooked_coordination_ring_v0   $FEATURE $FIXED_BFS
sbatch scripts/CC_script.sh   16   128   4   20_000_000   3   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_coordination_ring_${FEATURE}/3  overcooked_coordination_ring_v0   $FEATURE $FIXED_BFS
sbatch scripts/CC_script.sh   16   128   4   20_000_000   4   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_coordination_ring_${FEATURE}/4  overcooked_coordination_ring_v0   $FEATURE $FIXED_BFS
sbatch scripts/CC_script.sh   16   128   4   20_000_000   5   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_coordination_ring_${FEATURE}/5  overcooked_coordination_ring_v0   $FEATURE $FIXED_BFS

#coutner circuit
sbatch scripts/CC_script.sh   16   128   4   20_000_000   1   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_counter_circuit_${FEATURE}/1  overcooked_counter_circuit_v0   $FEATURE $FIXED_BFS
sbatch scripts/CC_script.sh   16   128   4   20_000_000   2   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_counter_circuit_${FEATURE}/2  overcooked_counter_circuit_v0   $FEATURE $FIXED_BFS
sbatch scripts/CC_script.sh   16   128   4   20_000_000   3   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_counter_circuit_${FEATURE}/3  overcooked_counter_circuit_v0   $FEATURE $FIXED_BFS
sbatch scripts/CC_script.sh   16   128   4   20_000_000   4   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_counter_circuit_${FEATURE}/4  overcooked_counter_circuit_v0   $FEATURE $FIXED_BFS
sbatch scripts/CC_script.sh   16   128   4   20_000_000   5   5   0.2   0.5   0.01   0.99   0.95   3e-4   data/ppo${BFS_SUFFIX}_counter_circuit_${FEATURE}/5  overcooked_counter_circuit_v0   $FEATURE $FIXED_BFS