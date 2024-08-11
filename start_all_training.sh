#!/bin/bash

# Define the list of intrinsic rewards
intrinsic_rewards=("extrinsic" "pseudocounts" "icm" "rnd" "ngu" "ride" "re3" "e3b" "disagreement")

# Loop over seeds (adjust the range as needed)
for seed in {1..3}; do
    # Loop over each intrinsic reward
    for reward in "${intrinsic_rewards[@]}"; do
        echo "Running with seed=${seed}, intrinsic_reward=${reward}, and without obs_rms"
        python src/train_ppo.py --env_id=MiniGrid-FourRooms-v0 --intrinsic_reward=${reward} --seed=${seed}
    done
done

# Loop over seeds (adjust the range as needed)
for seed in {1..3}; do
    # Loop over each intrinsic reward
    for reward in "${intrinsic_rewards[@]}"; do
        # Run without --obs_rms
        echo "Running with seed=${seed}, intrinsic_reward=${reward}, and without obs_rms"
        python src/train_ppo.py --env_id=MiniGrid-MultiRoom-N2-S4-v0 --intrinsic_reward=${reward} --seed=${seed}
    done
done

# Loop over seeds (adjust the range as needed)
for seed in {1..3}; do
    # Loop over each intrinsic reward
    for reward in "${intrinsic_rewards[@]}"; do
        # Run without --obs_rms
        echo "Running with seed=${seed}, intrinsic_reward=${reward}, and without obs_rms"
        python src/train_ppo.py --env_id=MiniGrid-DoorKey-5x5-v0 --intrinsic_reward=${reward} --seed=${seed}
    done
done

# Loop over seeds (adjust the range as needed)
for seed in {1..3}; do
    # Loop over each intrinsic reward
    for reward in "${intrinsic_rewards[@]}"; do
        # Run without --obs_rms
        echo "Running with seed=${seed}, intrinsic_reward=${reward}, and without obs_rms"
        python src/train_ppo.py --env_id=MiniGrid-DoorKey-6x6-v0 --intrinsic_reward=${reward} --seed=${seed}
    done
done