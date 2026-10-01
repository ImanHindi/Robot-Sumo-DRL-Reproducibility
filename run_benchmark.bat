@echo off
python scripts\benchmark_algorithms.py --algorithms A2C PPO DQN RecurrentPPO --seeds 0 1 2 3 4 --stage1-steps 200000 --stage2-steps 300000 --eval-episodes 100 --name benchmark_v112
