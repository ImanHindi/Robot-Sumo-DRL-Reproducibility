@echo off
if "%~1"=="" exit /b 2
python scripts\asymmetric_selfplay_master.py --benchmark-root "%~1" --seeds 0 1 2 3 4 --stage3-steps 500000 --checkpoint-every 50000 --checkpoint-eval-steps 100000 200000 300000 400000 500000 --intermediate-episodes 50 --final-h2h-episodes 100 --external-episodes 100 --name asymmetric_selfplay_v114
