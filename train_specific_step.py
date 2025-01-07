import subprocess


#ADD --shuffle IF NEEDED







# List of commands to execute
commands = [
"python train.py --model-config model_configs/config.yaml --data-path data/Loaded_MP_bandgap-sphere-images_standardized --dest-path exps/train/single_sample/t_step_1 --pin-memory --check-labels --batch-size 1  --train-specific-step 1",
"python train.py --model-config model_configs/config.yaml --data-path data/Loaded_MP_bandgap-sphere-images_standardized --dest-path exps/train/single_sample/t_step_20 --pin-memory --check-labels --batch-size 1 --train-specific-step 20",
"python train.py --model-config model_configs/config.yaml --data-path data/Loaded_MP_bandgap-sphere-images_standardized --dest-path exps/train/single_sample/t_step_100 --pin-memory --check-labels --batch-size 1 --train-specific-step 100",
"python train.py --model-config model_configs/config.yaml --data-path data/Loaded_MP_bandgap-sphere-images_standardized --dest-path exps/train/single_sample/t_step_500 --pin-memory --check-labels --batch-size 1 --train-specific-step 500",
"python train.py --model-config model_configs/config.yaml --data-path data/Loaded_MP_bandgap-sphere-images_standardized --dest-path exps/train/single_sample/t_step_900 --pin-memory --check-labels --batch-size 1 --train-specific-step 900",
"python train.py --model-config model_configs/config.yaml --data-path data/Loaded_MP_bandgap-sphere-images_standardized --dest-path exps/train/single_sample/t_step_980 --pin-memory --check-labels --batch-size 1 --train-specific-step 980",
"python train.py --model-config model_configs/config.yaml --data-path data/Loaded_MP_bandgap-sphere-images_standardized --dest-path exps/train/single_sample/t_step_999 --pin-memory --check-labels --batch-size 1 --train-specific-step 999"

]

import time 
start = time.time()
# Execute each command
for command in commands:
    print(f"Executing: {command}")
    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}: {command}")
        break

print("Total time:", time.time() - start)