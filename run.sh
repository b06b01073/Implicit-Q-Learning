#!/usr/bin/env bash
filename="result.txt"

if [ -f "$filename" ]; then
    echo "File '$filename' already exists. Recreating..."
    rm "$filename"
fi

touch "$filename"
echo "File '$filename' created successfully."

tasks=("walker2d-medium-expert-v2" "hopper-medium-expert-v2" "halfcheetah-medium-expert-v2" "walker2d-medium-replay-v2" "hopper-medium-replay-v2" "halfcheetah-medium-replay-v2" "walker2d-medium-v2" "hopper-medium-v2" "halfcheetah-medium-v2")

for env_name in ${tasks[@]}; do
    python3 main.py --env_name=$env_name --write_file=$filename 
done