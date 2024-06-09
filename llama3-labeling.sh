#!/bin/bash
#SBATCH --job-name=llama3-labeling
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=15G
#SBATCH --time=05:00:00
#SBATCH --output=changemyview_pei/llama3-labeling/llama3-labeling-%j.out
#SBATCH --mail-type=ALL

# start the server
./changemyview_pei/llama3-labeling/start_server.sh &

# run the python script
srun python changemyview_pei/llama3-labeling/assistant.py

# stop the server
./changemyview_pei/llama3-labeling/stop_server.sh