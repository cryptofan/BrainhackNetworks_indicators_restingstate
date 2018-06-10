#!/bin/bash
#SBATCH --job-name=jupyterTest
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem=20G
#SBATCH --time=2:00:00

#Load necessary modules
module load CUDA/9.0
module load python_gpu/3.6

cd /scratch/home/g.koehler/test_keras/BrainhackNetworks_indicators_restingstate/codes

let ipnport=($UID-6025)%65274
echo ipnport=$ipnport

ipnip=$(hostname -i)
echo ipnip=$ipnip

#Start the notebook
# manually copied to port since it got echoed with minus sign 
ipython notebook --ip=$ipnip --port=4982 --no-browser
