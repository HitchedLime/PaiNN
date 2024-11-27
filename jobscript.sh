#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J painn
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 20:00
### -- request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "select[sxm2]"
#BSUB -R "span[hosts=1]
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s134620@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.8

/appl/cuda/11.8.0/samples/bin/x86_64/linux/release/deviceQuery

module load python3/3.10.13

source $BLACKHOLE/group85/painn/bin/activate
python3 $BLACKHOLE/PaiNN/pyPainnMessageUpdate.py
