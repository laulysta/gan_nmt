#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=0_python--baseline--dropout0-5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p1080:1
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --nodes=1
#SBATCH --error=/home/fnd212/repos/nmt/err/slurm_%j.out
#SBATCH --output=/home/fnd212/repos/nmt/out/slurm_%j.out

module purge
module load theano/0.9.0
module load nltk/3.2.2
module load scikit-learn/intel/0.18.1
module load cuda/8.0.44
module load cudnn/8.0v5.1
module load pytables/intel/2.4.0
module load pytables/intel/3.4.2


SRC=/home/fnd212/repos/nmt/nmt
DST=/home/fnd212/repos/nmt/nmt/saved_models/de-en/baseline_dropout0-5/

cd $SRC ; THENAO_FLAGS=floatX=float32,device=gpu \
		  python -u nmt.py > $DST/log.log

#1156748