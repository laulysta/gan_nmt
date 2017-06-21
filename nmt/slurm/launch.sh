#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=0_python--u-create_data_beam.py--dw-620--d-1000--data-beam_europarl_en_de--r-_1000_0.001_64_5_18_432054.npz
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p1080:1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --error=/scratch/sl174/beam_nmt/beam_nmt/LOGS_QSUB/2017-06-12T16:57:34.120756__python_-u_create_data_beam.py_-dw_620_-d_1000_-data_beam_europarl_en_de_-r__1000_0.001_64_5_18_432054.npz/slurm_%j.err
#SBATCH --output=/scratch/sl174/beam_nmt/beam_nmt/LOGS_QSUB/2017-06-12T16:57:34.120756__python_-u_create_data_beam.py_-dw_620_-d_1000_-data_beam_europarl_en_de_-r__1000_0.001_64_5_18_432054.npz/slurm_%j.out



module purge
module load theano/0.9.0
module load nltk/3.2.2
module load scikit-learn/intel/0.18.1
module load cuda/8.0.44
module load cudnn/8.0v5.1



SRC=/scratch/sl174/beam_nmt/beam_nmt



cd $SRC ; python -u create_data_beam.py -dw 620 -d 1000 -data beam_europarl_en_de -r best_model/model_gru_beam_europarl_ende_baseline_620_1000_0.001_64_5_18_432054.npz > /scratch/sl174/beam_nmt/beam_nmt/LOGS_QSUB/2017-06-12T16:57:34.120756__python_-u_create_data_beam.py_-dw_620_-d_1000_-data_beam_europarl_en_de_-r__1000_0.001_64_5_18_432054.npz/0_python--u-create_data_beam.py--dw-620--d-1000--data-beam_europarl_en_de--r-_1000_0.001_64_5_18_432054.npz 2>&1