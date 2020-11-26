#!/bin/sh

## uncomment for slurm
##SBATCH -p gpu
##SBATCH --gres=gpu:1
##SBATCH -c 10

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
conda activate pt140  # pytorch 1.4.0 env
PYTHON=python

dataset=$1
exp_name=$2
upper=$3
exp_dir=exp/${dataset}/${exp_name}
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${result_dir}
cp tool/test.sh tool/test.py ${config} ${exp_dir}

export PYTHONPATH=./

for i in $(seq ${upper} $END)
do
  ep=$((i*10))
  now=$(date +"%Y%m%d_%H%M%S")
  $PYTHON -u ${exp_dir}/test.py \
  --config=${config} \
  TEST.model_path ${exp_dir}/model/train_epoch_${ep}.pth \
  TEST.save_folder ${result_dir}/epoch_${ep}/val/ss \
  2>&1 | tee ${result_dir}/test-$now-$ep.log
done
