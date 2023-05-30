#!/bin/bash  

#SBATCH --job-name="RAFT"  
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH -o runs/attack_log.o%j
#SBATCH --mem-per-gpu=7G


source /home/scheurek/Software/torch/bin/activate
export CUDA_VISIBLE_DEVICES=0
python train.py --flow --flow_weights_path model/model_flow.pth --classes_path data/ucf101_i3d_raft_sarinagara/classes.txt --data_path data/ucf101_i3d_raft_sarinagara --save_folder sari --num_freeze 15
# python train.py --flow --flow_weights_path model/sari/flow_best_model_2023-05-17_08:41:31.pth --classes_path data/ucf101_i3d_raft_sarinagara/classes.txt --data_path data/ucf101_i3d_raft_sarinagara --save_folder sari --num_freeze 0

