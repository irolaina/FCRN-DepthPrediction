#!/usr/bin/env bash
GPU=1
STEPS=300000
# Dataset doesn't exist on Olorin!
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape     --px valid --loss mse         --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape     --px valid --loss berhu       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape     --px valid --loss eigen       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape     --px valid --loss eigen_grads --max_steps ${STEPS} --ldecay --l2norm --data_aug  # Avoid! Does the Gradients calculation make sense for the 'valid' flag?
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape     --px all   --loss mse         --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape     --px all   --loss berhu       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape     --px all   --loss eigen       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape     --px all   --loss eigen_grads --max_steps ${STEPS} --ldecay --l2norm --data_aug

python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_depth       --px valid --loss mse         --max_steps ${STEPS}          --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_depth       --px valid --loss mse         --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_depth       --px valid --loss berhu       --max_steps ${STEPS}          --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_depth       --px valid --loss berhu       --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_depth       --px valid --loss eigen       --max_steps ${STEPS}          --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_depth       --px valid --loss eigen       --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_depth       --px valid --loss eigen_grads --max_steps ${STEPS}          --l2norm --data_aug  # Avoid! Does the Gradients calculation make sense for the 'valid' flag?
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_depth       --px valid --loss eigen_grads --max_steps ${STEPS} --ldecay --l2norm --data_aug  # Avoid! Does the Gradients calculation make sense for the 'valid' flag?

#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_depth      --px all   --loss mse         --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_depth      --px all   --loss berhu       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_depth      --px all   --loss eigen       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_depth      --px all   --loss eigen_grads --max_steps ${STEPS} --ldecay --l2norm --data_aug

#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_discrete   --px valid --loss mse         --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_discrete   --px valid --loss berhu       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_discrete   --px valid --loss eigen       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_discrete   --px valid --loss eigen_grads --max_steps ${STEPS} --ldecay --l2norm --data_aug  # Avoid! Does the Gradients calculation make sense for the 'valid' flag?
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_discrete   --px all   --loss mse         --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_discrete   --px all   --loss berhu       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_discrete   --px all   --loss eigen       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitti_discrete   --px all   --loss eigen_grads --max_steps ${STEPS} --ldecay --l2norm --data_aug

#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px valid --loss mse         --max_steps ${STEPS}          --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px valid --loss mse         --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px valid --loss berhu       --max_steps ${STEPS}          --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px valid --loss berhu       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px valid --loss eigen       --max_steps ${STEPS}          --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px valid --loss eigen       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px valid --loss eigen_grads --max_steps ${STEPS}          --l2norm --data_aug  # Avoid! Does the Gradients calculation make sense for the 'valid' flag?
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px valid --loss eigen_grads --max_steps ${STEPS} --ldecay --l2norm --data_aug  # Avoid! Does the Gradients calculation make sense for the 'valid' flag?

#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px all   --loss mse         --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px all   --loss berhu       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px all   --loss eigen       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px all   --loss eigen_grads --max_steps ${STEPS} --ldecay --l2norm --data_aug

#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth        --px valid --loss mse         --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth        --px valid --loss berhu       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth        --px valid --loss eigen       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth        --px valid --loss eigen_grads --max_steps ${STEPS} --ldecay --l2norm --data_aug  # Avoid! Does the Gradients calculation make sense for the 'valid' flag?
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth        --px all   --loss mse         --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth        --px all   --loss berhu       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth        --px all   --loss eigen       --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth        --px all   --loss eigen_grads --max_steps ${STEPS} --ldecay --l2norm --data_aug