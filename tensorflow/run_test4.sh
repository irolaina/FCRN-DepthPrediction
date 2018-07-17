#!/usr/bin/env bash
GPU=0
STEPS=150000
# Dataset doesn't exist on Olorin!
# python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape --px valid --loss mse   --max_steps ${STEPS} --ldecay --l2norm --data_aug
# python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape --px valid --loss berhu --max_steps ${STEPS} --ldecay --l2norm --data_aug
# python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape --px valid --loss silog --max_steps ${STEPS} --ldecay --l2norm --data_aug
# python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape --px all   --loss mse   --max_steps ${STEPS} --ldecay --l2norm --data_aug
# python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape --px all   --loss berhu --max_steps ${STEPS} --ldecay --l2norm --data_aug
# python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape --px all   --loss silog --max_steps ${STEPS} --ldecay --l2norm --data_aug

python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kittidepth --px valid --loss mse   --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kittidepth --px valid --loss berhu --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kittidepth --px valid --loss silog --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kittidepth --px all   --loss mse   --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kittidepth --px all   --loss berhu --max_steps ${STEPS} --ldecay --l2norm --data_aug  # Doesn't converge!
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kittidepth --px all   --loss silog --max_steps ${STEPS} --ldecay --l2norm --data_aug

python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kittidiscrete --px valid --loss mse   --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kittidiscrete --px valid --loss berhu --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kittidiscrete --px valid --loss silog --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kittidiscrete --px all   --loss mse   --max_steps ${STEPS} --ldecay --l2norm --data_aug
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kittidiscrete --px all   --loss berhu --max_steps ${STEPS} --ldecay --l2norm --data_aug  # Doesn't converge!
#python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kittidiscrete --px all   --loss silog --max_steps ${STEPS} --ldecay --l2norm --data_aug

python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px valid --loss mse   --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px valid --loss berhu --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px valid --loss silog --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px all   --loss mse   --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px all   --loss berhu --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --px all   --loss silog --max_steps ${STEPS} --ldecay --l2norm --data_aug

python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth --px valid --loss mse   --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth --px valid --loss berhu --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth --px valid --loss silog --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth --px all   --loss mse   --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth --px all   --loss berhu --max_steps ${STEPS} --ldecay --l2norm --data_aug
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth --px all   --loss silog --max_steps ${STEPS} --ldecay --l2norm --data_aug