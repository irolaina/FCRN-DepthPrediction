#!/usr/bin/env bash
GPU=0
STEPS = 150000
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s apolloscape --max_steps ${STEPS} --ldecay --l2norm -t
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kittidepth --loss mse --max_steps ${STEPS} --ldecay --l2norm
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kittidiscrete --loss mse --max_steps ${STEPS} --ldecay --l2norm
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s kitticontinuous --loss mse --max_steps ${STEPS} --ldecay --l2norm
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth --max_steps ${STEPS} --ldecay --l2norm -t