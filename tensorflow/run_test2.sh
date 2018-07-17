#!/usr/bin/env bash
GPU=1
STEPS = 150000
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth --loss mse --max_steps ${STEPS} --ldecay --l2norm
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth --loss berhu --max_steps ${STEPS} --ldecay --l2norm
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth --loss mse --max_steps ${STEPS} --ldecay --l2norm
python3 predict_nick.py -m train --machine ${USER} --gpu ${GPU} -s nyudepth --loss berhu --max_steps ${STEPS} --ldecay --l2norm
