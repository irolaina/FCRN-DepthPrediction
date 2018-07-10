#!/usr/bin/env bash
GPU=0
python3 predict_nick.py -m train --machine $USER --gpu $GPU -s apolloscape --max_steps 150000 --ldecay --l2norm -t
python3 predict_nick.py -m train --machine $USER --gpu $GPU -s kittidepth --loss mse --max_steps 150000 --ldecay --l2norm
python3 predict_nick.py -m train --machine $USER --gpu $GPU -s kittidiscrete --loss mse --max_steps 150000 --ldecay --l2norm
python3 predict_nick.py -m train --machine $USER --gpu $GPU -s kitticontinuous --loss mse --max_steps 150000 --ldecay --l2norm
python3 predict_nick.py -m train --machine $USER --gpu $GPU -s nyudepth --max_steps 150000 --ldecay --l2norm -t