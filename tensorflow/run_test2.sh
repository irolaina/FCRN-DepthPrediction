#!/usr/bin/env bash
python3 predict_nick.py -m train --machine $USER --gpu 1 -s nyudepth --loss mse --max_steps 150000 --ldecay --l2norm
python3 predict_nick.py -m train --machine $USER --gpu 1 -s nyudepth --loss berhu --max_steps 150000 --ldecay --l2norm
python3 predict_nick.py -m train --machine $USER --gpu 1 -s nyudepth --loss mse --max_steps 150000 --ldecay --l2norm
python3 predict_nick.py -m train --machine $USER --gpu 1 -s nyudepth --loss berhu --max_steps 150000 --ldecay --l2norm

