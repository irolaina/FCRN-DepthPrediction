#!/usr/bin/env bash
python3 predict_nick.py -m train -s kitticontinuous_residential --max_steps 100000 --ldecay --gpu 0
python3 predict_nick.py -m train -s kitticontinuous_residential --max_steps 200000 --ldecay --gpu 0
python3 predict_nick.py -m train -s kitticontinuous_residential --max_steps 300000 --ldecay --gpu 0
python3 predict_nick.py -m train -s kitticontinuous_residential --max_steps 400000 --ldecay --gpu 0
python3 predict_nick.py -m train -s kitticontinuous_residential --max_steps 500000 --ldecay --gpu 0
