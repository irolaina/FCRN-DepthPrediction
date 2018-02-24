Run Single Prediction: 

    python predict.py ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt ../misc/nyu_example.png --gpu 1
    
Train:

    python3 predict_nick.py -s kittiraw_residential_continuous --max_steps 10 -d 0.5 --ldecay -t --gpu 0
