Run Single Prediction: 

    python predict.py ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt ../misc/nyu_example.png --gpu 1
    
Train:

    python3 predict_nick.py -m train -s kittiraw_residential_continuous --max_steps 10 -d 0.5 --ldecay -t --gpu 0

Test:

    python3 predict_nick.py -m test -s kittiraw_residential_continuous -r output/fcrn/2018-02-26_16-21-21/restore/model.fcrn -u --gpu 1

Predict:

    python3 predict_nick.py -m pred -r ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt -i ../misc/nyu_example.png --gpu 1