Run Single Prediction: 

    python predict.py ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt ../misc/nyu_example.png --gpu 1
    
Train:

    python3 predict_nick.py -m train -s kittiraw_residential_continuous --max_steps 10 -d 0.5 --ldecay --gpu 0 -t

Test:

    python3 predict_nick.py -m test -s kittiraw_residential_continuous -r output/fcrn/2018-02-26_17-08-45/restore/model.fcrn --gpu 1 -u

Predict:

    python3 predict_nick.py -m pred -r ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt -i ../misc/nyu_example.png --gpu 1

Predict OpenCV:

    python3 predict_cv.py ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt ../misc/drone_indoor.avi
    python3 predict_cv.py ../models/NYU_FCRN-checkpoint/NYU_FCRN.ckpt ../misc/drone_indoor.avi --gpu 1 2> error.txt

Encode Video:

    ffmpeg -r 30 -f image2 -s 304x288 -i frame%06d.png -i pred%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ../test.mp4