#!/usr/bin/env bash
# It's necessary to delete the '.meta' from filepath

# Evaluate trained models in the Eigen Split using 697 depth images generated from raw LiDAR measurements (sparse)
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_continuous/all_px/berhu/2019-03-18_10-02-45/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_continuous/all_px/eigen/2019-03-19_17-39-37/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_continuous/all_px/eigen_grads/2019-03-20_21-19-23/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_continuous/all_px/mse/2019-03-16_22-39-37/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/berhu/2019-02-20_18-23-56/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/berhu/2019-02-23_13-30-01/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/berhu/2019-02-25_18-00-44/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/berhu/2019-02-27_17-47-33/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/berhu/2019-03-01_15-31-00/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/berhu/2019-03-12_22-12-27/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/eigen/2019-03-14_08-56-04/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/eigen_grads/2019-03-15_12-43-52/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/mse/2019-02-17_22-48-02/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/mse/2019-03-11_12-06-08/restore/model.fcrn

python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_depth/all_px/berhu/2019-01-22_20-47-15/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_depth/all_px/eigen/2019-01-27_03-15-30/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_depth/all_px/eigen_grads/2019-01-31_07-35-41/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_depth/all_px/mse/2019-01-18_10-57-53/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_depth/valid_px/berhu/2018-12-31_16-14-43/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_depth/valid_px/berhu/2019-01-06_16-57-24/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_depth/valid_px/eigen/2019-01-10_10-05-12/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_depth/valid_px/eigen_grads/2019-01-15_04-39-05/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_depth/valid_px/mse/2018-12-19_18-47-38/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_depth/valid_px/mse/2018-12-27_21-38-06/restore/model.fcrn

python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_discrete/all_px/berhu/2019-02-12_19-40-06/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_discrete/all_px/eigen/2019-02-14_10-16-42/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_discrete/all_px/eigen_grads/2019-02-16_01-09-21/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_discrete/all_px/mse/2019-02-11_08-03-00/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_discrete/valid_px/berhu/2019-02-06_02-53-55/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_discrete/valid_px/eigen/2019-02-07_20-34-58/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_discrete/valid_px/eigen_grads/2019-02-09_17-33-42/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen --max_depth 80.0 -r output/fcrn/kitti_discrete/valid_px/mse/2019-02-04_07-45-39/restore/model.fcrn

# Evaluate trained models in the Eigen Split based on 652 KITTI Depth annotated ground truth images (semi-dense)
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_continuous/all_px/berhu/2019-03-18_10-02-45/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_continuous/all_px/eigen/2019-03-19_17-39-37/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_continuous/all_px/eigen_grads/2019-03-20_21-19-23/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_continuous/all_px/mse/2019-03-16_22-39-37/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/berhu/2019-02-20_18-23-56/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/berhu/2019-02-23_13-30-01/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/berhu/2019-02-25_18-00-44/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/berhu/2019-02-27_17-47-33/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/berhu/2019-03-01_15-31-00/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/berhu/2019-03-12_22-12-27/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/eigen/2019-03-14_08-56-04/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/eigen_grads/2019-03-15_12-43-52/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/mse/2019-02-17_22-48-02/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_continuous/valid_px/mse/2019-03-11_12-06-08/restore/model.fcrn

python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_depth/all_px/berhu/2019-01-22_20-47-15/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_depth/all_px/eigen/2019-01-27_03-15-30/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_depth/all_px/eigen_grads/2019-01-31_07-35-41/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_depth/all_px/mse/2019-01-18_10-57-53/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_depth/valid_px/berhu/2018-12-31_16-14-43/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_depth/valid_px/berhu/2019-01-06_16-57-24/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_depth/valid_px/eigen/2019-01-10_10-05-12/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_depth/valid_px/eigen_grads/2019-01-15_04-39-05/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_depth/valid_px/mse/2018-12-19_18-47-38/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_depth/valid_px/mse/2018-12-27_21-38-06/restore/model.fcrn

python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_discrete/all_px/berhu/2019-02-12_19-40-06/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_discrete/all_px/eigen/2019-02-14_10-16-42/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_discrete/all_px/eigen_grads/2019-02-16_01-09-21/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_discrete/all_px/mse/2019-02-11_08-03-00/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_discrete/valid_px/berhu/2019-02-06_02-53-55/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_discrete/valid_px/eigen/2019-02-07_20-34-58/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_discrete/valid_px/eigen_grads/2019-02-09_17-33-42/restore/model.fcrn
python3 predict_nick.py -m test --machine ${USER} -s kitti_depth --eval_tool monodepth --test_split eigen_kitti_depth --max_depth 80.0 -r output/fcrn/kitti_discrete/valid_px/mse/2019-02-04_07-45-39/restore/model.fcrn
