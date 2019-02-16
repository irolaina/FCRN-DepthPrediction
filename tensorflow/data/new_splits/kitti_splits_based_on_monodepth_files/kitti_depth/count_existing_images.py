import os
import pandas as pd

# Select
column = {'raw_data': 0, 'depth': 1}
selected_column = column['raw_data']

# Input
eigen_test = pd.read_csv('eigen_test_files.txt', delimiter='\t', header=None)
# eigen_test = pd.read_csv('eigen_train_files.txt', delimiter='\t', header=None)

# Preprocessing
eigen_test[selected_column] = eigen_test[selected_column].apply(lambda x: '/media/nicolas/nicolas_seagate/datasets/kitti/' + x)
eigen_test['exists'] = eigen_test[selected_column].apply(lambda x: os.path.exists(x))

# Results
print(eigen_test[[selected_column, 'exists']])
print()

print("Missing Files: {}".format(len(eigen_test.index) - sum(eigen_test['exists'])) )
print("Done.")