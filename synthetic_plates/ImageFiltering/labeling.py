from utils.Labeling import Labeling
import pandas as pd

no_label_data_file = 'no_labels/normalized_noise_vectors.csv'
evaluation = Labeling(train_data_file='train_data/normalized_noise_vectors.csv',
                      no_label_data_file=no_label_data_file)

for method in ['neater']:
    print("Method:", method)
    X_train, y_prediction = evaluation.run(method=method)
    data = pd.read_csv(no_label_data_file)
    data['label'] = y_prediction
    data.to_csv("labeled_data/normalized_noise_vectors.csv")
