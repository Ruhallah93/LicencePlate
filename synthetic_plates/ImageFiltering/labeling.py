from utils.Labeling import Labeling
import pandas as pd
import os

no_label_data_file = 'no_labels/normalized_noise_vectors.csv'
evaluation = Labeling(train_data_file='train_data/single_label/normalized_noise_vectors.csv',
                      no_label_data_file=no_label_data_file)

for method in ['mlp']:
    print("Method:", method)
    X_train, y_prediction = evaluation.run(method=method)
    data = pd.read_csv(no_label_data_file)
    data['label'] = y_prediction
    if not os.path.exists(f"labeled_data/{method}/"):
        os.makedirs(f"labeled_data/{method}/")
    data.to_csv(f"labeled_data/{method}/normalized_noise_vectors.csv")
