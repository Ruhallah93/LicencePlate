import pandas as pd

data = pd.read_csv("ImageFiltering/labeled_data/denormalized_noise_vectors.csv")
data[data['label'] == 1].to_csv("utils/noise/noise_vectors.csv")
