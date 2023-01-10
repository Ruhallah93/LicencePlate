import pandas as pd
import numpy as np
import researchpy as rp
import scipy.stats as stats

data = pd.read_csv("../labeled_data/neater/normalized_noise_vectors.csv")
features = data.columns.drop(['instance_name', 'label'])
features_base = data.columns.drop(['instance_name', 'label', 'pitch', 'yaw', 'roll'])
features_com = ['pitch', 'yaw', 'roll']

X = data[features].to_numpy()
y = data['label'].replace(['Correct', 'Corrupt'], [1, 0]).to_numpy()

rp.ttest(group1=data[features_base], group2=data[features_com])
