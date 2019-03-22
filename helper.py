import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def load_dataset_light():
	df = pd.read_csv("tf_course/data.csv")
	on_indices = df[df.state == 1].index
	off_indices = df[df.state == 0].index
	sampled_off_indices = np.random.choice(off_indices, len(on_indices), replace=False)
	sampled_indices = np.concatenate([on_indices, sampled_off_indices])
	np.random.shuffle(sampled_indices)
	df_sampled = df.loc[sampled_indices]
	cols = [str(c) for c in df_sampled.columns]
	feat_names = [c for c in cols if c not in ["state", "mean_onoff_state"]]
	feats = df_sampled[feat_names]
	label = df_sampled["state"]
	feats_train, feats_test, label_train, label_test = train_test_split(feats, label, test_size=0.2, shuffle=False)
	return feats_train, feats_test, label_train, label_test
