import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    df = data.frame
    df['target'] = data.target
    return df

def preprocess(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
