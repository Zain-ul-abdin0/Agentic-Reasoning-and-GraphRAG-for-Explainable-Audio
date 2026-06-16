import pandas as pd

df = pd.read_csv(
    "data/features/features.csv"
)

for col in [
    "energy",
    "pitch_std",
    "pause_ratio",
    "jitter"
]:

    print("\n", col)

    print(
        df[col].describe()
    )