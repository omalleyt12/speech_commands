import pandas as pd

df = pd.read_csv("my_guesses.csv")

print(df.count())

df["label"] = ["unknown" for i in range(df.shape[0])]

print(df["label"].unique())

df.to_csv("my_guesses_3.csv",index=False)

df2 = pd.read_csv("my_guesses_3.csv")
print(df2.head())
