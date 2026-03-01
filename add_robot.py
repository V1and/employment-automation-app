import pandas as pd

df = pd.read_csv("analysis_dataset.csv")
robot = pd.read_csv("robot_density.csv")

df = df.merge(robot, on=["country","year"], how="left")

df.to_csv("final_dataset.csv", index=False)

print(df.head())