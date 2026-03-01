import pandas as pd

df = pd.read_csv("analysis_dataset.csv")
robot = pd.read_csv("robot_density_full.csv")

df = df.merge(robot, on=["country","year"], how="left")

df.to_csv("final_dataset.csv", index=False)

print("최종 데이터 완성")
print(df.head())