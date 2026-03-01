import pandas as pd

robot = pd.read_csv("robot_density.csv")
robot["year"] = pd.to_numeric(robot["year"], errors="coerce").astype(int)
robot["robot_density"] = pd.to_numeric(robot["robot_density"], errors="coerce")

# 국가별 최소~최대 연도 범위로 연속 연도 생성
out = []
for c, g in robot.groupby("country"):
    g = g.sort_values("year")
    years = pd.DataFrame({"year": range(g["year"].min(), g["year"].max() + 1)})
    years["country"] = c
    merged = years.merge(g[["country", "year", "robot_density"]], on=["country", "year"], how="left")
    merged["robot_density"] = merged["robot_density"].interpolate(method="linear")
    out.append(merged)

robot_full = pd.concat(out, ignore_index=True).sort_values(["country", "year"])
robot_full.to_csv("robot_density_full.csv", index=False, encoding="utf-8-sig")

print("✅ 연속 연도 포함 보간 완료: robot_density_full.csv")
print(robot_full.head(30))