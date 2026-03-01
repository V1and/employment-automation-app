import pandas as pd

df = pd.read_csv("panel_dataset.csv")

# 연구 기간 설정
df = df[df["year"] >= 1995]

# 핵심 변수 결측 제거
df = df.dropna(subset=["industry","service","gdp"])

df.to_csv("analysis_dataset.csv", index=False)

print("분석용 데이터 완성")
print(df.head())
print("행 수:", len(df))
print("국가 수:", df["country"].nunique())