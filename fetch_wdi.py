import glob
import pandas as pd

COUNTRIES = ["KOR","JPN","CHN","DEU","USA","FRA","ITA","ESP","NLD","CAN","MEX","IND","BRA","TUR","SGP"]

def pick_file(pattern: str) -> str:
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"파일 못 찾음: {pattern}\n"
            f"현재 폴더 csv 목록: {glob.glob('*.csv')}"
        )
    matches.sort(key=len, reverse=True)
    return matches[0]

def clean_wdi(file_path: str, value_name: str) -> pd.DataFrame:
    # World Bank WDI CSV는 앞 4줄이 메타(쓰레기 라인)인 경우가 많음
    df = pd.read_csv(file_path, skiprows=4)

    # 필수 컬럼 검증
    required = {"Country Code"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"'Country Code' 컬럼이 없습니다. "
            f"skiprows=4가 맞는지 확인 필요. 현재 컬럼: {list(df.columns)[:10]}"
        )

    df = df[df["Country Code"].isin(COUNTRIES)]

    year_cols = [c for c in df.columns if str(c).isdigit()]
    if not year_cols:
        raise ValueError(f"연도 컬럼(2000, 2001...)을 못 찾음. 현재 컬럼 일부: {list(df.columns)[:20]}")

    df = df[["Country Code"] + year_cols]

    df = df.melt(
        id_vars="Country Code",
        value_vars=year_cols,
        var_name="year",
        value_name=value_name
    )

    df.rename(columns={"Country Code": "country"}, inplace=True)
    df["year"] = df["year"].astype(int)

    return df

# ✅ 폴더 안에서 자동으로 파일 찾기
industry_file = pick_file("API_SL.IND.EMPL.ZS*.csv")
service_file  = pick_file("API_SL.SRV.EMPL.ZS*.csv")
gdp_file      = pick_file("API_NY.GDP.PCAP*.csv")  # KD든 CD든 패턴으로 잡음

print("사용 파일:")
print(" -", industry_file)
print(" -", service_file)
print(" -", gdp_file)

industry = clean_wdi(industry_file, "industry")
service  = clean_wdi(service_file, "service")
gdp      = clean_wdi(gdp_file, "gdp")

# merge
df = industry.merge(service, on=["country", "year"], how="inner")
df = df.merge(gdp, on=["country", "year"], how="inner")

df = df.sort_values(["country", "year"])

df.to_csv("panel_dataset.csv", index=False, encoding="utf-8-sig")

print("✅ 완료: panel_dataset.csv 생성")
print("국가 수:", df["country"].nunique(), "| 행 수:", len(df))
print(df.head())