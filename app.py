import os
import warnings

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
import streamlit as st
from scipy import stats

st.set_page_config(page_title="자동화와 고용 구조 변화 분석", page_icon="📊", layout="wide")
warnings.filterwarnings("ignore")

COUNTRIES = ["KOR", "JPN", "CHN", "DEU", "USA", "FRA", "ITA", "ESP", "NLD", "CAN", "MEX", "IND", "BRA", "TUR", "SGP"]
COUNTRY_NAME = {
    "KOR": "대한민국", "JPN": "일본", "CHN": "중국", "DEU": "독일", "USA": "미국",
    "FRA": "프랑스", "ITA": "이탈리아", "ESP": "스페인", "NLD": "네덜란드", "CAN": "캐나다",
    "MEX": "멕시코", "IND": "인도", "BRA": "브라질", "TUR": "튀르키예", "SGP": "싱가포르"
}
START_YEAR = 2010
END_YEAR = 2023
COUNTRY_PARAM = ";".join(COUNTRIES)


def setup_font():
    for path in [
        os.path.join(os.getcwd(), "fonts", "NotoSansKR-Regular.otf"),
        os.path.join(os.getcwd(), "fonts", "NotoSansKR-Regular.ttf"),
    ]:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                name = fm.FontProperties(fname=path).get_name()
                mpl.rcParams["font.family"] = name
                mpl.rcParams["font.sans-serif"] = [name]
                break
            except Exception:
                pass
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.dpi"] = 120


setup_font()

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .small { font-size: 0.92rem; color: #666; }
      .card { background:#fafafa; border:1px solid #eee; border-radius:14px; padding:0.9rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=86400, show_spinner=False)
def wb_indicator(indicator_code: str, value_name: str) -> pd.DataFrame:
    url = (
        f"https://api.worldbank.org/v2/country/{COUNTRY_PARAM}/indicator/{indicator_code}"
        f"?format=json&date={START_YEAR}:{END_YEAR}&per_page=20000"
    )
    response = requests.get(url, timeout=40)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list) or len(data) < 2 or data[1] is None:
        raise ValueError(f"World Bank API 응답이 비어 있습니다: {indicator_code}")

    out = pd.DataFrame(data[1])[["countryiso3code", "date", "value"]].copy()
    out = out.rename(columns={"countryiso3code": "country", "date": "year", "value": value_name})
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    out = out.dropna(subset=["country", "year"])
    out["year"] = out["year"].astype(int)
    return out


@st.cache_data(show_spinner=False)
def robot_density_full() -> pd.DataFrame:
    rows = [
        ("KOR", 2010, 347), ("KOR", 2015, 531), ("KOR", 2020, 855), ("KOR", 2023, 1012),
        ("JPN", 2010, 308), ("JPN", 2015, 305), ("JPN", 2020, 364), ("JPN", 2023, 419),
        ("CHN", 2010, 68), ("CHN", 2015, 49), ("CHN", 2020, 246), ("CHN", 2023, 470),
        ("DEU", 2010, 261), ("DEU", 2015, 309), ("DEU", 2020, 371), ("DEU", 2023, 415),
        ("USA", 2010, 176), ("USA", 2015, 189), ("USA", 2020, 255), ("USA", 2023, 285),
        ("FRA", 2010, 122), ("FRA", 2015, 132), ("FRA", 2020, 177), ("FRA", 2023, 194),
        ("ITA", 2010, 159), ("ITA", 2015, 185), ("ITA", 2020, 224), ("ITA", 2023, 241),
        ("ESP", 2010, 136), ("ESP", 2015, 152), ("ESP", 2020, 191), ("ESP", 2023, 209),
        ("NLD", 2010, 84), ("NLD", 2015, 92), ("NLD", 2020, 110), ("NLD", 2023, 125),
        ("CAN", 2010, 121), ("CAN", 2015, 130), ("CAN", 2020, 165), ("CAN", 2023, 180),
        ("MEX", 2010, 33), ("MEX", 2015, 36), ("MEX", 2020, 47), ("MEX", 2023, 55),
        ("IND", 2010, 3), ("IND", 2015, 3), ("IND", 2020, 4), ("IND", 2023, 5),
        ("BRA", 2010, 10), ("BRA", 2015, 11), ("BRA", 2020, 13), ("BRA", 2023, 15),
        ("TUR", 2010, 14), ("TUR", 2015, 18), ("TUR", 2020, 26), ("TUR", 2023, 33),
        ("SGP", 2010, 220), ("SGP", 2015, 398), ("SGP", 2020, 605), ("SGP", 2023, 730),
    ]
    anchor = pd.DataFrame(rows, columns=["country", "year", "robot_density"])
    grid = pd.MultiIndex.from_product([COUNTRIES, range(START_YEAR, END_YEAR + 1)], names=["country", "year"]).to_frame(index=False)
    df = grid.merge(anchor, on=["country", "year"], how="left").sort_values(["country", "year"])
    df["robot_density"] = df.groupby("country")["robot_density"].transform(lambda s: s.interpolate(limit_direction="both"))
    return df


@st.cache_data(ttl=86400, show_spinner=True)
def build_from_api() -> pd.DataFrame:
    industry = wb_indicator("SL.IND.EMPL.ZS", "industry")
    service = wb_indicator("SL.SRV.EMPL.ZS", "service")
    gdp = wb_indicator("NY.GDP.PCAP.KD", "gdp")

    df = industry.merge(service, on=["country", "year"], how="inner")
    df = df.merge(gdp, on=["country", "year"], how="inner")
    df = df.merge(robot_density_full(), on=["country", "year"], how="inner")
    df = df[df["country"].isin(COUNTRIES)].copy()
    df["country_name"] = df["country"].map(COUNTRY_NAME)
    df = df.dropna(subset=["industry", "service", "gdp", "robot_density"])
    return df.sort_values(["country", "year"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def fallback_csv() -> pd.DataFrame:
    df = pd.read_csv("final_dataset.csv")
    for col in ["year", "industry", "service", "gdp", "robot_density"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["country"].isin(COUNTRIES)].copy()
    df = df[df["year"].between(START_YEAR, END_YEAR)]
    df["country_name"] = df["country"].map(COUNTRY_NAME)
    df = df.dropna(subset=["industry", "service", "gdp", "robot_density"])
    return df.sort_values(["country", "year"]).reset_index(drop=True)


def load_dataset():
    try:
        return build_from_api(), "World Bank API 자동 수집 + 로봇밀도 보간"
    except Exception as err:
        try:
            st.warning(f"API 호출 실패. 저장소의 final_dataset.csv를 예비 데이터로 사용합니다. 오류: {err}")
            return fallback_csv(), "저장소 final_dataset.csv 예비 데이터"
        except Exception as err2:
            st.error("API 데이터와 예비 CSV 로드가 모두 실패했습니다.")
            st.exception(err2)
            st.stop()


def label(col: str) -> str:
    return {
        "robot_density": "로봇밀도",
        "industry": "산업 고용 비중",
        "service": "서비스업 고용 비중",
        "gdp": "1인당 GDP",
        "log_gdp": "log(1인당 GDP)",
    }.get(col, col)


def safe_log(s: pd.Series) -> pd.Series:
    return np.log(pd.to_numeric(s, errors="coerce").clip(lower=1))


def strength(r: float) -> str:
    ar = abs(float(r))
    if ar < 0.3:
        return "약한 상관"
    if ar < 0.7:
        return "중간 정도의 상관"
    return "강한 상관"


def fit_explain(data: pd.DataFrame, y_col: str):
    reg = data.dropna(subset=[y_col, "robot_density", "gdp"]).copy()
    if len(reg) < 20:
        return None
    reg["log_gdp"] = safe_log(reg["gdp"])
    formula = f"{y_col} ~ robot_density + log_gdp + C(country) + C(year)"
    try:
        return smf.ols(formula, data=reg).fit(cov_type="cluster", cov_kwds={"groups": reg["country"]})
    except Exception:
        try:
            return smf.ols(formula, data=reg).fit(cov_type="HC1")
        except Exception:
            return None


def fit_forecast(data: pd.DataFrame, y_col: str):
    reg = data.dropna(subset=[y_col, "robot_density", "gdp"]).copy()
    if len(reg) < 20:
        return None
    reg["log_gdp"] = safe_log(reg["gdp"])
    try:
        return smf.ols(f"{y_col} ~ robot_density + log_gdp + C(country)", data=reg).fit(cov_type="HC1")
    except Exception:
        return None


def cagr(country_df: pd.DataFrame) -> float:
    start = country_df[country_df["year"] == 2015]["robot_density"]
    end = country_df[country_df["year"] == country_df["year"].max()]["robot_density"]
    if len(start) == 0 or len(end) == 0 or float(start.iloc[0]) <= 0:
        return 0.03
    years = int(country_df["year"].max()) - 2015
    return (float(end.iloc[0]) / float(start.iloc[0])) ** (1 / years) - 1 if years > 0 else 0.03


def scatter_line(data: pd.DataFrame, x_col: str, y_col: str, x_name: str, y_name: str):
    tmp = data[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(tmp[x_col], tmp[y_col])
    if len(tmp) >= 2 and tmp[x_col].std() > 0:
        slope, intercept = np.polyfit(tmp[x_col], tmp[y_col], 1)
        xs = np.linspace(float(tmp[x_col].min()), float(tmp[x_col].max()), 80)
        plt.plot(xs, slope * xs + intercept)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)


def line_chart(data: pd.DataFrame, y_col: str):
    fig = plt.figure(figsize=(9, 5))
    for country, group in data.groupby("country"):
        group = group.sort_values("year")
        plt.plot(group["year"], group[y_col], marker="o", label=country)
    plt.xlabel("연도")
    plt.ylabel(label(y_col))
    plt.title(f"국가별 {label(y_col)} 추세")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    st.pyplot(fig, clear_figure=True)


def tonggrami_df(data: pd.DataFrame) -> pd.DataFrame:
    return data[["country", "year", "robot_density", "industry", "service", "gdp"]].rename(columns={
        "country": "국가",
        "year": "연도",
        "robot_density": "로봇밀도",
        "industry": "산업고용비중",
        "service": "서비스업고용비중",
        "gdp": "GDP",
    })


def corr_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country, group in data.groupby("country"):
        for x_col, y_col in [("robot_density", "industry"), ("robot_density", "service"), ("industry", "service")]:
            t = group[[x_col, y_col]].dropna()
            if len(t) >= 3 and t[x_col].std() > 0 and t[y_col].std() > 0:
                r, p = stats.pearsonr(t[x_col], t[y_col])
                rows.append({
                    "국가": country,
                    "X변수": label(x_col),
                    "Y변수": label(y_col),
                    "표본수": len(t),
                    "상관계수_r": r,
                    "p_value": p,
                    "판정_0.05": "유의함" if p < 0.05 else "유의하지 않음",
                })
    return pd.DataFrame(rows)


def parse_numbers(text: str):
    parts = [p for p in text.replace("\n", ",").replace(" ", "").split(",") if p]
    try:
        return [float(p) for p in parts]
    except Exception:
        return None


st.title("자동화 확대와 고용 구조 변화 분석 앱")
st.caption("World Bank API 기반 고용·GDP 데이터 + 로봇밀도 보간 + 상관분석·회귀분석·예측·통그라미 CSV 생성")

with st.spinner("데이터셋을 생성하는 중입니다..."):
    df, data_note = load_dataset()

st.sidebar.header("분석 필터")
selected = st.sidebar.multiselect("국가 선택", COUNTRIES, default=["KOR", "JPN", "CHN", "DEU", "USA"])
year_range = st.sidebar.slider(
    "연도 범위",
    int(df["year"].min()),
    int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max())),
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 데이터 출처")
st.sidebar.markdown("- World Bank WDI: 고용 구조·GDP\n- 로봇밀도: 기준값 입력 후 선형 보간\n- API 실패 시 저장소 CSV 예비 사용")

d = df[df["country"].isin(selected) & df["year"].between(year_range[0], year_range[1])].copy()
if d.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "① 연구 소개",
    "② 추세 분석",
    "③ 상관관계 분석",
    "④ 회귀 분석",
    "⑤ 미래 예측",
    "⑥ 데이터/다운로드",
    "⑦ 학생용 계산기",
])

with tab1:
    st.subheader("연구 질문")
    st.markdown(
        """
        **핵심 연구 질문**

        > 자동화 수준을 나타내는 로봇밀도 증가는 국가별 고용 구조, 특히 산업 고용 비중과 서비스업 고용 비중에 어떤 관계를 보이는가?

        **분석 흐름**

        1. World Bank API에서 고용 구조와 GDP 데이터를 자동 수집
        2. 로봇밀도 기준값을 연도별로 보간
        3. 국가·연도별 데이터셋 구성
        4. 상관분석, 가설검정, 회귀분석, 미래 예측 수행
        5. 통그라미 업로드용 CSV 자동 생성
        """
    )
    st.info("**H0**: 로봇밀도와 고용 비중 사이에는 상관관계가 없다.\n\n**H1**: 로봇밀도와 고용 비중 사이에는 상관관계가 있다.")
    st.warning("`SL.IND.EMPL.ZS`는 제조업만이 아니라 산업 부문 고용 비중이다. 보고서에는 '산업 고용 비중'으로 표기하는 것이 안전하다.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("국가 수", f"{df['country'].nunique()}개")
    c2.metric("연도 범위", f"{df['year'].min()}~{df['year'].max()}")
    c3.metric("전체 행 수", f"{len(df)}개")
    c4.metric("선택 행 수", f"{len(d)}개")
    st.caption(f"현재 데이터 생성 방식: {data_note}")

with tab2:
    st.subheader("국가별 추세 분석")
    trend_var = st.selectbox("추세를 볼 변수", ["robot_density", "industry", "service", "gdp"], format_func=label)
    line_chart(d, trend_var)
    st.dataframe(d.groupby("country")[trend_var].agg(["mean", "min", "max"]).reset_index(), use_container_width=True)

with tab3:
    st.subheader("상관관계 분석과 가설검정")
    pair = st.selectbox("분석할 변수 조합", ["로봇밀도 vs 산업 고용 비중", "로봇밀도 vs 서비스업 고용 비중", "산업 고용 비중 vs 서비스업 고용 비중"])
    if pair == "로봇밀도 vs 산업 고용 비중":
        x_col, y_col = "robot_density", "industry"
    elif pair == "로봇밀도 vs 서비스업 고용 비중":
        x_col, y_col = "robot_density", "service"
    else:
        x_col, y_col = "industry", "service"
    alpha = st.selectbox("유의수준 α", [0.10, 0.05, 0.01], index=1)
    test = d[[x_col, y_col, "country", "year"]].dropna()
    st.info(f"**H0**: {label(x_col)}와(과) {label(y_col)} 사이에는 상관관계가 없다.\n\n**H1**: {label(x_col)}와(과) {label(y_col)} 사이에는 상관관계가 있다.")
    if len(test) < 3 or test[x_col].std() == 0 or test[y_col].std() == 0:
        st.error("상관분석을 수행하기에 표본 수가 부족하거나 변수의 분산이 0입니다.")
    else:
        r, p = stats.pearsonr(test[x_col], test[y_col])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("표본 수 n", f"{len(test)}")
        c2.metric("상관계수 r", f"{r:.4f}")
        c3.metric("p-value", f"{p:.6f}")
        c4.metric("판정", "귀무가설 기각" if p < alpha else "기각 불가")
        if p < alpha:
            st.success(f"p-value={p:.6f} < α={alpha}. 귀무가설을 기각한다. 두 변수 사이에는 통계적으로 유의한 {strength(r)}이 있다.")
        else:
            st.warning(f"p-value={p:.6f} ≥ α={alpha}. 귀무가설을 기각하지 못한다. 이번 표본에서는 유의한 상관이라고 말할 근거가 부족하다.")
        scatter_line(test, x_col, y_col, label(x_col), label(y_col))
        rows = []
        for country, group in d.groupby("country"):
            t = group[[x_col, y_col]].dropna()
            if len(t) >= 3 and t[x_col].std() > 0 and t[y_col].std() > 0:
                rr, pp = stats.pearsonr(t[x_col], t[y_col])
                rows.append({"국가": country, "표본수": len(t), "상관계수 r": rr, "p-value": pp, "해석": strength(rr), "판정": "유의함" if pp < alpha else "유의하지 않음"})
        if rows:
            st.markdown("### 국가별 상관계수 랭킹")
            st.dataframe(pd.DataFrame(rows).sort_values("상관계수 r"), use_container_width=True)

with tab4:
    st.subheader("고정효과 회귀 분석")
    y_col = st.selectbox("종속변수 선택", ["industry", "service"], format_func=label, key="reg_y")
    st.code(f"{y_col} ~ robot_density + log_gdp + C(country) + C(year)", language="text")
    model = fit_explain(d, y_col)
    if model is None:
        st.error("회귀분석을 수행할 표본이 부족합니다.")
    else:
        beta = float(model.params.get("robot_density", np.nan))
        p_value = float(model.pvalues.get("robot_density", np.nan))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("로봇밀도 계수 β", f"{beta:.5f}")
        c2.metric("p-value", f"{p_value:.6f}")
        c3.metric("R²", f"{model.rsquared:.3f}")
        c4.metric("표본 수", f"{int(model.nobs)}")
        if p_value < 0.05:
            st.success("로봇밀도 계수가 통계적으로 유의합니다.")
        else:
            st.warning("이번 회귀모형에서는 로봇밀도 계수가 통계적으로 유의하다고 보기 어렵습니다.")
        coef = pd.DataFrame({"계수": model.params, "p-value": model.pvalues}).reset_index().rename(columns={"index": "항목"})
        st.dataframe(coef[coef["항목"].isin(["Intercept", "robot_density", "log_gdp"])], use_container_width=True)
        with st.expander("전체 회귀 결과 보기"):
            st.text(model.summary())

with tab5:
    st.subheader("미래 예측")
    country = st.selectbox("예측할 국가", COUNTRIES, index=COUNTRIES.index("KOR"), format_func=lambda x: f"{x} ({COUNTRY_NAME.get(x, x)})")
    y_col = st.selectbox("예측할 고용 비중", ["industry", "service"], format_func=label, key="forecast_y")
    country_df = df[df["country"] == country].sort_values("year")
    base = country_df.iloc[-1]
    base_year = int(base["year"])
    horizon = st.slider("예측 종료 연도", base_year + 1, 2035, 2030)
    mode = st.radio("로봇밀도 미래 경로", ["최근 추세 자동 적용", "직접 성장률 설정"], horizontal=True)
    rate = cagr(country_df)
    if mode == "직접 성장률 설정":
        rate = st.slider("연평균 로봇밀도 증가율", -0.05, 0.20, float(rate), 0.01)
    years = list(range(base_year, horizon + 1))
    robot_path = [float(base["robot_density"]) * ((1 + rate) ** (year - base_year)) for year in years]
    model = fit_forecast(df, y_col)
    if model is None:
        st.error("예측 모델을 만들 수 없습니다.")
    else:
        future = pd.DataFrame({"country": country, "year": years, "robot_density": robot_path, "gdp": float(base["gdp"])})
        future["log_gdp"] = safe_log(future["gdp"])
        future[f"pred_{y_col}"] = model.predict(future)
        c1, c2, c3 = st.columns(3)
        c1.metric("기준 연도", str(base_year))
        c2.metric("로봇밀도 경로", f"{robot_path[0]:.1f} → {robot_path[-1]:.1f}")
        c3.metric("예측 변화", f"{future[f'pred_{y_col}'].iloc[-1] - future[f'pred_{y_col}'].iloc[0]:.2f}%p")
        fig = plt.figure(figsize=(9, 5))
        plt.plot(future["year"], future[f"pred_{y_col}"], marker="o")
        plt.xlabel("연도")
        plt.ylabel(f"예측 {label(y_col)}")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)
        st.dataframe(future, use_container_width=True)
        st.caption("예측은 확정값이 아니라 시나리오 기반 추정이다.")

with tab6:
    st.subheader("데이터 확인 / 다운로드")
    c1, c2, c3 = st.columns(3)
    c1.metric("선택 국가 수", f"{d['country'].nunique()}개")
    c2.metric("연도 범위", f"{year_range[0]}~{year_range[1]}")
    c3.metric("행 수", f"{len(d)}개")
    st.dataframe(d.head(100), use_container_width=True)
    st.download_button("📥 필터 적용 데이터 다운로드", d.to_csv(index=False).encode("utf-8-sig"), f"filtered_dataset_{year_range[0]}_{year_range[1]}.csv", "text/csv")
    st.download_button("📥 전체 final_dataset 다운로드", df.to_csv(index=False).encode("utf-8-sig"), "final_dataset.csv", "text/csv")
    st.download_button("📥 통그라미용 CSV 다운로드", tonggrami_df(df).to_csv(index=False).encode("utf-8-sig"), "통그라미용_자동화_고용구조.csv", "text/csv")
    st.download_button("📥 국가별 상관분석 결과표 다운로드", corr_summary(df).to_csv(index=False).encode("utf-8-sig"), "correlation_summary.csv", "text/csv")

with tab7:
    st.subheader("학생용 상관관계 계산기")
    left, right = st.columns(2)
    with left:
        x_name = st.text_input("X 변수 이름", value="키(cm)")
        x_raw = st.text_area("X 값들", value="160, 165, 170, 175, 180", height=130)
    with right:
        y_name = st.text_input("Y 변수 이름", value="몸무게(kg)")
        y_raw = st.text_area("Y 값들", value="55, 60, 65, 72, 78", height=130)
    alpha = st.selectbox("유의수준 α", [0.10, 0.05, 0.01], index=1, key="calc_alpha")
    if st.button("상관관계 계산하기"):
        xs = parse_numbers(x_raw)
        ys = parse_numbers(y_raw)
        if xs is None or ys is None:
            st.error("숫자 파싱 실패. 문자나 특수기호를 확인하세요.")
            st.stop()
        if len(xs) != len(ys):
            st.error(f"X 개수({len(xs)})와 Y 개수({len(ys)})가 다릅니다.")
            st.stop()
        if len(xs) < 3:
            st.error("최소 3쌍 이상의 데이터가 필요합니다.")
            st.stop()
        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            st.error("X 또는 Y가 전부 같은 값입니다.")
            st.stop()
        r, p = stats.pearsonr(x, y)
        c1, c2, c3 = st.columns(3)
        c1.metric("표본 수 n", f"{len(x)}")
        c2.metric("상관계수 r", f"{r:.4f}")
        c3.metric("p-value", f"{p:.6f}")
        st.info(f"**H0**: {x_name}와(과) {y_name} 사이에는 상관관계가 없다.\n\n**H1**: {x_name}와(과) {y_name} 사이에는 상관관계가 있다.")
        if p < alpha:
            st.success(f"p-value={p:.6f} < α={alpha}. 귀무가설을 기각한다. 통계적으로 유의한 {strength(r)}이 있다.")
        else:
            st.warning(f"p-value={p:.6f} ≥ α={alpha}. 귀무가설을 기각하지 못한다.")
        out = pd.DataFrame({x_name: x, y_name: y})
        st.dataframe(out, use_container_width=True)
        scatter_line(out, x_name, y_name, x_name, y_name)
        st.download_button("입력 데이터 CSV 다운로드", out.to_csv(index=False).encode("utf-8-sig"), "correlation_input.csv", "text/csv")
