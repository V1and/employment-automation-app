import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import statsmodels.formula.api as smf
import os
from scipy import stats

def setup_korean_font():
    # 경로는 대소문자 정확히!
    font_path = os.path.join(os.getcwd(), "fonts", "NotoSansKR-Regular.otf")

    if os.path.isfile(font_path):
        # 1) 폰트 등록
        fm.fontManager.addfont(font_path)
        # 2) 폰트 이름 얻기
        font_name = fm.FontProperties(fname=font_path).get_name()
        # 3) 기본 폰트로 지정
        mpl.rcParams["font.family"] = font_name
        mpl.rcParams["font.sans-serif"] = [font_name]
    else:
        # 폰트 없으면 기본(영문만 보장)
        mpl.rcParams["font.family"] = "DejaVu Sans"

    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["font.size"] = 13
    mpl.rcParams["figure.dpi"] = 120

setup_korean_font()


# =========================
# 기본 설정 / 스타일
# =========================
st.set_page_config(page_title="자동화×고용 구조 분석", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
      h1 { margin-bottom: 0.2rem; }
      .subtle { color: #666; font-size: 0.95rem; margin-top: -0.2rem; }
      .card { background: #fafafa; padding: 0.9rem 1rem; border-radius: 14px; border: 1px solid #eee; }
      .small { font-size: 0.9rem; color: #666; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("자동화(로봇 밀도)와 고용 구조 변화 분석")
st.markdown('<div class="subtle">15개국 패널데이터 기반 · 추세/관계/회귀/시나리오 예측을 한 화면에서 확인</div>', unsafe_allow_html=True)

DATA_FILE = "final_dataset.csv"

# =========================
# 데이터 로드
# =========================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in ["year", "industry", "service", "gdp", "robot_density"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["country", "year"]).copy()
    df["year"] = df["year"].astype(int)

    return df

def safe_log(s: pd.Series) -> pd.Series:
    return np.log(s.clip(lower=1))

def fit_panel_explain(reg_df: pd.DataFrame, y_col: str):
    """설명/분석용: 국가 + 연도 FE, (가능하면) 국가별 클러스터 SE"""
    reg_df = reg_df.dropna(subset=[y_col, "robot_density", "gdp"]).copy()
    if reg_df.empty:
        return None

    reg_df["log_gdp"] = safe_log(reg_df["gdp"])
    formula = f"{y_col} ~ robot_density + log_gdp + C(country) + C(year)"

    n_groups = reg_df["country"].nunique()
    if n_groups >= 2:
        return smf.ols(formula, data=reg_df).fit(
            cov_type="cluster",
            cov_kwds={"groups": reg_df["country"]}
        )
    else:
        return smf.ols(formula, data=reg_df).fit(cov_type="HC1")


def fit_panel_forecast(reg_df: pd.DataFrame, y_col: str):
    """예측용: 국가 FE만 사용(연도 FE 제거), robust"""
    reg_df = reg_df.dropna(subset=[y_col, "robot_density", "gdp"]).copy()
    if reg_df.empty:
        return None

    reg_df["log_gdp"] = safe_log(reg_df["gdp"])
    formula = f"{y_col} ~ robot_density + log_gdp + C(country)"

    # 예측에서는 표준오차보다 안정적인 predict가 중요 → HC1로 고정
    return smf.ols(formula, data=reg_df).fit(cov_type="HC1")

def compute_cagr(country_df: pd.DataFrame, start_year: int, end_year: int):
    g = country_df.dropna(subset=["robot_density"]).sort_values("year")
    s = g[g["year"] == start_year]["robot_density"]
    e = g[g["year"] == end_year]["robot_density"]
    if len(s) == 0 or len(e) == 0:
        return None
    s_val, e_val = float(s.iloc[0]), float(e.iloc[0])
    if s_val <= 0 or e_val <= 0 or end_year <= start_year:
        return None
    years = end_year - start_year
    return (e_val / s_val) ** (1 / years) - 1

def pick_cagr_window(country_df: pd.DataFrame):
    years = sorted(country_df.dropna(subset=["robot_density"])["year"].unique().tolist())
    if not years:
        return None
    # 가능하면 2015~2023, 없으면 마지막 8~10년 구간, 없으면 전체(>=5년)
    if 2015 in years and 2023 in years:
        return (2015, 2023)
    last = years[-1]
    for span in [8, 10, 6, 5]:
        if (last - span) in years:
            return (last - span, last)
    if years[-1] - years[0] >= 5:
        return (years[0], years[-1])
    return None

def project_robot_path(base_rd: float, base_year: int, horizon_year: int, cagr: float):
    years = list(range(base_year, horizon_year + 1))
    vals = [base_rd * ((1 + cagr) ** (y - base_year)) for y in years]
    return pd.Series(vals, index=years)

# =========================
# 본문 시작
# =========================
try:
    df = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(f"❌ '{DATA_FILE}' 파일이 app.py와 같은 폴더에 없습니다. final_dataset.csv를 같은 폴더에 두세요.")
    st.stop()

countries = sorted(df["country"].unique().tolist())
default_country = "KOR" if "KOR" in countries else countries[0]
min_year, max_year = int(df["year"].min()), int(df["year"].max())

# =========================
# 사이드바
# =========================
st.sidebar.header("설정")

view_mode = st.sidebar.radio("보기 방식", ["단일 국가", "다국가 비교"], index=0)

if view_mode == "단일 국가":
    selected = [st.sidebar.selectbox("국가 선택 (ISO3)", countries, index=countries.index(default_country))]
else:
    base_defaults = [c for c in [default_country, "CHN", "DEU"] if c in countries]
    if not base_defaults:
        base_defaults = [default_country]
    selected = st.sidebar.multiselect("국가 선택 (ISO3)", countries, default=base_defaults)

    st.sidebar.markdown("---")
st.sidebar.markdown("### 데이터 출처")
st.sidebar.markdown(
    """
- World Bank WDI (고용 구조·GDP)
- OECD Employment Database (보조 확인)
- IFR / Robot density (로봇 밀도)
- UN Comtrade / WITS (수출 집중도 분석용)
"""
)

# 로봇 데이터가 의미 있는 구간을 기본으로
default_start = 2010 if max_year >= 2010 else min_year
default_end = 2023 if max_year >= 2023 else max_year
year_range = st.sidebar.slider("연도 범위(과거 데이터)", min_year, max_year, (default_start, default_end))

d = df[df["country"].isin(selected) & df["year"].between(year_range[0], year_range[1])].copy()
d = d.sort_values(["country", "year"])

# =========================
# 상단 요약 카드
# =========================
st.subheader("요약")
m1, m2, m3, m4 = st.columns(4)
m1.metric("선택 국가 수", f"{d['country'].nunique():,}개")
m2.metric("데이터 행 수", f"{len(d):,}개")
m3.metric("기간", f"{year_range[0]}–{year_range[1]}")
miss_robot = d["robot_density"].isna().mean() * 100 if len(d) else 0
m4.metric("로봇밀도 결측률", f"{miss_robot:.1f}%")

st.markdown(
    '<div class="card small">TIP: 1960~ 구간은 고용/로봇 데이터 결측이 많습니다. 분석은 보통 2010년 이후가 안정적입니다.</div>',
    unsafe_allow_html=True
)

st.divider()

# =========================
# 탭 구성
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "① 추세 분석",
    "② 관계 분석",
    "③ 회귀 분석",
    "④ 미래 예측",
    "⑤ 데이터/다운로드",
    "⑥ 가설검정(상관)"
])

# =========================
# ① 추세
# =========================
with tab1:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("고용 구조 추세 (제조업 vs 서비스업)")
        fig = plt.figure()
        for c in selected:
            sub = d[d["country"] == c].sort_values("year")
            if sub.empty:
                continue
            plt.plot(sub["year"], sub["industry"], marker="o", label=f"{c} 제조업")
            plt.plot(sub["year"], sub["service"], marker="o", linestyle="--", label=f"{c} 서비스업")
            sub = d[d["country"] == c].sort_values("year").dropna(subset=["service"])
        plt.xlabel("연도")
        plt.ylabel("전체 고용 대비 비율(%)")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("로봇 밀도 추세")
        fig2 = plt.figure()
        for c in selected:
            sub = d[d["country"] == c].dropna(subset=["robot_density"]).sort_values("year")
            if sub.empty:
                continue
            plt.plot(sub["year"], sub["robot_density"], marker="o", label=c)
        plt.xlabel("연도")
        plt.ylabel("근로자 1만 명당 로봇 수")
        plt.grid(True, alpha=0.3)
        plt.legend()
        st.pyplot(fig2, clear_figure=True)

# =========================
# ② 관계 (상관/산점도)
# =========================
with tab2:
    st.subheader("로봇 밀도 ↔ 고용 비중 관계 (설명적 분석)")
    rel = d.dropna(subset=["robot_density", "industry", "service"]).copy()

    if rel.empty:
        st.error("분석 가능한 행이 없습니다(로봇 밀도/고용 비중 결측). 연도 범위를 2010년 이후로 줄여보세요.")
    else:
        target_label = st.selectbox("분석 대상", ["제조업 고용 비중", "서비스업 고용 비중"], index=0)
        target = "industry" if target_label.startswith("제조업") else "service"

        x = rel["robot_density"].to_numpy()
        y = rel[target].to_numpy()
        corr = np.corrcoef(x, y)[0, 1] if len(rel) > 2 else np.nan

        c1, c2 = st.columns(2)
        c1.metric("상관계수", f"{corr:.3f}" if np.isfinite(corr) else "NA")
        c2.metric("사용된 행 수", f"{len(rel):,}")

        fig = plt.figure()
        plt.scatter(x, y)
        plt.xlabel("로봇 밀도(근로자 1만 명당)")
        plt.ylabel(target_label + "(%)")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)

        st.caption("주의: 상관계수는 인과를 증명하지 않습니다. 통제변수/고정효과를 포함한 회귀로 보완합니다.")

# =========================
# ③ 회귀 (국가/연도 고정효과)
# =========================
with tab3:
    st.subheader("패널 회귀 분석 (국가·연도 고정효과, GDP 통제)")

    # 회귀는 다국가 기반이 더 안정적이므로, 선택 화면이 단일 국가여도 reg는 전체에서 기간만 자름
    reg = df[df["year"].between(year_range[0], year_range[1])].dropna(
        subset=["robot_density", "industry", "service", "gdp"]
    ).copy()

    if reg.empty:
        st.error("회귀에 사용할 데이터가 없습니다. 연도 범위를 조정하세요.")
    else:
        y_label = st.selectbox("종속변수(Y)", ["제조업 고용 비중(%)", "서비스업 고용 비중(%)"], index=0)
        y_col = "industry" if y_label.startswith("제조업") else "service"

        model = fit_panel_explain(reg, y_col)
        if model is None:
            st.error("모형 적합 실패(결측/데이터 부족).")
        else:
            beta = float(model.params.get("robot_density", np.nan))
            se = float(model.bse.get("robot_density", np.nan))
            pval = float(model.pvalues.get("robot_density", np.nan))

            a, b, c = st.columns(3)
            a.metric("로봇 밀도 계수(β)", f"{beta:.4f}")
            b.metric("표준오차(SE)", f"{se:.4f}")
            c.metric("유의확률(p-value)", f"{pval:.4g}")

            st.markdown(
                '<div class="card small">'
                "해석: 로봇 밀도 1 증가(=근로자 1만명당 로봇 1대 증가) 시, "
                f"다른 조건을 통제한 상태에서 <b>{y_label}</b>가 β만큼 변한다고 추정."
                "</div>",
                unsafe_allow_html=True
            )

            with st.expander("회귀 결과 전체 보기(전문)"):
                st.text(model.summary().as_text())

                st.subheader("구간 민감도 분석(β 부호/크기 변화)")

    windows = [
        (2010, 2023),
        (2013, 2023),
        (2015, 2023),
        (2017, 2023),
    ]

rows = []
for a, b in windows:
    reg_w = df[df["year"].between(a, b)].dropna(subset=["robot_density","industry","service","gdp"]).copy()
    m = fit_panel_explain(reg, y_col)
    if m is None:
        rows.append([f"{a}-{b}", np.nan, np.nan, len(reg_w), reg_w["country"].nunique()])
        continue
    beta_w = float(m.params.get("robot_density", np.nan))
    p_w = float(m.pvalues.get("robot_density", np.nan))
    rows.append([f"{a}-{b}", beta_w, p_w, len(reg_w), reg_w["country"].nunique()])

sens = pd.DataFrame(rows, columns=["기간", "β(로봇밀도)", "p-value", "표본 행", "국가 수"])
st.dataframe(sens, use_container_width=True)

st.markdown("### 한 줄 요약")
if pval < 0.05:
    st.success(f"로봇밀도 계수는 통계적으로 유의(p<0.05)하며, {y_col}에 대한 방향성(부호)이 명확합니다.")
else:
    st.info("로봇밀도 계수는 통계적으로 유의하지 않을 수 있습니다. 표본기간/국가구성에 따른 민감도 분석이 필요합니다.")

# =========================
# ④ 예측(시나리오)
# =========================
with tab4:
    st.subheader("미래 시나리오 예측")

    # -------------------------
    # 1) 예측용 회귀 모형 적합
    # -------------------------
    reg = df[df["year"].between(year_range[0], year_range[1])].dropna(
        subset=["robot_density", "industry", "service", "gdp"]
    ).copy()

    y_col = st.selectbox("예측 대상 선택", ["industry", "service"])
    model = fit_panel_forecast(reg, y_col)

    if model is None:
        st.error("회귀모형 적합 실패")
        st.stop()

    # -------------------------
    # 2) 기준 국가 선택
    # -------------------------
    country = st.selectbox("예측 국가 선택", countries)

    base = df[df["country"] == country].dropna(
        subset=["robot_density", y_col, "gdp"]
    ).sort_values("year")

    if base.empty:
        st.error("해당 국가 데이터 부족")
        st.stop()

    last = base.iloc[-1]
    base_year = int(last["year"])
    base_rd = float(last["robot_density"])
    base_gdp = float(last["gdp"])

    st.write(f"기준 연도: {base_year}")
    st.write(f"현재 로봇 밀도: {base_rd:.1f}")

    # -------------------------
    # 3) 미래 연도 범위 먼저 생성
    # -------------------------
    horizon = st.slider(
        "예측 목표 연도",
        base_year,
        base_year + 10,
        base_year + 5
    )
    future_years = list(range(base_year, horizon + 1))

    # -------------------------
    # 4) 로봇 밀도 경로 생성 (자동/시나리오)
    # -------------------------
    st.markdown("### 로봇 밀도 예측 방식")

    auto_mode = st.radio(
        "로봇 밀도 예측 방식",
        ["자동 예측(기본)", "시나리오(직접 설정)"],
        index=0,
        horizontal=True
    )

    hist = df[df["country"] == country].dropna(subset=["robot_density"]).sort_values("year")
    if hist.empty:
        st.error("로봇 밀도 시계열이 없어 자동 예측 불가")
        st.stop()

    if auto_mode == "자동 예측(기본)":
        method = st.selectbox("자동 예측 방법", ["선형 추세", "CAGR(연평균성장률)"], index=0)

        if method == "선형 추세":
            N = st.slider("추세 추정에 쓸 최근 연도 수", 5, 15, 8)
            hist2 = hist.tail(N)

            x = hist2["year"].to_numpy()
            y = hist2["robot_density"].to_numpy()

            a, b = np.polyfit(x, y, 1)
            rd_path = a * np.array(future_years) + b
            rd_path = np.clip(rd_path, 0, None)

        else:
            win = pick_cagr_window(hist)
            if win is None:
                st.warning("CAGR 추정 불가 → 선형 추세로 대체합니다.")
                x = hist["year"].tail(8).to_numpy()
                y = hist["robot_density"].tail(8).to_numpy()
                a, b = np.polyfit(x, y, 1)
                rd_path = np.clip(a * np.array(future_years) + b, 0, None)
            else:
                cagr = compute_cagr(hist, win[0], win[1]) or 0.0
                st.caption(f"CAGR 추정 구간: {win[0]}→{win[1]} / 성장률: {cagr*100:.2f}%")
                rd_path = project_robot_path(base_rd, base_year, horizon, cagr).values

    else:
        future_rd = st.slider(
            "목표 로봇 밀도(시나리오)",
            float(base_rd),
            float(max(2000.0, base_rd * 2)),
            float(base_rd)
        )
        rd_path = np.linspace(base_rd, future_rd, len(future_years))

    # -------------------------
    # 5) 예측용 데이터프레임 생성 + 예측
    # -------------------------
    future_df = pd.DataFrame({
        "country": country,
        "year": future_years,
        "robot_density": rd_path,
        "gdp": base_gdp
    })
    future_df["log_gdp"] = np.log(future_df["gdp"].clip(lower=1))

    y_pred = model.predict(future_df)

    # -------------------------
    # 6) 그래프 출력
    # -------------------------
    fig = plt.figure()
    plt.plot(future_years, y_pred, marker="o")
    plt.xlabel("연도")
    plt.ylabel(f"{'제조업' if y_col=='industry' else '서비스업'} 고용 비중(%) 예측")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)
# -------------------------
# 7) 해석 블록 (자동 생성)
# -------------------------
beta = float(model.params.get("robot_density", np.nan))
delta_rd = float(rd_path[-1] - rd_path[0])
delta_y = float(y_pred.iloc[-1] - y_pred.iloc[0])

label_kor = "제조업" if y_col == "industry" else "서비스업"

st.markdown("### 해석(자동 생성)")
c1, c2, c3 = st.columns(3)
c1.metric("로봇밀도 계수 β", f"{beta:.6f}" if np.isfinite(beta) else "NA")
c2.metric("로봇밀도 변화(Δ)", f"{delta_rd:+.2f}")
c3.metric(f"{label_kor} 예측 변화(Δ)", f"{delta_y:+.3f} %p")

if not np.isfinite(beta):
    st.warning("로봇밀도 계수를 계산할 수 없습니다. 데이터 결측/표본 기간을 점검하세요.")
else:
    if abs(beta) < 1e-4:
        st.info(
            f"β가 0에 매우 가까워 로봇밀도 변화가 {label_kor} 예측치에 거의 반영되지 않습니다. "
            "이 경우 고용 구조는 로봇보다 GDP·국가 특성의 영향이 더 클 수 있습니다."
        )
    elif beta > 0:
        st.success(
            f"β>0 이므로, 로봇밀도 증가가 {label_kor} 비중 증가와 같은 방향으로 추정됩니다."
        )
    else:
        st.warning(
            f"β<0 이므로, 로봇밀도 증가가 {label_kor} 비중 감소와 같은 방향으로 추정됩니다."
        )

st.caption("주의: 본 예측은 ‘시나리오/추세 기반’이며, 미래의 정책·경기 충격을 완전히 반영하지 않습니다.")
st.caption(f"로봇밀도 경로: {rd_path[0]:.1f} → {rd_path[-1]:.1f}")
# =========================
# ⑤ 데이터/다운로드
# =========================
with tab5:
    st.subheader("데이터 확인 / 다운로드")

    st.markdown("### 현재 필터 적용 데이터")
    st.write(f"- 선택 국가 수: **{d['country'].nunique()}개**")
    st.write(f"- 연도 범위: **{year_range[0]} ~ {year_range[1]}**")
    st.write(f"- 행 수: **{len(d)}개**")

    st.dataframe(d.head(50), use_container_width=True)

    st.markdown("### CSV 다운로드")

    # 다운로드용 CSV 생성
    csv_bytes = d.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 필터된 데이터 CSV 다운로드",
        data=csv_bytes,
        file_name=f"filtered_dataset_{year_range[0]}_{year_range[1]}.csv",
        mime="text/csv"
    )

    st.markdown("### 전체 데이터 다운로드(원본)")
    full_csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 전체 데이터 CSV 다운로드",
        data=full_csv,
        file_name="full_dataset.csv",
        mime="text/csv"
    )

    with tab6:

        st.subheader("연구 데이터 상관관계 분석")

        corr_df = df.dropna(
        subset=["robot_density","industry","service"]
        )

        option = st.selectbox(
        "분석 관계 선택",
        [
        "로봇밀도 vs 제조업 고용",
        "로봇밀도 vs 서비스업 고용",
        "제조업 vs 서비스업"
        ]
        )

        if option == "로봇밀도 vs 제조업 고용":
        x = corr_df["robot_density"]
        y = corr_df["industry"]
        label_y = "제조업 고용 비중"

        elif option == "로봇밀도 vs 서비스업 고용":
        x = corr_df["robot_density"]
        y = corr_df["service"]
        label_y = "서비스업 고용 비중"

        else:
        x = corr_df["industry"]
        y = corr_df["service"]
        label_y = "서비스업 고용 비중"

        from scipy import stats
        r, p = stats.pearsonr(x, y)

        st.metric("상관계수 r", f"{r:.3f}")
        st.metric("p-value", f"{p:.5f}")

        if p < 0.05:
        st.success("통계적으로 유의한 상관관계 존재")
        else:
        st.warning("통계적으로 유의하지 않음")

        fig = plt.figure()
        plt.scatter(x, y)
        plt.xlabel("X 변수")
        plt.ylabel(label_y)
        plt.grid(True)
        st.pyplot(fig)