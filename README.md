# 자동화와 고용 구조 변화 분석 앱

Streamlit 기반 통계 분석 앱입니다.

## 주요 기능

- World Bank API에서 고용 구조와 GDP 데이터 자동 수집
- 로봇밀도 기준값 보간
- 국가별 추세 분석
- 상관관계 분석 및 가설검정
- 고정효과 회귀분석
- 미래 예측
- 통그라미용 CSV 다운로드
- 학생용 상관관계 계산기

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 데이터 주의

World Bank의 `SL.IND.EMPL.ZS`는 제조업만이 아니라 산업 부문 고용 비중입니다. 보고서나 포스터에서는 `산업 고용 비중` 또는 `제조업 포함 산업 고용 비중`으로 표기하는 것이 안전합니다.
