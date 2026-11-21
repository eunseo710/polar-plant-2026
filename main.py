import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os

st.set_page_config(page_title="극지식물 생육 차이 분석", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
    html, body, [class*="css"] {
       font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- 파일 읽기 ----------
@st.cache_data
def load_csv(file):
    if file is None:
        return None
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

def school_name_from_file(file):
    if file is None:
        return ""
    name = os.path.splitext(os.path.basename(file.name))[0]
    for suffix in ("등학교", "학교"):
        name = name.replace(suffix, "")
    return name.replace("인천아라고", "인천 아라고").strip()

env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")
growth_items = st.sidebar.multiselect(
    "생육 측정 항목",
    ["생중량", "잎 수", "지상부 길이", "지하부 길이"],
    ["생중량", "잎 수", "지상부 길이", "지하부 길이"],
)

env_df = load_csv(env_file)
growth_df = load_csv(growth_file)

if env_df is not None and not env_df.empty:
    env_df = std_cols(env_df)
if growth_df is not None and not growth_df.empty:
    growth_df = std_cols(growth_df)

school_name = school_name_from_file(env_file) or school_name_from_file(growth_file)

tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.subheader("연구 배경 및 목적")
        st.write(
            "극지 환경에 자라는 식물의 생육이 온도·습도 같은 주변 환경 조건과 어떤 관계가 있는지 "
            "개체별로 비교합니다. 환경 데이터로 성장 조건을 살펴보고, 생육 데이터로 개체별 차이를 확인합니다."
        )

        st.subheader("데이터 요약")
        rows = []
        if growth_df is not None and not growth_df.empty:
            growth_valid = growth_df.dropna(subset=["생중량", "잎 수", "지상부 길이", "지하부 길이"])
            rows.append(["개체 수", len(growth_valid)])
        if env_df is not None and not env_df.empty and "time" in env_df.columns:
            t_min = env_df["time"].min()
            t_max = env_df["time"].max()
            rows.append(["측정 기간", f"{t_min.strftime('%Y-%m-%d')} ~ {t_max.strftime('%Y-%m-%d')}"])
        if env_df is not None and not env_df.empty:
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_cols:
                rows.append(["환경 측정 항목", ", ".join(env_cols)])
        if growth_df is not None and not growth_df.empty:
            growth_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
            if growth_cols:
                rows.append(["생육 측정 항목", ", ".join(growth_cols)])
        if rows:
            summary_df = pd.DataFrame(rows, columns=["항목", "값"])
            st.dataframe(summary_df, width="stretch")
        else:
            st.info("데이터가 없어 요약 정보를 표시할 수 없어요.")

        st.subheader("주요 지표")
        has_growth = growth_df is not None and not growth_df.empty and "생중량" in growth_df.columns
        has_env = env_df is not None and not env_df.empty
        avg_weight = growth_df["생중량"].mean() if has_growth else np.nan
        avg_temp = env_df["temperature"].mean() if (has_env and "temperature" in env_df.columns) else np.nan
        avg_humidity = env_df["humidity"].mean() if (has_env and "humidity" in env_df.columns) else np.nan

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 개체수", len(growth_df.dropna(subset=["생중량"])) if has_growth else "—")
        with col2:
            st.metric("평균 온도", f"{avg_temp:.1f}°C" if not np.isnan(avg_temp) else "—")
        with col3:
            st.metric("평균 습도", f"{avg_humidity:.1f}%" if not np.isnan(avg_humidity) else "—")
        with col4:
            st.metric("평균 생중량", f"{avg_weight:.2f}g" if not np.isnan(avg_weight) else "—")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab2:
    try:
        st.subheader("환경 요약 통계")
        if env_df is None or env_df.empty or "time" not in env_df.columns:
            st.info("환경 데이터가 없어요. 사이드바에서 환경 CSV를 올려주세요.")
        else:
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if not env_cols:
                st.info("환경 측정 항목(온도·습도·pH·EC)이 없어요.")
            else:
                env_summary = env_df[env_cols].agg(["mean", "min", "max"]).T
                env_summary.index = ["평균", "최소", "최대"]
                env_summary.columns.name = ""
                st.dataframe(env_summary, width="stretch")

                st.subheader("날짜별 환경 변화 (일별 평균)")
                env_daily = env_df.dropna(subset=env_cols).copy()
                env_daily["date"] = env_daily["time"].dt.normalize()
                daily_avg = env_daily.groupby("date")[env_cols].mean().reset_index()
                fig = make_subplots(
                    rows=len(env_cols),
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=[f"{c}" for c in env_cols],
                    vertical_spacing=0.08,
                )
                for i, c in enumerate(env_cols, start=1):
                    fig.add_trace(
                        go.Scatter(
                            x=daily_avg["date"],
                            y=daily_avg[c],
                            mode="lines",
                            name=c,
                            line=dict(width=2),
                        ),
                        row=i,
                        col=1,
                    )
                fig.update_layout(
                    height=250 * len(env_cols) + 60,
                    title_text="날짜별 환경 지표 평균",
                    margin=dict(t=40, b=10),
                )
                fig.update_xaxes(title_text="날짜")
                fig.update_yaxes(title_text="값")
                st.plotly_chart(fig, width="stretch")

                with st.expander("환경 데이터 원본 보기 및 다운로드"):
                    st.dataframe(env_df, width="stretch")
                    csv_env = env_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "환경 데이터 CSV 다운로드",
                        data=csv_env,
                        file_name=f"{school_name}_환경데이터.csv" if school_name else "환경데이터.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab3:
    try:
        st.subheader("핵심 결과")
        if growth_df is None or growth_df.empty or "생중량" not in growth_df.columns:
            st.info("생육 데이터가 없어요. 사이드바에서 생육 CSV를 올려주세요.")
        else:
            growth_valid = growth_df.dropna(subset=["생중량"])
            if len(growth_valid) == 0:
                st.info("생중량이 있는 개체가 없어요.")
            else:
                avg_weight = growth_valid["생중량"].mean()
                max_idx = growth_valid["생중량"].idxmax()
                min_idx = growth_valid["생중량"].idxmin()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("평균 생중량", f"{avg_weight:.2f}g")
                with col2:
                    st.metric("최대 개체", growth_valid.loc[max_idx, "개체번호"])
                with col3:
                    st.metric("최소 개체", growth_valid.loc[min_idx, "개체번호"])

                st.subheader("개체별 생육 비교")
                growth_items = [c for c in growth_items if c in growth_df.columns]
                if len(growth_items) == 0:
                    st.info("선택한 생육 측정 항목이 없어요.")
                else:
                    growth_grouped = growth_valid.groupby("개체번호", as_index=False)[growth_items].mean()
                    n = len(growth_items)
                    nrows = (n + 1) // 2
                    fig = make_subplots(
                        rows=nrows,
                        cols=2,
                        subplot_titles=growth_items,
                        vertical_spacing=0.12,
                        horizontal_spacing=0.1,
                    )
                    for i, item in enumerate(growth_items):
                        r = i // 2 + 1
                        c = i % 2 + 1
                        if item == "생중량":
                            q75 = growth_grouped["생중량"].quantile(0.75)
                            q25 = growth_grouped["생중량"].quantile(0.25)
                            marker_color = np.where(
                                growth_grouped["생중량"] >= q75,
                                "#ff6b6b",
                                np.where(growth_grouped["생중량"] <= q25, "#4d9fff", "#b0b0b0"),
                            )
                            fig.add_trace(
                                go.Bar(
                                    x=growth_grouped["개체번호"].astype(str),
                                    y=growth_grouped[item],
                                    marker_color=marker_color,
                                    name="생중량",
                                ),
                                row=r,
                                col=c,
                            )
                        else:
                            fig.add_trace(
                                go.Bar(
                                    x=growth_grouped["개체번호"].astype(str),
                                    y=growth_grouped[item],
                                    name=item,
                                ),
                                row=r,
                                col=c,
                            )
                    fig.update_layout(
                        height=280 * nrows + 40,
                        title_text="개체별 생육 비교 (생중량: 빨강=상위 25%, 파랑=하위 25%)",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, width="stretch")

                st.subheader("생중량 분포")
                fig = go.Figure()
                fig.add_trace(go.Box(y=growth_valid["생중량"], name="생중량", boxpoints="outliers"))
                fig.update_layout(title="개체별 생중량 분포", height=400)
                st.plotly_chart(fig, width="stretch")

                st.subheader("상관관계 분석")
                corr_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_valid.columns]
                if len(corr_cols) >= 2:
                    corr_matrix = growth_valid[corr_cols].dropna().corr()
                    st.dataframe(corr_matrix, width="stretch")

                st.subheader("생중량과 생육 지표 관계")
                if "잎 수" in growth_valid.columns and "생중량" in growth_valid.columns:
                    sub = growth_valid[["잎 수", "생중량"]].dropna()
                    if len(sub) >= 2:
                        x = sub["잎 수"]
                        y = sub["생중량"]
                        slope, intercept = np.polyfit(x, y, 1)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="개체"))
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=intercept + slope * x,
                                mode="lines",
                                name="추세선",
                                line=dict(color="red", dash="dash"),
                            )
                        )
                        fig.update_layout(title="잎 수 vs 생중량", height=400)
                        st.plotly_chart(fig, width="stretch")

                if "지상부 길이" in growth_valid.columns and "생중량" in growth_valid.columns:
                    sub = growth_valid[["지상부 길이", "생중량"]].dropna()
                    if len(sub) >= 2:
                        x = sub["지상부 길이"]
                        y = sub["생중량"]
                        slope, intercept = np.polyfit(x, y, 1)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="개체"))
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=intercept + slope * x,
                                mode="lines",
                                name="추세선",
                                line=dict(color="red", dash="dash"),
                            )
                        )
                        fig.update_layout(title="지상부 길이 vs 생중량", height=400)
                        st.plotly_chart(fig, width="stretch")

                st.subheader("상위 25% vs 하위 25% 개체 비교")
                if len(growth_grouped) >= 4:
                    q75 = growth_grouped["생중량"].quantile(0.75)
                    q25 = growth_grouped["생중량"].quantile(0.25)
                    top = growth_grouped[growth_grouped["생중량"] >= q75]
                    bottom = growth_grouped[growth_grouped["생중량"] <= q25]
                    col_top, col_mid, col_bottom = st.columns([1, 1, 1])
                    with col_top:
                        st.metric("상위 25% 평균 생중량", f"{top['생중량'].mean():.2f}g")
                    with col_mid:
                        st.metric("전체 평균 생중량", f"{growth_grouped['생중량'].mean():.2f}g")
                    with col_bottom:
                        st.metric("하위 25% 평균 생중량", f"{bottom['생중량'].mean():.2f}g")

                    rows = []
                    for _, r in top.iterrows():
                        rows.append([r["개체번호"], "상위 25%", r["생중량"], r["잎 수"], r["지상부 길이"]])
                    for _, r in bottom.iterrows():
                        rows.append([r["개체번호"], "하위 25%", r["생중량"], r["잎 수"], r["지상부 길이"]])
                    if rows:
                        compare_df = pd.DataFrame(rows, columns=["개체번호", "그룹", "생중량(g)", "잎 수", "지상부 길이(mm)"])
                        st.dataframe(compare_df, width="stretch")

                with st.expander("생육 데이터 원본 보기 및 다운로드"):
                    st.dataframe(growth_df, width="stretch")
                    csv_growth = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "생육 데이터 CSV 다운로드",
                        data=csv_growth,
                        file_name=f"{school_name}_생육결과데이터.csv" if school_name else "생육결과데이터.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
