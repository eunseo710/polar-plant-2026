import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="그린란드고추냉이 생육 분석", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)


def read_csv(file):
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


env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

env_df = pd.DataFrame()
growth_df = pd.DataFrame()

if env_file is not None:
    env_df = read_csv(env_file)
    env_df = std_cols(env_df)

if growth_file is not None:
    growth_df = read_csv(growth_file)
    growth_df = std_cols(growth_df)

growth_file_name = growth_file.name.split(".")[0] if growth_file is not None else ""
school_name = growth_file_name.replace("_", " ").replace("-", " ").strip()


def make_bar_fig(x, y, title, y_label, color_map=None):
    fig = go.Figure()
    if color_map is not None:
        for name, vals, cols in color_map:
            fig.add_trace(go.Bar(x=vals, y=cols, marker_color=name, name=name, hovertemplate="%{y} " + y_label + ": %{x}<extra></extra>"))
    else:
        fig.add_trace(go.Bar(x=y, y=x, marker_color="steelblue", hovertemplate="%{y} " + y_label + ": %{x}<extra></extra>"))
    fig.update_layout(
        title=title,
        xaxis_title="값",
        yaxis_title="",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        height=420,
    )
    return fig


def make_scatter_fig(x, y, title, x_label, y_label):
    fig = go.Figure()
    valid = ~(x.isna() | y.isna())
    if valid.sum() >= 2:
        m, b = np.polyfit(x[valid], y[valid], 1)
        fig.add_trace(go.Scatter(x=x[valid], y=y[valid], mode="markers", name="개체", hovertemplate="%{x} " + x_label + ", %{y} " + y_label + "<extra></extra>"))
        fig.add_trace(go.Scatter(x=x[valid], y=m * x[valid] + b, mode="lines", name="추세선", line=dict(color="red", dash="dash"), hovertemplate="추세선<extra></extra>"))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        height=420,
    )
    return fig


def make_maybe_bar(metric, group_a, group_b, color_a, color_b, label):
    label_ko = {"잎 수": "잎 수", "지상부 길이": "지상부 길이", "지하부 길이": "지하부 길이", "지상부 질량": "지상부 질량", "지하부 질량": "지하부 질량"}.get(metric, metric)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[color_a], y=[group_a], marker_color=color_a, name=f"상위 25%", hovertemplate=label_ko + ": %{x}<extra></extra>"))
    fig.add_trace(go.Bar(x=[color_b], y=[group_b], marker_color=color_b, name=f"하위 25%", hovertemplate=label_ko + ": %{x}<extra></extra>"))
    fig.update_layout(
        title=f"그린란드고추냉이 생육 그룹별 {label_ko} 비교",
        xaxis_title="그룹",
        yaxis_title=label_ko,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        height=420,
    )
    return fig


def safe_group_stats(df, value_col, group_col, n=25):
    vals = df[value_col].dropna()
    if vals.notna().sum() < 2:
        return None, None
    p75 = vals.quantile(0.75)
    p25 = vals.quantile(0.25)
    top = df[df[value_col] >= p75]
    bottom = df[df[value_col] <= p25]
    return top, bottom


def safe_corr(x, y):
    v = ~(x.isna() | y.isna())
    if v.sum() < 3:
        return None
    return float(np.corrcoef(x[v], y[v])[0, 1])


st.sidebar.title("그린란드고추냉이 생육 대시보드")
st.sidebar.info("환경·생육 CSV를 업로드하면 자동으로 열 이름을 표준화해요.")
if env_df.empty and growth_df.empty:
    st.sidebar.info("CSV 파일을 올려주세요.")

tab1, tab2, tab3 = st.tabs(["탐구의 배경", "생육 그룹별 특징 비교", "가설 검증과 최종 결론"])

with tab1:
    try:
        if growth_df.empty:
            st.info("생육 결과 데이터를 업로드하면 개체별 생중량을 비교할 수 있어요.")
        else:
            g = growth_df["생중량"].dropna()
            if g.notna().sum() < 2:
                st.info("생중량 데이터가 충분하지 않아요.")
            else:
                max_v = g.max()
                min_v = g.min()
                max_i = g.idxmax()
                min_i = g.idxmin()
                st.metric("데이터 출처", school_name or "업로드한 학교")
                st.metric("가장 무거운 개체", f"{max_v:g} g", f"개체번호 {max_i}")
                st.metric("가장 가벼운 개체", f"{min_v:g} g", f"개체번호 {min_i}")
                st.metric("개체 수", f"{len(g):,} 개", f"생중량 범위 {min_v:g}~{max_v:g} g")

                top1 = growth_df.loc[[max_i]]
                bottom1 = growth_df.loc[[min_i]]
                fig1 = make_bar_fig(
                    top1["생중량"].dropna(), top1["생중량"].dropna(),
                    "가장 무거운 개체", "생중량(g)",
                    color_map=[("상위", top1["생중량"].dropna(), "생중량"), ("하위", bottom1["생중량"].dropna(), "생중량")]
                )
                st.plotly_chart(fig1, width="stretch")

                fig1b = make_bar_fig(
                    ["가장 무거운 개체", "가장 가벼운 개체"],
                    [max_v, min_v],
                    "최댓값 vs 최솟값 비교", "생중량(g)",
                )
                st.plotly_chart(fig1b, width="stretch")

                st.markdown("### 탐구의 배경")
                st.markdown("- **탐구 대상**: 그린란드고추냉이 33개체")
                st.markdown("- 같은 스마트팜에서 같은 기간 재배했지만 개체별 생중량 차이가 발생했어요.")
                st.markdown("- 가장 무거운 개체: 25.82g / 가장 가벼운 개체: 8.40g")
                st.markdown("- 탐구 가설: 덜 자란 개체일수록 뿌리 질량 대비 길이가 클 것이다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab2:
    try:
        if growth_df.empty:
            st.info("생육 결과 데이터를 업로드하면 그룹을 비교할 수 있어요.")
        else:
            g = growth_df["생중량"].dropna()
            if g.notna().sum() < 2:
                st.info("생중량 데이터가 충분하지 않아요.")
            else:
                top, bottom = safe_group_stats(growth_df, "생중량", "생중량")
                if top is None or bottom is None or top.empty or bottom.empty:
                    st.info("상위·하위 25% 그룹을 나눌 만큼 데이터가 충분하지 않아요.")
                else:
                    leaf_top = top["잎 수"].dropna().mean()
                    leaf_bottom = bottom["잎 수"].dropna().mean()
                    rootlen_top = top["지상부 길이"].dropna().mean()
                    rootlen_bottom = bottom["지상부 길이"].dropna().mean()
                    rootmass_top = top["지하부 질량"].dropna().mean()
                    rootmass_bottom = bottom["지하부 질량"].dropna().mean()

                    st.metric("상위 25% 그룹 잎 수", f"{leaf_top:.2f}장" if pd.notna(leaf_top) else "-")
                    st.metric("하위 25% 그룹 잎 수", f"{leaf_bottom:.2f}장" if pd.notna(leaf_bottom) else "-")
                    st.metric("상위 25% 그룹 뿌리 길이", f"{rootlen_top:.2f}mm" if pd.notna(rootlen_top) else "-")
                    st.metric("하위 25% 그룹 뿌리 길이", f"{rootlen_bottom:.2f}mm" if pd.notna(rootlen_bottom) else "-")
                    st.metric("상위 25% 그룹 뿌리 질량", f"{rootmass_top:.2f}g" if pd.notna(rootmass_top) else "-")
                    st.metric("하위 25% 그룹 뿌리 질량", f"{rootmass_bottom:.2f}g" if pd.notna(rootmass_bottom) else "-")

                    fig2a = make_maybe_bar("잎 수", leaf_top, leaf_bottom, "seagreen", "salmon", "잎 수")
                    st.plotly_chart(fig2a, width="stretch")

                    fig2b = make_maybe_bar("지상부 길이", rootlen_top, rootlen_bottom, "seagreen", "salmon", "지상부 길이")
                    fig2c = make_maybe_bar("지하부 질량", rootmass_top, rootmass_bottom, "seagreen", "salmon", "지하부 질량")
                    cols = st.columns(2)
                    cols[0].plotly_chart(fig2b, width="stretch")
                    cols[1].plotly_chart(fig2c, width="stretch")

                    st.markdown("### 핵심 내용")
                    st.markdown("- 덜 자란 그룹(하위 25%)은 잎 수와 뿌리 질량이 적었어요.")
                    st.markdown("- 뿌리 길이는 두 그룹 사이에 큰 차이가 없었어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab3:
    try:
        if growth_df.empty:
            st.info("생육 결과 데이터를 업로드하면 가설 검증 결과를 볼 수 있어요.")
        else:
            g = growth_df["생중량"].dropna()
            if g.notna().sum() < 2:
                st.info("생중량 데이터가 충분하지 않아요.")
            else:
                top, bottom = safe_group_stats(growth_df, "생중량", "생중량")
                if top is None or bottom is None or top.empty or bottom.empty:
                    st.info("상위·하위 25% 그룹을 나눌 만큼 데이터가 충분하지 않아요.")
                else:
                    root_ratio_top = (top["지상부 길이"] / top["지하부 질량"]).dropna()
                    root_ratio_bottom = (bottom["지상부 길이"] / bottom["지하부 질량"]).dropna()

                    st.metric("상위 그룹 뿌리 질량 대비 길이", f"{root_ratio_top.mean():.2f}mm/g" if pd.notna(root_ratio_top.mean()) else "-")
                    st.metric("하위 그룹 뿌리 질량 대비 길이", f"{root_ratio_bottom.mean():.2f}mm/g" if pd.notna(root_ratio_bottom.mean()) else "-")

                    fig3a = make_maybe_bar("지상부 길이", root_ratio_top.mean(), root_ratio_bottom.mean(), "seagreen", "salmon", "뿌리 질량 대비 길이(mm/g)")
                    st.plotly_chart(fig3a, width="stretch")

                    if "생중량" in growth_df.columns and {"지상부 질량", "지하부 질량"} <= set(growth_df.columns):
                        growth_df["지상부 질량"] = pd.to_numeric(growth_df["지상부 질량"], errors="coerce")
                        growth_df["지하부 질량"] = pd.to_numeric(growth_df["지하부 질량"], errors="coerce")
                        root_ratio_all = (growth_df["지상부 길이"] / growth_df["지하부 질량"]).dropna()
                        common = growth_df[["생중량", "지상부 길이", "지하부 질량"]].dropna()
                        if common.shape[0] >= 3 and root_ratio_all.notna().sum() >= 3:
                            r = safe_corr(common["생중량"], root_ratio_all)
                            st.metric("생중량 vs 뿌리 질량 대비 길이 상관계수", f"r = {r:.3f}" if r is not None else "-")

                            fig3b = make_scatter_fig(common["생중량"], root_ratio_all, "생중량 vs 뿌리 질량 대비 길이", "생중량(g)", "뿌리 질량 대비 길이(mm/g)")
                            st.plotly_chart(fig3b, width="stretch")

                            st.markdown("### 최종 결론")
                            st.markdown("- 덜 자란 개체일수록 뿌리 질량 대비 길이가 크게 나타났어요.")
                            st.markdown("- 하지만 뿌리가 더 길어서가 아니라 뿌리 질량이 적어서 비율이 커진 것이었어요.")
                            st.markdown("- 덜 자란 개체의 주요 특징은 상대적으로 적은 잎 수와 뿌리 질량이었어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
