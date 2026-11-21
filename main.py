import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# 페이지 기본 설정
st.set_page_config(page_title="그린란드고추냉이의 생육 수준에 따른 뿌리 특성 비교", layout="wide")

# 한글 폰트 깨짐 방지 CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_school_name(uploaded_file) -> str:
    # 파일명 앞부분에서 학교명 읽기
    if uploaded_file is None or not getattr(uploaded_file, "name", ""):
        return ""
    base = os.path.splitext(str(uploaded_file.name).strip())[0]
    for sep in ("_환경", "_생육", "_"):
        if sep in base:
            prefix = base.split(sep, 1)[0].strip()
            if prefix:
                return prefix
    return base


@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    # utf-8-sig 우선, 실패 시 cp949로 읽기
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            if not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def read_csv(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    return read_csv_bytes(data) if data else pd.DataFrame()


@st.cache_data(show_spinner=False)
def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
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
    if "개체번호" in df.columns:
        df["개체번호"] = [
            np.nan if pd.isna(v) else (str(v).strip()[:-2] if str(v).strip().endswith(".0") and str(v).strip()[:-2].isdigit() else str(v).strip())
            for v in df["개체번호"]
        ]
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


def style_fig(fig, title: str = "", height: int = 400):
    # Plotly 공통 글꼴 및 레이아웃 적용 (width 인자는 st.plotly_chart에만 사용)
    fig.update_layout(
        title=title,
        height=height,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif", size=13),
        plot_bgcolor="rgba(248,250,252,0.85)",
        margin=dict(t=55, b=45, l=50, r=25),
    )
    return fig


def bar_chart(x, y, title: str, y_title: str, colors=None, texts=None, height: int = 400):
    # 재사용 막대그래프 생성 함수
    fig = go.Figure(
        go.Bar(
            x=x,
            y=y,
            marker_color=colors if colors is not None else "#3B7A57",
            text=texts if texts is not None else [f"{v:.2f}" for v in y],
            textposition="outside",
        )
    )
    fig.update_yaxes(title_text=y_title)
    return style_fig(fig, title=title, height=height)


@st.cache_data(show_spinner=False)
def prep_growth(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    g = df.copy()
    if "지하부 길이" in g.columns and "지하부 질량" in g.columns:
        ok = g["지하부 질량"].notna() & (g["지하부 질량"] > 0)
        g["뿌리 질량 대비 길이"] = np.where(ok, g["지하부 길이"] / g["지하부 질량"], np.nan)
    if "생중량" in g.columns:
        s = g["생중량"].dropna()
        if len(s) >= 4:
            q75, q25 = s.quantile(0.75), s.quantile(0.25)
            g["생육그룹"] = [
                "미분류" if pd.isna(v) else ("상위 25%" if v >= q75 else ("하위 25%" if v <= q25 else "중간 그룹"))
                for v in g["생중량"]
            ]
    return g


def group_mean_std(sub_df: pd.DataFrame, col: str):
    if col not in sub_df.columns:
        return np.nan, np.nan
    s = sub_df[col].dropna()
    if len(s) == 0:
        return np.nan, np.nan
    return float(s.mean()), float(s.std()) if len(s) >= 2 else 0.0


st.title("그린란드고추냉이의 생육 수준에 따른 뿌리 특성 비교")
st.caption("같은 환경에서 자란 극지식물 개체별 생육 차이와 뿌리 특성(길이·질량·비율)을 분석합니다.")

# 사이드바: CSV 2개 업로드 + 생육 측정 항목 선택
st.sidebar.header("📂 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 CSV", type="csv", key="env_csv")
growth_file = st.sidebar.file_uploader("생육 CSV", type="csv", key="growth_csv")

school_name = extract_school_name(growth_file) or extract_school_name(env_file) or "우리 학교"

with st.spinner("데이터를 불러오고 표준화하는 중입니다..."):
    env_df = std_cols(read_csv(env_file))
    growth_df = prep_growth(std_cols(read_csv(growth_file)))

if env_file is None and growth_file is None:
    st.info("왼쪽 사이드바에서 환경 CSV와 생육 CSV를 업로드하세요. 한 파일만 올려도 해당 영역이 동작합니다.")

st.sidebar.header("🔎 생육 측정 항목 선택")
metric_candidates = ["생중량", "잎 수", "지하부 길이", "지하부 질량", "뿌리 질량 대비 길이", "지상부 길이", "지상부 질량"]
avail_metrics = [c for c in metric_candidates if not growth_df.empty and c in growth_df.columns and growth_df[c].notna().any()]
default_metrics = [c for c in ["생중량", "잎 수", "지하부 길이", "지하부 질량", "뿌리 질량 대비 길이"] if c in avail_metrics]
selected_metrics = st.sidebar.multiselect("비교 항목 (개체번호 제외)", options=avail_metrics, default=default_metrics)

tab1, tab2, tab3 = st.tabs([
    "탭 1. 탐구의 배경 — 같은 환경인데 왜 다르게 자랐을까?",
    "탭 2. 생육 그룹별 특징 비교 — 덜 자란 개체는 무엇이 달랐을까?",
    "탭 3. 가설 검증과 최종 결론 — 처음 예상은 맞았을까?",
])

# 탭 1. 탐구의 배경
with tab1:
    try:
        st.header("탐구의 배경 — 같은 환경인데 왜 다르게 자랐을까?")
        st.markdown(
            f"""
            - **탐구 대상**: {school_name}에서 같은 스마트팜 환경·기간에 재배한 **그린란드고추냉이** 개체들
            - **탐구 배경**: 같은 조건에서 길렀지만 개체마다 생중량 차이가 발생함
            - **탐구 가설**: 덜 자란 개체일수록 뿌리 질량 대비 길이(mm/g)가 클 것이다.
            - **용어 설명**: **평균**은 중심값, **표준편차**는 값이 평균에서 흩어진 정도를 뜻합니다.
            """
        )
        if growth_df.empty or "생중량" not in growth_df.columns:
            st.info("생육 CSV를 업로드하면 개체별 생중량 비교와 요약 통계를 볼 수 있습니다.")
        else:
            valid_w = growth_df.dropna(subset=["생중량"]).copy()
            if len(valid_w) < 2:
                st.info("생중량 데이터가 2개 미만이어서 비교 통계를 계산할 수 없습니다.")
            else:
                max_i, min_i = valid_w["생중량"].idxmax(), valid_w["생중량"].idxmin()
                max_v, min_v = float(valid_w.loc[max_i, "생중량"]), float(valid_w.loc[min_i, "생중량"])
                mean_v, std_v = float(valid_w["생중량"].mean()), float(valid_w["생중량"].std())
                max_id = str(valid_w.loc[max_i, "개체번호"]) if "개체번호" in valid_w.columns else f"{max_i + 1}"
                min_id = str(valid_w.loc[min_i, "개체번호"]) if "개체번호" in valid_w.columns else f"{min_i + 1}"

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("개체 수", f"{len(valid_w)}개체")
                c2.metric("가장 무거운 개체", f"{max_v:.2f}g", f"개체 {max_id}")
                c3.metric("가장 가벼운 개체", f"{min_v:.2f}g", f"개체 {min_id}")
                c4.metric("최대·최소 비율", f"{max_v / min_v:.2f}배" if min_v > 0 else "계산 불가", f"평균 {mean_v:.2f}g · 표준편차 {std_v:.2f}g")

                ordered = valid_w.sort_values("생중량", ascending=False).reset_index(drop=True)
                x_labels = ordered["개체번호"].astype(str).tolist() if "개체번호" in ordered.columns else [f"개체{i+1}" for i in range(len(ordered))]
                bar_cols = ["#1E6F50" if i == 0 else ("#C85A32" if i == len(ordered) - 1 else "#7FA6C9") for i in range(len(ordered))]
                fig_all = bar_chart(x_labels, ordered["생중량"], "개체별 생중량 막대그래프 (초록: 최대, 주황: 최소)", "생중량 (g)", colors=bar_cols)
                fig_all.update_xaxes(type="category")
                st.plotly_chart(fig_all, width="stretch")

                fig_pair = bar_chart(
                    [f"가장 가벼운 개체 ({min_id})", f"가장 무거운 개체 ({max_id})"],
                    [min_v, max_v],
                    "가장 무거운 개체와 가장 가벼운 개체의 생중량 비교",
                    "생중량 (g)",
                    colors=["#C85A32", "#1E6F50"],
                    texts=[f"{min_v:.2f}g", f"{max_v:.2f}g"],
                    height=350,
                )
                st.plotly_chart(fig_pair, width="stretch")

                if selected_metrics:
                    rows = []
                    for col in selected_metrics:
                        s = growth_df[col].dropna()
                        if len(s) > 0:
                            rows.append({
                                "항목": col,
                                "개체수": int(len(s)),
                                "평균": round(float(s.mean()), 2),
                                "표준편차": round(float(s.std()), 2) if len(s) >= 2 else 0.0,
                                "최소": round(float(s.min()), 2),
                                "최대": round(float(s.max()), 2),
                            })
                    if rows:
                        sum_df = pd.DataFrame(rows)
                        st.dataframe(sum_df, width="stretch")
                        st.download_button("📥 생육 요약 CSV 다운로드", sum_df.to_csv(index=False).encode("utf-8-sig"), f"{school_name}_생육요약.csv", "text/csv")

        st.divider()
        st.subheader("재배 환경 데이터 요약")
        if env_df.empty:
            st.info("환경 CSV를 업로드하면 환경 그래프를 확인할 수 있습니다.")
        else:
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if not env_cols:
                st.info("환경 데이터에서 표준 환경 열(온도·습도·pH·EC)을 찾지 못했습니다.")
            elif "time" in env_df.columns and env_df["time"].notna().sum() >= 2:
                ep = env_df.dropna(subset=["time"]).sort_values("time")
                fig_e = make_subplots(rows=len(env_cols), cols=1, shared_xaxes=True, subplot_titles=env_cols)
                for idx_e, col_e in enumerate(env_cols, start=1):
                    sc_e = ep.dropna(subset=[col_e])
                    if len(sc_e) > 0:
                        fig_e.add_trace(go.Scatter(x=sc_e["time"], y=sc_e[col_e], mode="lines", name=col_e), row=idx_e, col=1)
                st.plotly_chart(style_fig(fig_e, "재배 환경 시계열 변화", height=200 * len(env_cols)), width="stretch")
            else:
                st.info("시간 열 정보가 부족해 평균값만 표시합니다.")
            means_dict = {c: round(float(env_df[c].dropna().mean()), 2) for c in env_cols if env_df[c].notna().any()}
            if means_dict:
                st.write("환경 평균:", means_dict)
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# 탭 2. 생육 그룹별 특징 비교
with tab2:
    try:
        st.header("생육 그룹별 특징 비교 — 덜 자란 개체는 무엇이 달랐을까?")
        st.markdown("- 생중량 **상위 25%** 그룹과 **하위 25%** 그룹의 평균을 비교합니다.")
        if growth_df.empty or "생육그룹" not in growth_df.columns:
            st.info("생육 CSV를 업로드하면 상위 25%와 하위 25% 그룹을 비교할 수 있습니다.")
        else:
            top_g = growth_df[growth_df["생육그룹"] == "상위 25%"]
            bot_g = growth_df[growth_df["생육그룹"] == "하위 25%"]
            if len(top_g) == 0 or len(bot_g) == 0:
                st.info("상위·하위 그룹을 나눌 개체 수가 부족합니다.")
            else:
                lt_m, _ = group_mean_std(top_g, "잎 수")
                lb_m, _ = group_mean_std(bot_g, "잎 수")
                rt_m, _ = group_mean_std(top_g, "지하부 길이")
                rb_m, _ = group_mean_std(bot_g, "지하부 길이")
                mt_m, _ = group_mean_std(top_g, "지하부 질량")
                mb_m, _ = group_mean_std(bot_g, "지하부 질량")

                m1, m2, m3 = st.columns(3)
                m1.metric("잎 수 (상위 / 하위)", f"{lt_m:.2f}장 / {lb_m:.2f}장" if pd.notna(lt_m) and pd.notna(lb_m) else "없음")
                m2.metric("뿌리 길이 (상위 / 하위)", f"{rt_m:.2f}mm / {rb_m:.2f}mm" if pd.notna(rt_m) and pd.notna(rb_m) else "없음")
                m3.metric("뿌리 질량 (상위 / 하위)", f"{mt_m:.2f}g / {mb_m:.2f}g" if pd.notna(mt_m) and pd.notna(mb_m) else "없음")

                if pd.notna(lt_m) and pd.notna(lb_m):
                    st.plotly_chart(
                        bar_chart(["하위 25%", "상위 25%"], [lb_m, lt_m], "상위·하위 그룹의 잎 수 비교", "평균 잎 수 (장)", colors=["#5C80BC", "#2E8B57"], texts=[f"{lb_m:.2f}장", f"{lt_m:.2f}장"]),
                        width="stretch",
                    )
                else:
                    st.info("잎 수 열이 없어 잎 수 비교 그래프를 건너뜁니다.")

                if pd.notna(rt_m) and pd.notna(rb_m) and pd.notna(mt_m) and pd.notna(mb_m):
                    fig_rm = make_subplots(rows=1, cols=2, subplot_titles=("뿌리 길이 평균 (mm)", "뿌리 질량 평균 (g)"))
                    fig_rm.add_trace(go.Bar(x=["하위 25%", "상위 25%"], y=[rb_m, rt_m], marker_color=["#5C80BC", "#2E8B57"], text=[f"{rb_m:.2f}mm", f"{rt_m:.2f}mm"], textposition="outside"), row=1, col=1)
                    fig_rm.add_trace(go.Bar(x=["하위 25%", "상위 25%"], y=[mb_m, mt_m], marker_color=["#5C80BC", "#2E8B57"], text=[f"{mb_m:.2f}g", f"{mt_m:.2f}g"], textposition="outside"), row=1, col=2)
                    fig_rm.update_layout(showlegend=False)
                    st.plotly_chart(style_fig(fig_rm, "상위·하위 그룹의 뿌리 길이와 뿌리 질량 비교"), width="stretch")
                else:
                    st.info("뿌리 길이 또는 뿌리 질량 열이 없어 뿌리 비교 그래프를 건너뜁니다.")

                st.markdown("- **핵심 내용**: 덜 자란 그룹은 잎 수와 뿌리 질량이 적었고, 뿌리 길이는 두 그룹 간 큰 차이가 없었습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# 탭 3. 가설 검증과 최종 결론
with tab3:
    try:
        st.header("가설 검증과 최종 결론 — 처음 예상은 맞았을까?")
        st.markdown("- **상관계수(r)**는 -1~+1 범위로 두 변수의 선형 관계 방향과 세기를 나타냅니다. 음수(-)는 한쪽이 클수록 다른 쪽이 작아지는 경향입니다.")
        if growth_df.empty or "생육그룹" not in growth_df.columns or "뿌리 질량 대비 길이" not in growth_df.columns:
            st.info("생육 CSV를 업로드하면 뿌리 질량 대비 길이와 상관계수를 확인할 수 있습니다.")
        else:
            top_g = growth_df[growth_df["생육그룹"] == "상위 25%"]
            bot_g = growth_df[growth_df["생육그룹"] == "하위 25%"]
            tr_s, br_s = top_g["뿌리 질량 대비 길이"].dropna(), bot_g["뿌리 질량 대비 길이"].dropna()
            if len(tr_s) == 0 or len(br_s) == 0:
                st.info("상위·하위 그룹의 뿌리 질량 대비 길이 데이터가 부족합니다.")
            else:
                tr_m, br_m = float(tr_s.mean()), float(br_s.mean())
                pair_df = growth_df[["생중량", "뿌리 질량 대비 길이"]].dropna() if "생중량" in growth_df.columns else pd.DataFrame()
                r_val = float(pair_df["생중량"].corr(pair_df["뿌리 질량 대비 길이"])) if len(pair_df) >= 2 and pair_df["생중량"].nunique() >= 2 else np.nan

                k1, k2, k3 = st.columns(3)
                k1.metric("상위 그룹 뿌리 질량 대비 길이", f"{tr_m:.2f} mm/g")
                k2.metric("하위 그룹 뿌리 질량 대비 길이", f"{br_m:.2f} mm/g", f"{br_m / tr_m:.2f}배" if tr_m > 0 else None)
                k3.metric("상관계수 (r)", f"{r_val:.3f}" if pd.notna(r_val) else "계산 불가", "가설 수치상 지지" if pd.notna(r_val) and r_val < 0 else None)

                st.plotly_chart(
                    bar_chart(["하위 25%", "상위 25%"], [br_m, tr_m], "상위·하위 그룹의 뿌리 질량 대비 길이 비교", "뿌리 질량 대비 길이 (mm/g)", colors=["#5C80BC", "#2E8B57"], texts=[f"{br_m:.2f}mm/g", f"{tr_m:.2f}mm/g"]),
                    width="stretch",
                )

                if len(pair_df) >= 2 and pair_df["생중량"].nunique() >= 2:
                    x_v = pair_df["생중량"].to_numpy(dtype=float)
                    y_v = pair_df["뿌리 질량 대비 길이"].to_numpy(dtype=float)
                    slope, intercept = np.polyfit(x_v, y_v, 1)
                    x_fit = np.linspace(float(x_v.min()), float(x_v.max()), 100)
                    fig_sc = go.Figure()
                    fig_sc.add_trace(go.Scatter(x=x_v, y=y_v, mode="markers", name="개체", marker=dict(size=9, color="#2B5B84")))
                    fig_sc.add_trace(go.Scatter(x=x_fit, y=slope * x_fit + intercept, mode="lines", name=f"추세선 (r={r_val:.3f})" if pd.notna(r_val) else "추세선", line=dict(color="#D95F02", width=2.5)))
                    fig_sc.update_xaxes(title_text="생중량 (g)")
                    fig_sc.update_yaxes(title_text="뿌리 질량 대비 길이 (mm/g)")
                    st.plotly_chart(style_fig(fig_sc, "생중량과 뿌리 질량 대비 길이의 관계 산점도"), width="stretch")
                else:
                    st.info("산점도를 그릴 유효 데이터가 2개 이상 필요합니다.")

                st.markdown(
                    """
                    - **최종 결론**: 덜 자란 개체일수록 뿌리 질량 대비 길이가 컸지만, 뿌리가 길어서가 아니라 **뿌리 질량이 적어서** 비율이 높아진 것이며, 덜 자란 개체는 **잎 수와 뿌리 질량이 적다**는 특징을 보였습니다.
                    """
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
