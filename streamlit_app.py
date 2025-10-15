
import io
import sys
import pandas as pd
import numpy as np
import streamlit as st

# ---- Optional backends: try matplotlib first, else use plotly ----
USE_MPL = True
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except Exception as e:
    USE_MPL = False
    import plotly.express as px
    import plotly.graph_objects as go

st.set_page_config(page_title="학생 성적 분포 분석", layout="wide")

st.title("모의고사 원점수 리딩 결과 학생 성적 분포 히스토그램")
st.caption("univ로 리딩한 csv 파일을 업로드하면 국어/수학 '유형'별, 영어, 한국사, 탐구 과목 점수 분포를 시각화합니다. (막대 높이가 높을수록 붉은색, 낮을수록 하늘색)")

if not USE_MPL:
    st.warning("matplotlib을 불러오지 못해 Plotly로 대체합니다. requirements.txt에 matplotlib가 포함되어 있는지 확인하세요.")

uploaded = st.file_uploader("CSV 파일을 업로드하세요 (.csv)", type=["csv"])
bin_width = st.sidebar.number_input("히스토그램 구간 폭(점)", min_value=1, max_value=50, value=5, step=1)
normalize = st.sidebar.checkbox("비율(%)로 보기", value=False)

def _read_csv(file):
    # 한국어 헤더/인코딩 대응
    raw = file.read()
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            buf = io.StringIO(raw.decode(enc))
            df = pd.read_csv(buf)
            return df
        except Exception:
            continue
    buf = io.StringIO(raw.decode(errors="ignore"))
    return pd.read_csv(buf)

def _coerce_numeric(series):
    # 쉼표, 공백 제거 후 숫자 변환
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.strip(), errors="coerce")

def _ensure_columns(df):
    """
    - H: 국어유형 (7), I: 국어점수 (8)
    - J: 수학유형 (9), K: 수학점수 (10)
    - L: 영어점수 (11), M: 한국사점수 (12)
    - N: 탐구1(점수/과목명) (13), O: 탐구2(점수/과목명) (14)
    """
    if df.shape[1] < 15:
        st.warning("열 개수가 부족합니다. 파일 형식을 확인해주세요.")
    cols = df.columns.tolist()
    def safe_col(idx, default_name):
        if idx < len(cols):
            return cols[idx]
        else:
            return default_name

    h_col = safe_col(7, "H_국어유형")
    i_col = safe_col(8, "I_국어점수")
    j_col = safe_col(9, "J_수학유형")
    k_col = safe_col(10, "K_수학점수")
    l_col = safe_col(11, "L_영어점수")
    m_col = safe_col(12, "M_한국사점수")
    n_col = safe_col(13, "N_탐구1(점수/과목명)")
    o_col = safe_col(14, "O_탐구2(점수/과목명)")

    keep_cols = [c for c in [h_col, i_col, j_col, k_col, l_col, m_col, n_col, o_col] if c in df.columns]
    view = df[keep_cols].copy()

    if i_col in view.columns:
        view[i_col] = _coerce_numeric(view[i_col])
    if k_col in view.columns:
        view[k_col] = _coerce_numeric(view[k_col])
    if l_col in view.columns:
        view[l_col] = _coerce_numeric(view[l_col])
    if m_col in view.columns:
        view[m_col] = _coerce_numeric(view[m_col])

    mapping = dict(h=h_col, i=i_col, j=j_col, k=k_col, l=l_col, m=m_col, n=n_col, o=o_col)
    return view, mapping

def split_inquiry_cols(df, n_col, o_col):
    """N/O: '점수/과목명' → [탐구점수, 탐구과목명]"""
    parts = []
    for col in [n_col, o_col]:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            s = s.replace(["", "nan", "None", "NaN"], np.nan).dropna()
            left = s.str.split("/", n=1, expand=True)
            if left is not None and left.shape[1] == 2:
                score = pd.to_numeric(left[0].str.replace(",", "").str.strip(), errors="coerce")
                subject = left[1].str.strip()
                part = pd.DataFrame({"탐구점수": score, "탐구과목명": subject})
                parts.append(part)
    if parts:
        res = pd.concat(parts, axis=0, ignore_index=True)
        res = res[(res["탐구점수"].notna()) & (res["탐구점수"] >= 0) & (res["탐구점수"] <= 100)]
        return res
    return pd.DataFrame(columns=["탐구점수", "탐구과목명"])

def _build_bins(series, binsize):
    series = pd.Series(series).dropna()
    if series.empty:
        return None, None, None
    min_v = max(0, np.floor(series.min()))
    max_v = min(100, np.ceil(series.max()))
    edges = np.arange(min_v, max_v + binsize, binsize)
    if len(edges) < 2:
        edges = np.linspace(min_v, max_v, 11)
    counts, bin_edges = np.histogram(series, bins=edges)
    if normalize:
        total = counts.sum() if counts.sum() > 0 else 1
        counts_vis = (counts / total) * 100.0
    else:
        counts_vis = counts
    return counts, counts_vis, bin_edges

def _colors_from_counts(counts):
    # 0~1 정규화 → skyblue(#87CEEB)→red(#FF0000)
    if counts is None or len(counts) == 0:
        return []
    mx = np.max(counts) if np.max(counts) > 0 else 1.0
    vals = (counts / mx).astype(float)
    if USE_MPL:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("skyblue_red", ["#87CEEB", "#FF0000"])
        return [cmap(v) for v in vals]
    else:
        # plotly는 rgba/hex 모두 허용
        import matplotlib
        try:
            # 만약 런타임에 matplotlib.colors를 쓸 수 있으면 동일 그라데이션 사용
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("skyblue_red", ["#87CEEB", "#FF0000"])
            return [matplotlib.colors.to_hex(cmap(v)) for v in vals]
        except Exception:
            # 간단한 보간
            def lerp(a, b, t): return int(a + (b - a) * t)
            colors = []
            for t in vals:
                r = lerp(135, 255, t)
                g = lerp(206, 0, t)
                b = lerp(235, 0, t)
                colors.append(f"rgb({r},{g},{b})")
            return colors

def plot_histogram(series, title, binsize):
    data = pd.Series(series).dropna()
    if data.empty:
        st.info(f"표시할 데이터가 없습니다: {title}")
        return
    counts, counts_vis, bin_edges = _build_bins(data, binsize)
    colors = _colors_from_counts(counts)

    if USE_MPL:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        width = np.diff(bin_edges)
        ax.bar(bin_edges[:-1], counts_vis, width=width, align='edge', edgecolor="black", color=colors)
        ax.set_title(title)
        ax.set_xlabel("점수")
        ax.set_ylabel("인원수" if not normalize else "비율(%)")
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        st.pyplot(fig)
    else:
        # Plotly: 막대를 수동 작성(각 bin 별 색 적용)
        lefts = bin_edges[:-1]
        rights = bin_edges[1:]
        fig = go.Figure()
        for i in range(len(lefts)):
            fig.add_trace(go.Bar(
                x=[lefts[i]],
                y=[counts_vis[i]],
                width=[rights[i] - lefts[i]],
                marker_color=colors[i],
                marker_line_color="black",
                marker_line_width=1,
                hovertemplate=f"{lefts[i]:.0f} ~ {rights[i]:.0f} 점<br>" + ("인원수: %{y}" if not normalize else "비율: %{y:.2f}%") + "<extra></extra>"
            ))
        fig.update_layout(
            title=title,
            xaxis_title="점수",
            yaxis_title=("인원수" if not normalize else "비율(%)"),
            barmode="overlay",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_histogram_by_group(df, score_col, group_col, title_prefix, binsize):
    if group_col not in df.columns or score_col not in df.columns:
        st.info(f"필요한 열이 없어 표시할 수 없습니다. ({group_col}, {score_col})")
        return
    types = [str(x) for x in df[group_col].dropna().unique() if str(x).strip() != ""]
    if not types:
        st.info(f"'{group_col}'에 유효한 유형 값이 없습니다.")
        return
    selected = st.multiselect(f"{title_prefix} 표시할 유형 선택", options=types, default=types)
    n = len(selected)
    if n == 0:
        st.info("선택된 유형이 없습니다.")
        return
    cols = st.columns(min(3, n))
    for i, t in enumerate(selected):
        sub = df[df[group_col].astype(str) == t][score_col]
        with cols[i % len(cols)]:
            plot_histogram(sub, f"{title_prefix} ({t})", binsize)

if uploaded is not None:
    df = _read_csv(uploaded)
    st.success(f"파일 로드 완료. 행 {df.shape[0]}개, 열 {df.shape[1]}개")
    view, mp = _ensure_columns(df)

    with st.expander("매핑 확인 (파일의 실제 열 이름)", expanded=False):
        st.write(pd.DataFrame({
            "설명": ["H:국어유형","I:국어점수","J:수학유형","K:수학점수","L:영어점수","M:한국사점수","N:탐구1(점수/과목명)","O:탐구2(점수/과목명)"],
            "열이름": [mp["h"], mp["i"], mp["j"], mp["k"], mp["l"], mp["m"], mp["n"], mp["o"]]
        }))

    st.subheader("국어 점수 (유형별)")
    plot_histogram_by_group(view, mp["i"], mp["h"], "국어 점수 분포", bin_width)

    st.subheader("수학 점수 (유형별)")
    plot_histogram_by_group(view, mp["k"], mp["j"], "수학 점수 분포", bin_width)

    st.subheader("영어 점수")
    if mp["l"] in view.columns:
        plot_histogram(view[mp["l"]], "영어 점수 분포", bin_width)
    else:
        st.info("영어 점수 열(L)이 보이지 않습니다.")

    st.subheader("한국사 점수")
    if mp["m"] in view.columns:
        plot_histogram(view[mp["m"]], "한국사 점수 분포", bin_width)
    else:
        st.info("한국사 점수 열(M)이 보이지 않습니다.")

    st.subheader("탐구 과목 점수")
    inquiry = split_inquiry_cols(view, mp["n"], mp["o"])
    if inquiry.empty:
        st.info("탐구(N/O) 열에서 유효한 '점수/과목명' 데이터를 찾지 못했습니다.")
    else:
        subjects = sorted([s for s in inquiry["탐구과목명"].dropna().unique() if str(s).strip() != ""])
        choice = st.selectbox("탐구 과목 선택", options=["전체"] + subjects, index=0)
        if choice == "전체":
            plot_histogram(inquiry["탐구점수"], "탐구(전체) 점수 분포", bin_width)
        else:
            plot_histogram(inquiry.loc[inquiry["탐구과목명"] == choice, "탐구점수"], f"탐구({choice}) 점수 분포", bin_width)
else:
    st.info("업로드 영역에서 CSV 파일을 선택해주세요.")

st.markdown("---")
st.caption("ⓘ 히스토그램 색상: 막대 높이가 높을수록 붉은색, 낮을수록 하늘색으로 표시됩니다. 점수 범위는 기본 0~100으로 가정합니다.")
