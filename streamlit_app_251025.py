
import io
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="학생 성적 분포 분석", layout="wide")

st.title("학생 성적 분포 히스토그램")
st.caption("파일을 업로드하면 국어/수학 '유형'별, 영어, 한국사, 탐구 과목 점수 분포를 시각화합니다. (막대 높이가 높을수록 붉은색, 낮을수록 하늘색)")

uploaded = st.file_uploader("CSV 파일을 업로드하세요 (.csv)", type=["csv"])
bin_width = st.sidebar.number_input("히스토그램 구간 폭(점)", min_value=1, max_value=50, value=5, step=1)
normalize = st.sidebar.checkbox("비율(%)로 보기", value=False)

def _read_csv(file):
    # 한국어 헤더/인코딩 대응
    # 업로더가 주는 객체는 바이너리. utf-8-sig 우선, 실패시 cp949 시도
    raw = file.read()
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            buf = io.StringIO(raw.decode(enc))
            df = pd.read_csv(buf)
            return df
        except Exception:
            continue
    # 마지막 수단: 판다스 기본 추정
    buf = io.StringIO(raw.decode(errors="ignore"))
    return pd.read_csv(buf)

def _coerce_numeric(series):
    # 쉼표, 공백 제거 후 숫자 변환
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.strip(), errors="coerce")

def _ensure_columns(df):
    """
    사용자의 설명을 기준으로 열을 위치로 매핑합니다.
    - H: 국어 유형, I: 국어 점수
    - J: 수학 유형, K: 수학 점수
    - L: 영어 점수, M: 한국사 점수
    - N, O: 탐구 (점수/과목명)
    실제 파일 헤더가 다를 수 있으므로, 우선 위치(0-index)로 가져오고,
    헤더 이름 또한 반환하여 화면에 표시합니다.
    """
    # 최소 15열(N=13, O=14 index) 가정
    if df.shape[1] < 15:
        st.warning("열 개수가 부족합니다. 파일 형식을 확인해주세요.")
    cols = df.columns.tolist()
    # 안전하게 인덱스 접근
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

    # 필요한 열만 추려서 사본 생성
    keep_cols = [c for c in [h_col, i_col, j_col, k_col, l_col, m_col, n_col, o_col] if c in df.columns]
    view = df[keep_cols].copy()

    # 숫자 강제 변환
    if i_col in view.columns:
        view[i_col] = _coerce_numeric(view[i_col])
    if k_col in view.columns:
        view[k_col] = _coerce_numeric(view[k_col])
    if l_col in view.columns:
        view[l_col] = _coerce_numeric(view[l_col])
    if m_col in view.columns:
        view[m_col] = _coerce_numeric(view[m_col])

    mapping = dict(
        h=h_col, i=i_col, j=j_col, k=k_col, l=l_col, m=m_col, n=n_col, o=o_col
    )
    return view, mapping

def split_inquiry_cols(df, n_col, o_col):
    """
    N, O 열은 '점수/과목명' 형태라고 가정. 공백을 허용적으로 처리.
    결과: long-form DataFrame [탐구과목명, 탐구점수]
    """
    parts = []
    for col in [n_col, o_col]:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            # 빈값/결측 제외
            s = s.replace(["", "nan", "None", "NaN"], np.nan).dropna()
            # "점수/과목명" -> 두 칼럼으로 분리
            left = s.str.split("/", n=1, expand=True)
            if left is not None and left.shape[1] == 2:
                score = pd.to_numeric(left[0].str.replace(",", "").str.strip(), errors="coerce")
                subject = left[1].str.strip()
                part = pd.DataFrame({"탐구점수": score, "탐구과목명": subject})
                parts.append(part)
    if parts:
        res = pd.concat(parts, axis=0, ignore_index=True)
        # 점수 유효 범위 필터(0~100 가정)
        res = res[(res["탐구점수"].notna()) & (res["탐구점수"] >= 0) & (res["탐구점수"] <= 100)]
        return res
    else:
        return pd.DataFrame(columns=["탐구점수", "탐구과목명"])

def gradient_colors(counts):
    """
    bin count 값을 0~1로 정규화하여 하늘색(#87CEEB) -> 빨강(#FF0000) 그라디언트로 색 지정.
    """
    if len(counts) == 0:
        return []
    max_c = counts.max() if np.max(counts) > 0 else 1.0
    vals = (counts / max_c).astype(float)

    cmap = LinearSegmentedColormap.from_list("skyblue_red", ["#87CEEB", "#FF0000"])
    return [cmap(v) for v in vals]

def plot_histogram(data, title, bins=10, value_as_percent=False):
    """
    data: 1차원 숫자 데이터(점수 시리즈)
    value_as_percent: True면 y축을 %로 변환
    """
    data = pd.Series(data).dropna()
    if data.empty:
        st.info(f"표시할 데이터가 없습니다: {title}")
        return

    # 구간 생성
    min_v = max(0, np.floor(data.min()))
    max_v = min(100, np.ceil(data.max()))
    # bin width 기준으로 edges 계산
    edges = np.arange(min_v, max_v + bins, bins)
    if len(edges) < 2:
        # fallback
        edges = np.linspace(min_v, max_v, 11)

    counts, bin_edges = np.histogram(data, bins=edges)
    if value_as_percent:
        total = counts.sum() if counts.sum() > 0 else 1
        counts_vis = (counts / total) * 100.0
    else:
        counts_vis = counts

    colors = gradient_colors(counts)
    fig, ax = plt.subplots()
    # 막대 그리기
    width = np.diff(bin_edges)
    ax.bar(bin_edges[:-1], counts_vis, width=width, align='edge', edgecolor="black", color=colors)
    ax.set_title(title)
    ax.set_xlabel("점수")
    ax.set_ylabel("인원수" if not value_as_percent else "비율(%)")
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    st.pyplot(fig)

def plot_histogram_by_group(df, score_col, group_col, title_prefix, bins, value_as_percent):
    """
    group_col(유형) 별로 필터 후 각각의 히스토그램을 그립니다.
    Streamlit에서 컬럼 레이아웃으로 나란히 배치.
    """
    if group_col not in df.columns or score_col not in df.columns:
        st.info(f"필요한 열이 없어 표시할 수 없습니다. ({group_col}, {score_col})")
        return

    types = [str(x) for x in df[group_col].dropna().unique() if str(x).strip() != ""]
    if not types:
        st.info(f"'{group_col}'에 유효한 유형 값이 없습니다.")
        return

    selected = st.multiselect(f"{title_prefix} 표시할 유형 선택", options=types, default=types)
    # 2~3개씩 가로 배치
    n = len(selected)
    if n == 0:
        st.info("선택된 유형이 없습니다.")
        return

    cols = st.columns(min(3, n))
    for i, t in enumerate(selected):
        sub = df[df[group_col].astype(str) == t][score_col]
        with cols[i % len(cols)]:
            plot_histogram(sub, f"{title_prefix} ({t})", bins=bins, value_as_percent=value_as_percent)

if uploaded is not None:
    df = _read_csv(uploaded)
    st.success(f"파일 로드 완료. 행 {df.shape[0]}개, 열 {df.shape[1]}개")
    view, mp = _ensure_columns(df)

    with st.expander("매핑 확인 (파일의 실제 열 이름)", expanded=False):
        st.write(pd.DataFrame({
            "설명": ["H:국어유형","I:국어점수","J:수학유형","K:수학점수","L:영어점수","M:한국사점수","N:탐구1(점수/과목명)","O:탐구2(점수/과목명)"],
            "열이름": [mp["h"], mp["i"], mp["j"], mp["k"], mp["l"], mp["m"], mp["n"], mp["o"]]
        }))

    # ===== 국어(유형별) =====
    st.subheader("국어 점수 (유형별)")
    plot_histogram_by_group(view, mp["i"], mp["h"], "국어 점수 분포", bins=bin_width, value_as_percent=normalize)

    # ===== 수학(유형별) =====
    st.subheader("수학 점수 (유형별)")
    plot_histogram_by_group(view, mp["k"], mp["j"], "수학 점수 분포", bins=bin_width, value_as_percent=normalize)

    # ===== 영어 =====
    st.subheader("영어 점수")
    if mp["l"] in view.columns:
        plot_histogram(view[mp["l"]], "영어 점수 분포", bins=bin_width, value_as_percent=normalize)
    else:
        st.info("영어 점수 열(L)이 보이지 않습니다.")

    # ===== 한국사 =====
    st.subheader("한국사 점수")
    if mp["m"] in view.columns:
        plot_histogram(view[mp["m"]], "한국사 점수 분포", bins=bin_width, value_as_percent=normalize)
    else:
        st.info("한국사 점수 열(M)이 보이지 않습니다.")

    # ===== 탐구 =====
    st.subheader("탐구 과목 점수")
    inquiry = split_inquiry_cols(view, mp["n"], mp["o"])
    if inquiry.empty:
        st.info("탐구(N/O) 열에서 유효한 '점수/과목명' 데이터를 찾지 못했습니다.")
    else:
        # 과목 선택
        subjects = sorted([s for s in inquiry["탐구과목명"].dropna().unique() if str(s).strip() != ""])
        choice = st.selectbox("탐구 과목 선택", options=["전체"] + subjects, index=0)
        if choice == "전체":
            plot_histogram(inquiry["탐구점수"], "탐구(전체) 점수 분포", bins=bin_width, value_as_percent=normalize)
        else:
            plot_histogram(inquiry.loc[inquiry["탐구과목명"] == choice, "탐구점수"], f"탐구({choice}) 점수 분포", bins=bin_width, value_as_percent=normalize)

else:
    st.info("업로드 영역에서 CSV 파일을 선택해주세요.")

st.markdown("---")
st.caption("ⓘ 히스토그램 색상: 막대 높이가 높을수록 붉은색, 낮을수록 하늘색으로 표시됩니다. 점수 범위는 기본 0~100으로 가정합니다.")
