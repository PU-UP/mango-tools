from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional, List, Iterable
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import io
import os
import time
from typing import List, Tuple
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

@dataclass
class LogEntry:
    time: datetime
    thread: Optional[int]
    level: Optional[str]
    tag: Optional[str]
    file: Optional[str]
    lineno: Optional[int]
    func: Optional[str]
    message: str
    event_code: Optional[str]
    primary: Optional[str]
    secondary: Optional[str]
    raw: str


TIMESTAMP_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,\.]\d{3,6})\s+(?P<rest>.*)$'
)
THREAD_LVL_RE = re.compile(
    r'^\[(?P<thread>\d+)\]\s+(?P<level>[IWE])\/.*?:\s+(?P<after>.*)$'
)
TAG_RE = re.compile(r'^(?P<tag>\[[A-Z]+\])\s+(?P<after>.*)$')
TS2_RE = re.compile(
    r'^(?P<ts2>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,\.]\d{3,6})\s+(?P<after>.*)$'
)
FILE_LINE_FUNC_RE = re.compile(
    r'^(?P<file>[\w\/\.\-]+):(?P<lineno>\d+)\s+\((?P<func>[^\)]+)\)\s+(?P<message>.*)$'
)
EVENT_RE = re.compile(
    r'\[(?P<code>SLAM_EVENT_(?P<primary>INT|EXT|ISM)(?:_(?P<secondary>[A-Z0-9]+))?)\]'
)


def _parse_time_flex(ts: str) -> datetime:
    ts = ts.replace(',', '.')
    if '.' in ts:
        base, frac = ts.split('.')
        frac = (frac + '000000')[:6]
        ts_norm = f'{base}.{frac}'
    else:
        ts_norm = ts
    return datetime.fromisoformat(ts_norm)


def parse_log_lines(lines: Iterable[str]) -> list[LogEntry]:
    entries: list[LogEntry] = []
    current: Optional[LogEntry] = None
    for raw_line in lines:
        line = raw_line.rstrip('\n')
        m_ts = TIMESTAMP_RE.match(line)
        if not m_ts:
            if current is not None:
                current.message += ('\n' + line)
            continue
        first_ts = _parse_time_flex(m_ts.group('ts'))
        rest = m_ts.group('rest')

        thread = None
        level = None
        tag = None
        file = None
        lineno = None
        func = None
        message = rest

        m_tl = THREAD_LVL_RE.match(rest)
        after = rest
        if m_tl:
            try:
                thread = int(m_tl.group('thread'))
            except Exception:
                thread = None
            level = m_tl.group('level')
            after = m_tl.group('after')

        m_tag = TAG_RE.match(after)
        if m_tag:
            tag = m_tag.group('tag').strip('[]')
            after = m_tag.group('after')

        m_ts2 = TS2_RE.match(after)
        if m_ts2:
            after = m_ts2.group('after')

        m_flf = FILE_LINE_FUNC_RE.match(after)
        if m_flf:
            file = m_flf.group('file')
            try:
                lineno = int(m_flf.group('lineno'))
            except Exception:
                lineno = None
            func = m_flf.group('func')
            message = m_flf.group('message')
        else:
            message = after

        event_code = None
        primary = None
        secondary = None
        m_ev = EVENT_RE.search(line)
        if m_ev:
            event_code = m_ev.group('code')
            primary = m_ev.group('primary')
            secondary = m_ev.group('secondary')

        current = LogEntry(
            time=first_ts,
            thread=thread,
            level=level,
            tag=tag,
            file=file,
            lineno=lineno,
            func=func,
            message=message,
            event_code=event_code,
            primary=primary,
            secondary=secondary,
            raw=line,
        )
        entries.append(current)
    return entries


def parse_log(path_or_str: str) -> pd.DataFrame:
    if '\n' in path_or_str or path_or_str.strip().startswith('202'):
        lines = path_or_str.splitlines(True)
    else:
        with open(path_or_str, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    entries = parse_log_lines(lines)
    df = pd.DataFrame([asdict(e) for e in entries])
    if not df.empty:
        df['time'] = pd.to_datetime(df['time'])
    return df


def extract_events(df: pd.DataFrame) -> pd.DataFrame:
    ev_df = df[df['event_code'].notna()].copy().reset_index(drop=True)
    ev_df['eid'] = ev_df.index
    def after_code(row):
        code = f'[{row["event_code"]}]'
        msg = row.get('message') or ''
        pos = msg.find(code)
        if pos >= 0:
            return msg[pos + len(code):].strip()
        return msg.strip()
    ev_df['msg_after_code'] = ev_df.apply(after_code, axis=1)
    return ev_df[['eid','time','event_code','primary','secondary','msg_after_code','file','func','tag','level','thread']]


def filter_events(ev_df: pd.DataFrame, primary: Optional[str]=None, secondary: Optional[str]=None, regex: Optional[str]=None) -> pd.DataFrame:
    out = ev_df
    if primary:
        out = out[out['primary'] == primary]
    if secondary:
        out = out[out['secondary'] == secondary]
    if regex:
        out = out[out['msg_after_code'].fillna('').str.contains(regex, regex=True)]
    return out.reset_index(drop=True)


def between_events(df: pd.DataFrame, from_eid: int, to_eid: int, pad_seconds: int=10) -> pd.DataFrame:
    ev = extract_events(df)
    if ev.empty:
        return df.iloc[0:0].copy()
    if from_eid < 0 or to_eid < 0 or from_eid >= len(ev) or to_eid >= len(ev):
        raise IndexError("from_eid/to_eid out of range.")
    t0 = ev.loc[min(from_eid, to_eid), 'time']
    t1 = ev.loc[max(from_eid, to_eid), 'time']
    start = t0 - pd.Timedelta(seconds=pad_seconds)
    end = t1 + pd.Timedelta(seconds=pad_seconds)
    return df[(df['time'] >= start) & (df['time'] <= end)].copy().reset_index(drop=True)


PRIMARY_TO_Y = {'INT':0, 'EXT':1, 'ISM':2}


def plot_timeline(ev_df: pd.DataFrame, title: str='SLAM Events Timeline (µs)'):
    if ev_df.empty:
        raise ValueError("No events to plot.")
    df = ev_df.copy()
    df['y'] = df['primary'].map(PRIMARY_TO_Y)
    fig, ax = plt.subplots(figsize=(11,3.2))
    ax.scatter(df['time'], df['y'])
    ax.set_yticks([0,1,2]); ax.set_yticklabels(['INT','EXT','ISM'])
    ax.set_xlabel('Time'); ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
    fig.autofmt_xdate(rotation=20)
    N = max(1, len(df)//20)
    for i, row in df.iloc[::N].iterrows():
        label = row['secondary'] or row['event_code']
        ax.annotate(label, (row['time'], row['y']), xytext=(3,5), textcoords='offset points', fontsize=8)
    plt.tight_layout()
    return fig, ax


def save_timeline_png(ev_df: pd.DataFrame, out_path: str, title: str='SLAM Events Timeline (µs)'):
    fig, ax = plot_timeline(ev_df, title=title)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def df_to_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def df_to_log(df: pd.DataFrame, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for _, r in df.iterrows():
            t_str = pd.Timestamp(r['time']).to_pydatetime().strftime('%Y-%m-%d %H:%M:%S.%f')
            f.write(f"{t_str} [{r.get('thread','')}] {r.get('level','')}: {r.get('tag','')} "
                    f"{(r.get('file') or '')}:{(r.get('lineno') or '')} ({(r.get('func') or '')}) "
                    f"{r.get('message') or ''}\n")


def extract_pose_from_log(df: pd.DataFrame, position_mode: str = 'enu') -> pd.DataFrame:
    """从日志数据中提取pose信息"""
    if position_mode == 'enu':
        # 匹配 location/pose/enu : (x, y, yaw, confidence, calibrate, slip, relocal_mode) x y yaw conf cal slip reloc
        pattern = re.compile(
            r"location/pose/enu\s*:\s*\([^)]+\)\s+([\-\d\.eE]+)\s+([\-\d\.eE]+)\s+([\-\d\.eE]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
        )
    else:
        # 匹配 location/pose : (x, y, yaw, confidence, calibrate, slip, relocal_mode) x y yaw conf cal slip reloc
        pattern = re.compile(
            r"location/pose\s*:\s*\([^)]+\)\s+([\-\d\.eE]+)\s+([\-\d\.eE]+)\s+([\-\d\.eE]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
        )
    
    poses = []
    for _, row in df.iterrows():
        message = row.get('message', '')
        match = pattern.search(message)
        if match:
            x, y, yaw, conf, cal, slip, reloc = match.groups()
            # 使用第一个时间戳（行首的时间戳）
            time_str = row['time'].strftime('%H:%M:%S.%f')[:-3]  # 保留毫秒
            poses.append({
                'time': row['time'],
                'time_str': time_str,
                'x': float(x),
                'y': float(y),
                'yaw': float(yaw),
                'conf': int(conf),
                'cal': int(cal),
                'slip': int(slip),
                'reloc': int(reloc)
            })
    
    return pd.DataFrame(poses)


def plot_trajectory(poses_df: pd.DataFrame, title: str = '轨迹图'):
    """绘制轨迹图"""
    if poses_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # 绘制轨迹线
    fig.add_trace(go.Scatter(
        x=poses_df['x'],
        y=poses_df['y'],
        mode='lines+markers',
        name='轨迹',
        line=dict(color='blue', width=2),
        marker=dict(size=4, color='blue'),
        hovertemplate=(
            "时间: %{customdata[0]}<br>"
            "X: %{x:.3f}<br>"
            "Y: %{y:.3f}<br>"
            "Yaw: %{customdata[1]:.3f}<br>"
            "Conf: %{customdata[2]}<br>"
            "Cal: %{customdata[3]}<br>"
            "Slip: %{customdata[4]}<br>"
            "Reloc: %{customdata[5]}<extra></extra>"
        ),
        customdata=poses_df[['time_str', 'yaw', 'conf', 'cal', 'slip', 'reloc']].values
    ))
    
    # 添加起始点
    if not poses_df.empty:
        fig.add_trace(go.Scatter(
            x=[poses_df.iloc[0]['x']],
            y=[poses_df.iloc[0]['y']],
            mode='markers',
            name='起点',
            marker=dict(size=10, color='green', symbol='circle'),
            hovertemplate="起点<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>"
        ))
        
        # 添加终点
        fig.add_trace(go.Scatter(
            x=[poses_df.iloc[-1]['x']],
            y=[poses_df.iloc[-1]['y']],
            mode='markers',
            name='终点',
            marker=dict(size=10, color='red', symbol='circle'),
            hovertemplate="终点<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>"
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='X坐标',
        yaxis_title='Y坐标',
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    # 设置等比例坐标轴
    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig


def find_event_ids(ev_df: pd.DataFrame, code_like: str) -> list[int]:
    m = ev_df[ev_df['event_code'].str.contains(code_like, na=False)]
    return m['eid'].tolist()



st.set_page_config(page_title="SLAM Log Dashboard", layout="wide")

# --- Sidebar: data source selection ---
st.sidebar.title("数据源")
mode = st.sidebar.radio("选择模式", ["上传文件（离线）", "本地路径（实时追踪）"])

def ensure_time_dtype(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    return df

@st.cache_data(show_spinner=False)
def parse_uploaded_text(text: str) -> pd.DataFrame:
    df = parse_log(text)
    return ensure_time_dtype(df)

def read_file_tail(path: str, offset: int) -> Tuple[str, int]:
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(offset)
            data = f.read()
            new_off = f.tell()
        return data, new_off
    except FileNotFoundError:
        return "", offset

# Session states
if "tail_df" not in st.session_state:
    st.session_state["tail_df"] = pd.DataFrame()
if "tail_offset" not in st.session_state:
    st.session_state["tail_offset"] = 0
if "tail_path" not in st.session_state:
    st.session_state["tail_path"] = ""

df = pd.DataFrame()

# --- File modes ---
if mode == "上传文件（离线）":
    up = st.sidebar.file_uploader("选择日志文件", type=["log", "txt"])
    if up is not None:
        text = up.read().decode('utf-8', errors='replace')
        df = parse_uploaded_text(text)
    else:
        st.info("请在左侧上传日志文件。")
else:
    path = st.sidebar.text_input("本地日志路径", value=st.session_state.get("tail_path", ""))
    auto = st.sidebar.toggle("自动刷新", value=True)
    interval = st.sidebar.number_input("间隔(s)", min_value=1, max_value=60, value=3)
    btn_refresh = st.sidebar.button("立即刷新")

    if path and path != st.session_state["tail_path"]:
        st.session_state["tail_path"] = path
        st.session_state["tail_offset"] = 0
        st.session_state["tail_df"] = pd.DataFrame()

    if path:
        if st.session_state["tail_df"].empty and os.path.exists(path):
            text, new_off = read_file_tail(path, 0)
            st.session_state["tail_df"] = parse_uploaded_text(text)
            st.session_state["tail_offset"] = new_off
        if (auto or btn_refresh) and os.path.exists(path):
            new_text, new_off = read_file_tail(path, st.session_state["tail_offset"])
            if new_off < st.session_state["tail_offset"]:
                text, new_off2 = read_file_tail(path, 0)
                st.session_state["tail_df"] = parse_uploaded_text(text)
                st.session_state["tail_offset"] = new_off2
            else:
                if new_text:
                    df_new = parse_uploaded_text(new_text)
                    st.session_state["tail_df"] = pd.concat([st.session_state["tail_df"], df_new], ignore_index=True).drop_duplicates(subset=['time','raw'])
                    st.session_state["tail_offset"] = new_off
            if auto:
                time.sleep(interval)
                st.experimental_rerun()
        df = st.session_state["tail_df"]
    else:
        st.info("请输入本地日志路径以启用实时追踪。")

if df is None or df.empty:
    st.stop()

# --- Extract & filter events ---
ev_all = extract_events(df)
primaries = ["INT","EXT","ISM"]
sel_primary = st.sidebar.multiselect("一级分类", primaries, default=primaries)
sec_values = sorted([s for s in ev_all['secondary'].dropna().unique().tolist()])
sel_secondary = st.sidebar.multiselect("二级分类", sec_values, default=[])
regex = st.sidebar.text_input("说明正则筛选（可选）", value="")

# --- Trajectory settings ---
st.sidebar.subheader("轨迹设置")
pose_mode = st.sidebar.radio("位置模式", ["enu", "base"], index=0)
show_trajectory = st.sidebar.checkbox("显示轨迹", value=False)

ev = ev_all.copy()
if sel_primary:
    ev = ev[ev['primary'].isin(sel_primary)]
if sel_secondary:
    ev = ev[ev['secondary'].isin(sel_secondary)]
if regex:
    ev = ev[ev['msg_after_code'].fillna('').str.contains(regex, regex=True)]

ev = ev.reset_index(drop=True)
ev['eid'] = ev.index

# --- Timeline plot: color by secondary ---

def plot_timeline(ev_df: pd.DataFrame):
    if ev_df.empty:
        return go.Figure()
    fig = go.Figure()
    colors = px.colors.qualitative.Safe
    uniques = [s for s in ev_df['secondary'].dropna().unique()]
    for i, sec in enumerate(uniques):
        sub = ev_df[ev_df['secondary'] == sec]
        fig.add_trace(go.Scattergl(
            x=sub['time'], y=sub['primary'],
            mode='markers', name=str(sec),
            marker=dict(size=8, color=colors[i % len(colors)]),
            # customdata: [secondary, msg_after_code]
            customdata=sub[['secondary','msg_after_code']].values,
            hovertemplate=(
                "时间:%{x|%H:%M:%S.%f}<br>"
                "一级:%{y}<br>"
                "二级:%{customdata[0]}<br>"
                "说明:%{customdata[1]}<extra></extra>"
            )
        ))
    fig.update_yaxes(type='category', categoryorder='array', categoryarray=['INT','EXT','ISM'], title="Primary")
    fig.update_xaxes(title="时间（µs）")
    fig.update_layout(height=350, margin=dict(l=20,r=20,t=40,b=20), title="事件时间线（按二级颜色区分）")
    return fig


st.subheader("事件时间线")
st.plotly_chart(plot_timeline(ev), use_container_width=True)

# --- Event table with time range slider ---
st.subheader("事件列表")
if not ev.empty:
    tmin, tmax = ev['time'].min(), ev['time'].max()
    start_time, end_time = st.slider(
        "显示时间范围", min_value=tmin.to_pydatetime(),
        max_value=tmax.to_pydatetime(),
        value=(tmin.to_pydatetime(), tmax.to_pydatetime()),
        step=timedelta(seconds=1),
        format="YYYY-MM-DD HH:mm:ss.SSS"
    )
    ev = ev[(ev['time'] >= start_time) & (ev['time'] <= end_time)]

# --- Trajectory display ---
if show_trajectory:
    st.subheader("轨迹图")
    
    # 提取pose数据
    poses_df = extract_pose_from_log(df, pose_mode)
    
    if not poses_df.empty:
        # 根据滑块选择的时间范围过滤pose数据
        poses_filtered = poses_df[(poses_df['time'] >= start_time) & (poses_df['time'] <= end_time)]
        
        if not poses_filtered.empty:
            # 显示轨迹统计信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("轨迹点数", len(poses_filtered))
            with col2:
                st.metric("时间范围", f"{poses_filtered['time'].min().strftime('%H:%M:%S.%f')[:-3]} - {poses_filtered['time'].max().strftime('%H:%M:%S.%f')[:-3]}")
            with col3:
                distance = ((poses_filtered['x'].diff() ** 2 + poses_filtered['y'].diff() ** 2) ** 0.5).sum()
                st.metric("总距离", f"{distance:.2f}m")
            
            # 绘制轨迹图
            fig = plot_trajectory(poses_filtered, f"轨迹图 ({pose_mode.upper()}模式)")
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示pose数据表格
            with st.expander("查看pose数据"):
                st.dataframe(poses_filtered[['time_str', 'x', 'y', 'yaw', 'conf', 'cal', 'slip', 'reloc']], use_container_width=True)
        else:
            st.warning(f"在选定时间范围内未找到{pose_mode}模式的pose数据")
    else:
        st.warning(f"未找到{pose_mode}模式的pose数据")

show_cols = ['time','event_code','msg_after_code']
if not ev.empty:
    # 创建显示用的DataFrame，将时间格式化为毫秒精度
    display_ev = ev[show_cols].copy()
    display_ev['time'] = display_ev['time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    
    # 配置列宽：让message列占据最大空间，前两列更紧凑
    column_config = {
        'time': st.column_config.TextColumn('时间', width=160),
        'event_code': st.column_config.TextColumn('事件代码', width=200),
        'msg_after_code': st.column_config.TextColumn('说明', width=2000)
    }
    st.dataframe(display_ev, use_container_width=True, column_config=column_config)
else:
    st.warning("未找到任何事件数据")

# --- Between two events ---
st.subheader("区间日志（含±冗余）")
if not ev.empty:
    eid_min, eid_max = int(ev['eid'].min()), int(ev['eid'].max())
    col1, col2, col3 = st.columns(3)
    from_id = col1.number_input("起始 eid", eid_min, eid_max, eid_min)
    to_id = col2.number_input("结束 eid", eid_min, eid_max, eid_max)
    pad_s = col3.number_input("冗余秒数(±)", 0, 300, 10, step=1)

    t0, t1 = ev.loc[min(from_id,to_id),'time'], ev.loc[max(from_id,to_id),'time']
    start = t0 - pd.Timedelta(seconds=pad_s)
    end = t1 + pd.Timedelta(seconds=pad_s)
    slice_df = df[(df['time']>=start)&(df['time']<=end)].copy()

    st.caption(f"区间日志行数：{len(slice_df)}")

    log_cols = ['time','tag','file','func','message']
    # 创建显示用的DataFrame，将时间格式化为毫秒精度
    display_slice_df = slice_df[log_cols].copy()
    display_slice_df['time'] = display_slice_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    
    # 配置列宽：让message列占据最大空间，其他列更紧凑
    column_config = {
        'time': st.column_config.TextColumn('时间', width=160),
        'tag': st.column_config.TextColumn('标签', width=80),
        'file': st.column_config.TextColumn('文件', width=200),
        'func': st.column_config.TextColumn('函数', width=160),
        'message': st.column_config.TextColumn('消息', width=2000)
    }
    st.dataframe(display_slice_df, use_container_width=True, height=600, column_config=column_config)

    # Download log text
    def df_to_log_text(df_in: pd.DataFrame) -> str:
        out_lines = []
        for _, r in df_in.iterrows():
            t_str = pd.Timestamp(r['time']).to_pydatetime().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            out_lines.append(f"{t_str} [{r.get('tag','')}] {r.get('file','')}:{r.get('func','')} {r.get('message') or ''}")
        return "\n".join(out_lines)

    log_text = df_to_log_text(slice_df).encode('utf-8')
    st.download_button("下载区间日志 (.log)", data=log_text, file_name="between_events.log", mime="text/plain")

st.success("✅ 已加载并可交互控制：颜色分级、时间筛选与区间日志导出。")
