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
from typing import List, Tuple, Dict, Any
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yaml

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


@dataclass
class TrajectoryInfo:
    id: int
    connected_ids: List[int]
    trajectory_type: int
    time: str
    initial_size: int
    sub_area: int
    good_trajectory: bool


@dataclass
class PositionInfo:
    position: Dict[str, float]
    rotation: Dict[str, float]
    timestamp: float
    time: str
    trajectory_id: int
    save_timestamp: float
    save_time: str


@dataclass
class MapState:
    trajectory_index: int
    trajectory_num: int
    map_version: str
    trajectory_source: str
    base_position: PositionInfo
    trajectories: Dict[str, TrajectoryInfo]
    tag_position: PositionInfo


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


def parse_trajectory_state(yaml_content: str) -> MapState:
    """解析trajectory_state.yaml文件内容"""
    try:
        data = yaml.safe_load(yaml_content)
        
        # 解析base_position
        base_pos_data = data.get('base_position', {})
        base_position = PositionInfo(
            position=base_pos_data.get('position', {}),
            rotation=base_pos_data.get('rotation', {}),
            timestamp=base_pos_data.get('timestamp', 0),
            time=base_pos_data.get('time', ''),
            trajectory_id=base_pos_data.get('trajectory_id', -1),
            save_timestamp=base_pos_data.get('save_timestamp', 0),
            save_time=base_pos_data.get('save_time', '')
        )
        
        # 解析tag_position
        tag_pos_data = data.get('tag_position', {})
        tag_position = PositionInfo(
            position=tag_pos_data.get('position', {}),
            rotation=tag_pos_data.get('rotation', {}),
            timestamp=tag_pos_data.get('timestamp', 0),
            time=tag_pos_data.get('time', ''),
            trajectory_id=tag_pos_data.get('trajectory_id', -1),
            save_timestamp=tag_pos_data.get('save_timestamp', 0),
            save_time=tag_pos_data.get('save_time', '')
        )
        
        # 解析trajectories
        trajectories = {}
        for key, value in data.items():
            if key.startswith('trajectory_') and isinstance(value, dict):
                trajectory_info = TrajectoryInfo(
                    id=value.get('id', 0),
                    connected_ids=value.get('connected_ids', []),
                    trajectory_type=value.get('trajectory_type', 0),
                    time=value.get('time', ''),
                    initial_size=value.get('initial_size', 0),
                    sub_area=value.get('sub_area', -1),
                    good_trajectory=value.get('good_trajectory', True)
                )
                trajectories[key] = trajectory_info
        
        return MapState(
            trajectory_index=data.get('trajectory_index', 0),
            trajectory_num=data.get('trajectory_num', 0),
            map_version=data.get('map_version', ''),
            trajectory_source=data.get('trajectory_source', ''),
            base_position=base_position,
            trajectories=trajectories,
            tag_position=tag_position
        )
    except Exception as e:
        st.error(f"解析trajectory_state.yaml文件失败: {str(e)}")
        raise


def parse_path_file(path_content: str) -> pd.DataFrame:
    """解析路径文件内容，格式为 x y z"""
    lines = path_content.strip().split('\n')
    data = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                yaw = float(parts[2]) if len(parts) > 2 else 0.0
                time_str = f"{i//3600:02d}:{(i%3600)//60:02d}:{i%60:02d}"
                data.append({
                    'x': x,
                    'y': y,
                    'yaw': yaw,
                    'time_str': time_str
                })
            except ValueError:
                continue
    return pd.DataFrame(data)


def plot_map_trajectories(map_state: MapState, trajectory_paths: Dict[str, pd.DataFrame], 
                         pose_mode: str = 'base') -> go.Figure:
    """绘制地图轨迹"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    # 绘制轨迹
    for i, (traj_key, traj_info) in enumerate(map_state.trajectories.items()):
        if traj_key in trajectory_paths and not trajectory_paths[traj_key].empty:
            path_df = trajectory_paths[traj_key]
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=path_df['x'],
                y=path_df['y'],
                mode='lines+markers',
                name=f'轨迹{traj_info.id} (类型{traj_info.trajectory_type})',
                line=dict(color=color, width=2),
                marker=dict(size=4, color=color),
                hovertemplate=(
                    f"轨迹{traj_info.id}<br>"
                    "时间: %{customdata[0]}<br>"
                    "X: %{x:.3f}<br>"
                    "Y: %{y:.3f}<br>"
                    "Yaw: %{customdata[1]:.3f}<extra></extra>"
                ),
                customdata=path_df[['time_str', 'yaw']].values
            ))
    
    # 如果是base模式，显示base和tag位置
    if pose_mode == 'base':
        # 显示base位置
        if map_state.base_position.position:
            fig.add_trace(go.Scatter(
                x=[map_state.base_position.position.get('x', 0)],
                y=[map_state.base_position.position.get('y', 0)],
                mode='markers',
                name='基站位置',
                marker=dict(size=15, color='red', symbol='square'),
                hovertemplate=(
                    "基站位置<br>"
                    f"X: {map_state.base_position.position.get('x', 0):.3f}<br>"
                    f"Y: {map_state.base_position.position.get('y', 0):.3f}<br>"
                    f"时间: {map_state.base_position.time}<extra></extra>"
                )
            ))
        
        # 显示tag位置
        if map_state.tag_position.position:
            fig.add_trace(go.Scatter(
                x=[map_state.tag_position.position.get('x', 0)],
                y=[map_state.tag_position.position.get('y', 0)],
                mode='markers',
                name='二维码位置',
                marker=dict(size=15, color='orange', symbol='diamond'),
                hovertemplate=(
                    "二维码位置<br>"
                    f"X: {map_state.tag_position.position.get('x', 0):.3f}<br>"
                    f"Y: {map_state.tag_position.position.get('y', 0):.3f}<br>"
                    f"时间: {map_state.tag_position.time}<extra></extra>"
                )
            ))
    
    fig.update_layout(
        title=f"地图轨迹图 ({pose_mode.upper()}模式)",
        xaxis_title='X坐标',
        yaxis_title='Y坐标',
        height=600,
        showlegend=True,
        hovermode='closest'
    )
    
    # 设置等比例坐标轴
    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig



st.set_page_config(page_title="SLAM Log Dashboard", layout="wide")

# --- Sidebar: data source selection ---
st.sidebar.title("数据源")
mode = st.sidebar.radio("选择模式", ["上传文件（离线）", "本地路径（实时追踪）"])
multi_log = st.sidebar.checkbox("多日志模式", value=False, help="启用后可以同时处理多个日志文件")

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
if "map_state" not in st.session_state:
    st.session_state["map_state"] = None
if "trajectory_paths" not in st.session_state:
    st.session_state["trajectory_paths"] = {}
if "show_map_info" not in st.session_state:
    st.session_state["show_map_info"] = False

df = pd.DataFrame()

# --- Map file upload and processing ---
st.sidebar.subheader("地图文件")
map_file = st.sidebar.file_uploader("上传地图文件", type=["yaml", "yml"], help="上传trajectory_state.yaml文件")

# 显示地图信息按钮
if "map_state" in st.session_state and st.session_state["map_state"] is not None:
    if st.sidebar.button("显示地图信息", help="查看地图详细信息并上传轨迹路径"):
        st.session_state["show_map_info"] = True

# 处理地图文件
if map_file is not None:
    try:
        map_content = map_file.read().decode('utf-8')
        map_state = parse_trajectory_state(map_content)
        st.session_state["map_state"] = map_state
        st.sidebar.success(f"✅ 地图文件已加载，包含 {len(map_state.trajectories)} 个轨迹")
    except Exception as e:
        st.sidebar.error(f"❌ 地图文件解析失败: {str(e)}")

# --- File modes ---
if mode == "上传文件（离线）":
    if multi_log:
        uploaded_files = st.sidebar.file_uploader("选择日志文件（直接多选有bug，请一个一个按时间顺序上传）先xxx.log.1后xxx.log", type=["log", "txt", "1"], accept_multiple_files=True)
        if uploaded_files:
            dfs = []
            for i, up in enumerate(uploaded_files):
                text = up.read().decode('utf-8', errors='replace')
                df_temp = parse_uploaded_text(text)
                df_temp['log_source'] = f"文件{i+1}: {up.name}"
                dfs.append(df_temp)
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            else:
                df = pd.DataFrame()
        else:
            st.info("请在左侧上传日志文件。")
            df = pd.DataFrame()
    else:
        up = st.sidebar.file_uploader("选择日志文件", type=["log", "txt", "1"])
        if up is not None:
            text = up.read().decode('utf-8', errors='replace')
            df = parse_uploaded_text(text)
            df['log_source'] = f"文件: {up.name}"
        else:
            st.info("请在左侧上传日志文件。")
            df = pd.DataFrame()
else:
    if multi_log:
        st.sidebar.subheader("多路径模式")
        num_paths = st.sidebar.number_input("路径数量", min_value=1, max_value=5, value=1)
        paths = []
        for i in range(num_paths):
            path = st.sidebar.text_input(f"本地日志路径 {i+1}", value=st.session_state.get(f"tail_path_{i}", ""), key=f"path_{i}")
            if path:
                paths.append(path)
        
        if paths:
            auto = st.sidebar.toggle("自动刷新", value=True)
            interval = st.sidebar.number_input("间隔(s)", min_value=1, max_value=60, value=3)
            btn_refresh = st.sidebar.button("立即刷新")
            
            dfs = []
            for i, path in enumerate(paths):
                if path != st.session_state.get(f"tail_path_{i}", ""):
                    st.session_state[f"tail_path_{i}"] = path
                    st.session_state[f"tail_offset_{i}"] = 0
                    st.session_state[f"tail_df_{i}"] = pd.DataFrame()
                
                if os.path.exists(path):
                    if st.session_state[f"tail_df_{i}"].empty:
                        text, new_off = read_file_tail(path, 0)
                        st.session_state[f"tail_df_{i}"] = parse_uploaded_text(text)
                        st.session_state[f"tail_offset_{i}"] = new_off
                    
                    if (auto or btn_refresh):
                        new_text, new_off = read_file_tail(path, st.session_state[f"tail_offset_{i}"])
                        if new_off < st.session_state[f"tail_offset_{i}"]:
                            text, new_off2 = read_file_tail(path, 0)
                            st.session_state[f"tail_df_{i}"] = parse_uploaded_text(text)
                            st.session_state[f"tail_offset_{i}"] = new_off2
                        else:
                            if new_text:
                                df_new = parse_uploaded_text(new_text)
                                st.session_state[f"tail_df_{i}"] = pd.concat([st.session_state[f"tail_df_{i}"], df_new], ignore_index=True).drop_duplicates(subset=['time','raw'])
                                st.session_state[f"tail_offset_{i}"] = new_off
                    
                    df_temp = st.session_state[f"tail_df_{i}"]
                    if not df_temp.empty:
                        df_temp['log_source'] = f"路径{i+1}: {os.path.basename(path)}"
                        dfs.append(df_temp)
            
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            else:
                df = pd.DataFrame()
            
            if auto:
                time.sleep(interval)
                st.rerun()
        else:
            st.info("请输入本地日志路径以启用实时追踪。")
            df = pd.DataFrame()
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
                    st.rerun()
            df = st.session_state["tail_df"]
            if not df.empty:
                df['log_source'] = f"路径: {os.path.basename(path)}"
        else:
            st.info("请输入本地日志路径以启用实时追踪。")
            df = pd.DataFrame()

# --- Map info page ---
if "show_map_info" in st.session_state and st.session_state["show_map_info"] and "map_state" in st.session_state and st.session_state["map_state"] is not None:
    st.title("地图信息")
    
    map_state = st.session_state["map_state"]
    
    # 显示地图基本信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("轨迹数量", map_state.trajectory_num)
    with col2:
        st.metric("当前轨迹索引", map_state.trajectory_index)
    with col3:
        st.metric("地图版本", map_state.map_version)
    with col4:
        st.metric("轨迹源", map_state.trajectory_source)
    
    # 显示基站和二维码位置信息
    st.subheader("位置信息")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**基站位置 (base坐标系)**")
        if map_state.base_position.position:
            st.write(f"- X: {map_state.base_position.position.get('x', 0):.6f}")
            st.write(f"- Y: {map_state.base_position.position.get('y', 0):.6f}")
            st.write(f"- Z: {map_state.base_position.position.get('z', 0):.6f}")
            st.write(f"- 时间: {map_state.base_position.time}")
        else:
            st.write("无基站位置信息")
    
    with col2:
        st.write("**二维码位置 (base坐标系)**")
        if map_state.tag_position.position:
            st.write(f"- X: {map_state.tag_position.position.get('x', 0):.6f}")
            st.write(f"- Y: {map_state.tag_position.position.get('y', 0):.6f}")
            st.write(f"- Z: {map_state.tag_position.position.get('z', 0):.6f}")
            st.write(f"- 时间: {map_state.tag_position.time}")
        else:
            st.write("无二维码位置信息")
    
    # 显示轨迹信息
    st.subheader("轨迹信息")
    for traj_key, traj_info in map_state.trajectories.items():
        with st.expander(f"轨迹 {traj_info.id} (类型 {traj_info.trajectory_type})"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**轨迹ID:** {traj_info.id}")
                st.write(f"**轨迹类型:** {traj_info.trajectory_type}")
                st.write(f"**初始大小:** {traj_info.initial_size}")
            with col2:
                st.write(f"**创建时间:** {traj_info.time}")
                st.write(f"**子区域:** {traj_info.sub_area}")
                st.write(f"**良好轨迹:** {'是' if traj_info.good_trajectory else '否'}")
            with col3:
                st.write(f"**连接轨迹:** {traj_info.connected_ids if traj_info.connected_ids else '无'}")
            
            # 轨迹路径上传
            st.write("**上传轨迹路径:**")
            col1, col2 = st.columns(2)
            
            with col1:
                enu_path_file = st.file_uploader(
                    f"ENU路径文件 - 轨迹{traj_info.id}", 
                    type=["txt", "csv", "log"],
                    key=f"enu_path_{traj_info.id}",
                    help="上传ENU坐标系下的轨迹路径文件"
                )
                if enu_path_file is not None:
                    try:
                        content = enu_path_file.read().decode('utf-8')
                        # 这里可以添加解析路径文件的逻辑
                        st.session_state["trajectory_paths"][f"{traj_key}_enu"] = content
                        st.success("ENU路径文件已上传")
                    except Exception as e:
                        st.error(f"ENU路径文件上传失败: {str(e)}")
            
            with col2:
                base_path_file = st.file_uploader(
                    f"Base路径文件 - 轨迹{traj_info.id}", 
                    type=["txt", "csv", "log"],
                    key=f"base_path_{traj_info.id}",
                    help="上传Base坐标系下的轨迹路径文件"
                )
                if base_path_file is not None:
                    try:
                        content = base_path_file.read().decode('utf-8')
                        # 这里可以添加解析路径文件的逻辑
                        st.session_state["trajectory_paths"][f"{traj_key}_base"] = content
                        st.success("Base路径文件已上传")
                    except Exception as e:
                        st.error(f"Base路径文件上传失败: {str(e)}")
    
    # 返回主页面按钮
    if st.button("返回主页面"):
        st.session_state["show_map_info"] = False
        st.rerun()
    
    st.stop()

if df is None or df.empty:
    st.stop()

# --- Extract & filter events ---
ev_all = extract_events(df)
primaries = ["INT","EXT","ISM"]
sel_primary = st.sidebar.multiselect("一级分类", primaries, default=primaries)
sec_values = sorted([s for s in ev_all['secondary'].dropna().unique().tolist()])
sel_secondary = st.sidebar.multiselect("二级分类", sec_values, default=[])
regex = st.sidebar.text_input("说明正则筛选（可选）", value="")

# --- Log source filter ---
if 'log_source' in df.columns and df['log_source'].nunique() > 1:
    st.sidebar.subheader("日志源筛选")
    log_sources = df['log_source'].unique().tolist()
    selected_sources = st.sidebar.multiselect("选择日志源", log_sources, default=log_sources)
    if selected_sources:
        df = df[df['log_source'].isin(selected_sources)]
    else:
        st.warning("请至少选择一个日志源")
        st.stop()

# --- Trajectory settings ---
st.sidebar.subheader("轨迹设置")
pose_mode = st.sidebar.radio("位置模式", ["enu", "base"], index=0)
show_trajectory = st.sidebar.checkbox("显示轨迹", value=False)
show_map = st.sidebar.checkbox("显示地图", value=False, help="显示地图轨迹（需要先上传地图文件）")

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

# --- Trajectory and Map display ---
if show_trajectory or show_map:
    st.subheader("轨迹图")
    
    # 创建组合图表
    fig = go.Figure()
    
    # 显示轨迹数据
    if show_trajectory:
        poses_df = extract_pose_from_log(df, pose_mode)
        
        if not poses_df.empty:
            # 根据滑块选择的时间范围过滤pose数据
            poses_filtered = poses_df[(poses_df['time'] >= start_time) & (poses_df['time'] <= end_time)]
            
            if not poses_filtered.empty:
                # 绘制轨迹线
                fig.add_trace(go.Scatter(
                    x=poses_filtered['x'],
                    y=poses_filtered['y'],
                    mode='lines+markers',
                    name='实时轨迹',
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
                    customdata=poses_filtered[['time_str', 'yaw', 'conf', 'cal', 'slip', 'reloc']].values
                ))
                
                # 添加起始点
                fig.add_trace(go.Scatter(
                    x=[poses_filtered.iloc[0]['x']],
                    y=[poses_filtered.iloc[0]['y']],
                    mode='markers',
                    name='轨迹起点',
                    marker=dict(size=10, color='green', symbol='circle'),
                    hovertemplate="轨迹起点<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>"
                ))
                
                # 添加终点
                fig.add_trace(go.Scatter(
                    x=[poses_filtered.iloc[-1]['x']],
                    y=[poses_filtered.iloc[-1]['y']],
                    mode='markers',
                    name='轨迹终点',
                    marker=dict(size=10, color='red', symbol='circle'),
                    hovertemplate="轨迹终点<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>"
                ))
    
    # 显示地图轨迹
    if show_map and "map_state" in st.session_state and st.session_state["map_state"] is not None:
        map_state = st.session_state["map_state"]
        
        # 检查是否有轨迹路径数据
        trajectory_paths = {}
        has_path_data = False
        
        for traj_key, traj_info in map_state.trajectories.items():
            enu_key = f"{traj_key}_enu"
            base_key = f"{traj_key}_base"
            
            if pose_mode == 'enu' and "trajectory_paths" in st.session_state and enu_key in st.session_state["trajectory_paths"]:
                try:
                    path_content = st.session_state["trajectory_paths"][enu_key]
                    trajectory_paths[traj_key] = parse_path_file(path_content)
                    has_path_data = True
                except Exception as e:
                    st.warning(f"解析轨迹{traj_info.id}的ENU路径数据失败: {str(e)}")
            
            elif pose_mode == 'base' and "trajectory_paths" in st.session_state and base_key in st.session_state["trajectory_paths"]:
                try:
                    path_content = st.session_state["trajectory_paths"][base_key]
                    trajectory_paths[traj_key] = parse_path_file(path_content)
                    has_path_data = True
                except Exception as e:
                    st.warning(f"解析轨迹{traj_info.id}的Base路径数据失败: {str(e)}")
        
        if has_path_data:
            colors = px.colors.qualitative.Set1
            
            # 绘制地图轨迹
            for i, (traj_key, traj_info) in enumerate(map_state.trajectories.items()):
                if traj_key in trajectory_paths and not trajectory_paths[traj_key].empty:
                    path_df = trajectory_paths[traj_key]
                    color = colors[i % len(colors)]
                    
                    fig.add_trace(go.Scatter(
                        x=path_df['x'],
                        y=path_df['y'],
                        mode='lines+markers',
                        name=f'地图轨迹{traj_info.id}',
                        line=dict(color=color, width=2),
                        marker=dict(size=3, color=color),
                        hovertemplate=(
                            f"地图轨迹{traj_info.id}<br>"
                            "时间: %{customdata[0]}<br>"
                            "X: %{x:.3f}<br>"
                            "Y: %{y:.3f}<br>"
                            "Yaw: %{customdata[1]:.3f}<extra></extra>"
                        ),
                        customdata=path_df[['time_str', 'yaw']].values
                    ))
            
            # 如果是base模式，显示base和tag位置
            if pose_mode == 'base':
                # 显示base位置
                if map_state.base_position.position:
                    fig.add_trace(go.Scatter(
                        x=[map_state.base_position.position.get('x', 0)],
                        y=[map_state.base_position.position.get('y', 0)],
                        mode='markers',
                        name='基站位置',
                        marker=dict(size=15, color='red', symbol='square'),
                        hovertemplate=(
                            "基站位置<br>"
                            f"X: {map_state.base_position.position.get('x', 0):.3f}<br>"
                            f"Y: {map_state.base_position.position.get('y', 0):.3f}<br>"
                            f"时间: {map_state.base_position.time}<extra></extra>"
                        )
                    ))
                
                # 显示tag位置
                if map_state.tag_position.position:
                    fig.add_trace(go.Scatter(
                        x=[map_state.tag_position.position.get('x', 0)],
                        y=[map_state.tag_position.position.get('y', 0)],
                        mode='markers',
                        name='二维码位置',
                        marker=dict(size=15, color='orange', symbol='diamond'),
                        hovertemplate=(
                            "二维码位置<br>"
                            f"X: {map_state.tag_position.position.get('x', 0):.3f}<br>"
                            f"Y: {map_state.tag_position.position.get('y', 0):.3f}<br>"
                            f"时间: {map_state.tag_position.time}<extra></extra>"
                        )
                    ))
    
    # 设置图表布局
    fig.update_layout(
        title=f"轨迹图 ({pose_mode.upper()}模式)",
        xaxis_title='X坐标',
        yaxis_title='Y坐标',
        height=600,
        showlegend=True,
        hovermode='closest'
    )
    
    # 设置等比例坐标轴
    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示统计信息
    if show_trajectory:
        poses_df = extract_pose_from_log(df, pose_mode)
        if not poses_df.empty:
            poses_filtered = poses_df[(poses_df['time'] >= start_time) & (poses_df['time'] <= end_time)]
            if not poses_filtered.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("实时轨迹点数", len(poses_filtered))
                with col2:
                    st.metric("时间范围", f"{poses_filtered['time'].min().strftime('%H:%M:%S.%f')[:-3]} - {poses_filtered['time'].max().strftime('%H:%M:%S.%f')[:-3]}")
                with col3:
                    distance = ((poses_filtered['x'].diff() ** 2 + poses_filtered['y'].diff() ** 2) ** 0.5).sum()
                    st.metric("总距离", f"{distance:.2f}m")
                
                # 显示pose数据表格
                with st.expander("查看pose数据"):
                    st.dataframe(poses_filtered[['time_str', 'x', 'y', 'yaw', 'conf', 'cal', 'slip', 'reloc']], use_container_width=True)
    
    if show_map and "map_state" in st.session_state and st.session_state["map_state"] is not None:
        map_state = st.session_state["map_state"]
        trajectory_paths = {}
        
        for traj_key, traj_info in map_state.trajectories.items():
            enu_key = f"{traj_key}_enu"
            base_key = f"{traj_key}_base"
            
            if pose_mode == 'enu' and "trajectory_paths" in st.session_state and enu_key in st.session_state["trajectory_paths"]:
                try:
                    path_content = st.session_state["trajectory_paths"][enu_key]
                    trajectory_paths[traj_key] = parse_path_file(path_content)
                except Exception:
                    pass
            
            elif pose_mode == 'base' and "trajectory_paths" in st.session_state and base_key in st.session_state["trajectory_paths"]:
                try:
                    path_content = st.session_state["trajectory_paths"][base_key]
                    trajectory_paths[traj_key] = parse_path_file(path_content)
                except Exception:
                    pass
        
        if trajectory_paths:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("地图轨迹数量", len(trajectory_paths))
            with col2:
                st.metric("坐标系", pose_mode.upper())
            with col3:
                total_points = sum(len(df) for df in trajectory_paths.values())
                st.metric("地图轨迹点数", total_points)


# 选择显示的列
if 'log_source' in ev.columns and ev['log_source'].nunique() > 1:
    show_cols = ['time','event_code','msg_after_code','log_source']
else:
    show_cols = ['time','event_code','msg_after_code']

if not ev.empty:
    # 创建显示用的DataFrame，将时间格式化为毫秒精度
    display_ev = ev[show_cols].copy()
    display_ev['time'] = display_ev['time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    
    # 配置列宽：让message列占据最大空间，前几列更紧凑
    if 'log_source' in show_cols:
        column_config = {
            'time': st.column_config.TextColumn('时间', width=160),
            'event_code': st.column_config.TextColumn('事件代码', width=200),
            'msg_after_code': st.column_config.TextColumn('说明', width=1500),
            'log_source': st.column_config.TextColumn('日志源', width=200)
        }
    else:
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

    # 选择显示的列
    if 'log_source' in slice_df.columns and slice_df['log_source'].nunique() > 1:
        log_cols = ['time','tag','file','func','message','log_source']
    else:
        log_cols = ['time','tag','file','func','message']
    
    # 创建显示用的DataFrame，将时间格式化为毫秒精度
    display_slice_df = slice_df[log_cols].copy()
    display_slice_df['time'] = display_slice_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    
    # 配置列宽：让message列占据最大空间，其他列更紧凑
    if 'log_source' in log_cols:
        column_config = {
            'time': st.column_config.TextColumn('时间', width=160),
            'tag': st.column_config.TextColumn('标签', width=80),
            'file': st.column_config.TextColumn('文件', width=200),
            'func': st.column_config.TextColumn('函数', width=160),
            'message': st.column_config.TextColumn('消息', width=1500),
            'log_source': st.column_config.TextColumn('日志源', width=200)
        }
    else:
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
