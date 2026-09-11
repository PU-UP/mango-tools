# server_gui.py
# 双击启动的 HTTP 文件共享界面：显示本机 IP，选择要共享的文件夹。
# 复用 server.py 的上传/列目录实现，仅使用标准库。

import functools
import json
import os
import socket
import sys
import threading
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    def _fatal(msg):
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, msg, "HTTP 文件共享", 0x10)
        except Exception:
            pass
        print(msg, file=sys.stderr)
        sys.exit(1)

    _fatal("未找到 Tkinter。请安装带 Tcl/Tk 的 Python 3。")

import server

SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_CONFIG_PATH = SCRIPT_DIR / "server_gui.json"
DEFAULT_PORT = 8000
LOG_MAX_LINES = 400


def config_path():
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / "mango-tools" / "http_com" / "server_gui.json"


CONFIG_PATH = config_path()


def _show_crash(exc_type, exc, tb):
    import traceback
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    try:
        messagebox.showerror("程序出错", text)
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = _show_crash


class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def list_local_ipv4():
    """本机 IPv4：局域网地址在前，回环地址在后。"""
    seen = set()
    ips = []

    def add(ip):
        if not ip or ip in seen or ip == "0.0.0.0" or ip.startswith("169.254."):
            return
        seen.add(ip)
        ips.append(ip)

    add(server.get_local_ip())
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass

    lan = [ip for ip in ips if not ip.startswith("127.")]
    loopback = [ip for ip in ips if ip.startswith("127.")]
    if "127.0.0.1" not in loopback:
        loopback.append("127.0.0.1")
    return lan + loopback


def _read_config(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, ValueError, TypeError):
        pass
    return {}


def load_config():
    data = _read_config(CONFIG_PATH)
    if data:
        return data
    return _read_config(LEGACY_CONFIG_PATH)


def save_config(folder, port):
    payload = json.dumps({"folder": folder, "port": port}, ensure_ascii=False, indent=2)
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(payload, encoding="utf-8")
        if LEGACY_CONFIG_PATH.exists() and LEGACY_CONFIG_PATH.resolve() != CONFIG_PATH.resolve():
            LEGACY_CONFIG_PATH.unlink()
    except OSError:
        pass


def default_folder(saved):
    if saved:
        path = Path(saved)
        if path.is_dir():
            return str(path.resolve())
    for candidate in (Path.home() / "Desktop", Path.home()):
        if candidate.is_dir():
            return str(candidate)
    return str(Path.home())


def path_in_git_worktree(path):
    root = SCRIPT_DIR.parent
    if not (root / ".git").exists():
        return False
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def make_handler(directory, ui, log_cb):
    class Handler(server.UploadHandler):
        def log_message(self, fmt, *args):
            try:
                message = "%s - %s" % (self.address_string(), fmt % args)
                ui.after(0, lambda m=message: log_cb(m))
            except Exception:
                pass

    return functools.partial(Handler, directory=directory)


class App(tk.Tk):
    def __init__(self):
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        super().__init__()
        self.title("HTTP 文件共享服务")
        self.geometry("560x520")
        self.minsize(500, 440)

        cfg = load_config()
        try:
            port = int(cfg.get("port", DEFAULT_PORT))
        except (TypeError, ValueError):
            port = DEFAULT_PORT
        if not 1 <= port <= 65535:
            port = DEFAULT_PORT

        self.folder = tk.StringVar(value=default_folder(cfg.get("folder")))
        self.port = tk.StringVar(value=str(port))
        self.status = tk.StringVar(value="未启动")
        self.ip_text = tk.StringVar(value="")
        self.url_text = tk.StringVar(value="")

        self.httpd = None
        self.server_thread = None
        self.urls = []

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.port.trace_add("write", lambda *_: self.refresh_addresses())
        self.refresh_addresses()
        self.append_log("选择要共享的文件夹，然后点击“启动服务”。")
        self.append_log("对方用浏览器打开访问地址即可浏览、上传和下载。")

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        info = ttk.LabelFrame(root, text="本机地址", padding=8)
        info.pack(fill=tk.X)
        ip_row = ttk.Frame(info)
        ip_row.pack(fill=tk.X)
        ttk.Label(ip_row, text="当前 IP：").pack(side=tk.LEFT)
        ttk.Label(ip_row, textvariable=self.ip_text, font=("Segoe UI", 11, "bold")).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(ip_row, text="刷新", command=self.refresh_addresses).pack(side=tk.RIGHT)

        ttk.Label(info, text="访问地址：").pack(anchor="w", pady=(8, 0))
        url_row = ttk.Frame(info)
        url_row.pack(fill=tk.X, pady=(2, 0))
        self.url_entry = ttk.Entry(url_row, textvariable=self.url_text, state="readonly")
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(url_row, text="复制", command=self.copy_url).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(url_row, text="本机打开", command=self.open_in_browser).pack(side=tk.LEFT, padx=(6, 0))

        self.extra_urls = tk.Text(info, height=3, wrap="none", relief="flat", background=self.cget("bg"))
        self.extra_urls.pack(fill=tk.X, pady=(6, 0))
        self.extra_urls.configure(state="disabled")

        share = ttk.LabelFrame(root, text="共享内容", padding=8)
        share.pack(fill=tk.X, pady=(10, 0))
        folder_row = ttk.Frame(share)
        folder_row.pack(fill=tk.X)
        ttk.Label(folder_row, text="文件夹：").pack(side=tk.LEFT)
        self.folder_entry = ttk.Entry(folder_row, textvariable=self.folder)
        self.folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.btn_browse = ttk.Button(folder_row, text="浏览…", command=self.choose_folder)
        self.btn_browse.pack(side=tk.LEFT)
        self.btn_open_folder = ttk.Button(folder_row, text="打开", command=self.open_folder)
        self.btn_open_folder.pack(side=tk.LEFT, padx=(6, 0))

        port_row = ttk.Frame(share)
        port_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(port_row, text="端口：").pack(side=tk.LEFT)
        self.port_entry = ttk.Entry(port_row, textvariable=self.port, width=8)
        self.port_entry.pack(side=tk.LEFT, padx=6)
        ttk.Label(port_row, textvariable=self.status).pack(side=tk.LEFT, padx=(12, 0))

        actions = ttk.Frame(root)
        actions.pack(fill=tk.X, pady=(10, 0))
        self.btn_start = ttk.Button(actions, text="启动服务", command=self.start_server)
        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop = ttk.Button(actions, text="停止服务", command=self.stop_server, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(root, text="首次启动若弹出防火墙提示，请允许专用网络访问。").pack(
            anchor="w", pady=(6, 0))

        log_frame = ttk.LabelFrame(root, text="运行日志", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.log = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def refresh_addresses(self):
        try:
            port = int(str(self.port.get()).strip() or DEFAULT_PORT)
        except ValueError:
            port = DEFAULT_PORT
        ips = list_local_ipv4()
        lan = [ip for ip in ips if not ip.startswith("127.")]
        primary = lan[0] if lan else ips[0]
        self.ip_text.set("  /  ".join(lan) if lan else primary)
        self.urls = [f"http://{ip}:{port}/" for ip in ips]
        self.url_text.set(self.urls[0] if self.urls else "")
        extra = []
        if len(self.urls) > 1:
            extra.append("其他地址：")
            extra.extend(self.urls[1:])
        if not lan:
            extra.append("未检测到局域网 IP，对方可能无法访问。")
        self.extra_urls.configure(state="normal")
        self.extra_urls.delete("1.0", tk.END)
        self.extra_urls.insert("1.0", "\n".join(extra))
        self.extra_urls.configure(state="disabled")

    def choose_folder(self):
        initial = self.folder.get().strip() or default_folder(None)
        picked = filedialog.askdirectory(title="选择要让对方访问的文件夹", initialdir=initial)
        if picked:
            self.folder.set(str(Path(picked).resolve()))
            try:
                save_config(self.folder.get(), int(str(self.port.get()).strip() or DEFAULT_PORT))
            except ValueError:
                save_config(self.folder.get(), DEFAULT_PORT)

    def open_folder(self):
        path = Path(self.folder.get().strip())
        if not path.is_dir():
            messagebox.showwarning("提示", "请先选择一个存在的文件夹。")
            return
        try:
            os.startfile(path)
        except OSError as e:
            messagebox.showerror("错误", f"无法打开文件夹：{e}")

    def copy_url(self):
        url = self.url_text.get().strip()
        if not url:
            return
        self.clipboard_clear()
        self.clipboard_append(url)
        self.append_log(f"已复制：{url}")

    def open_in_browser(self):
        if self.httpd is None:
            messagebox.showinfo("提示", "请先启动服务，再在浏览器中打开。")
            return
        try:
            port = int(str(self.port.get()).strip() or DEFAULT_PORT)
        except ValueError:
            port = DEFAULT_PORT
        webbrowser.open(f"http://127.0.0.1:{port}/")

    def start_server(self):
        if self.httpd is not None:
            return
        folder = self.folder.get().strip()
        if not folder:
            messagebox.showwarning("提示", "请选择要共享的文件夹。")
            return
        path = Path(folder)
        try:
            path = path.expanduser().resolve()
        except OSError as e:
            messagebox.showerror("错误", f"无法解析文件夹：{e}")
            return
        if not path.is_dir():
            messagebox.showerror("错误", f"文件夹不存在：{path}")
            return
        if path_in_git_worktree(path):
            if not messagebox.askyesno(
                    "提示",
                    "所选文件夹位于本工具的 Git 仓库内，对方上传的文件会出现在 git 变更中。\n"
                    "建议改选仓库以外的目录。仍要继续吗？"):
                return
        try:
            os.listdir(path)
        except OSError as e:
            messagebox.showerror("错误", f"无法读取该文件夹：{e}")
            return
        try:
            port = int(str(self.port.get()).strip())
        except ValueError:
            messagebox.showerror("错误", "端口必须是 1 到 65535 之间的整数。")
            return
        if not 1 <= port <= 65535:
            messagebox.showerror("错误", "端口必须是 1 到 65535 之间的整数。")
            return

        self.folder.set(str(path))
        handler = make_handler(str(path), self, self.append_log)
        try:
            httpd = ReusableHTTPServer(("0.0.0.0", port), handler)
        except OSError as e:
            messagebox.showerror("启动失败", f"无法监听端口 {port}：{e}")
            return

        self.httpd = httpd
        self.server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self.server_thread.start()
        save_config(str(path), port)
        self.refresh_addresses()
        self._set_running(True)
        self.append_log(f"已共享：{path}")
        self.append_log(f"正在监听 0.0.0.0:{port}")
        for url in self.urls:
            self.append_log(f"访问：{url}")

    def stop_server(self):
        httpd = self.httpd
        self.httpd = None
        if httpd is None:
            self._set_running(False)
            return
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass
        self._set_running(False)
        self.append_log("服务已停止")

    def _set_running(self, running):
        self.status.set("运行中" if running else "未启动")
        state_idle = tk.DISABLED if running else tk.NORMAL
        state_run = tk.NORMAL if running else tk.DISABLED
        self.btn_start.config(state=state_idle)
        self.btn_stop.config(state=state_run)
        self.btn_browse.config(state=state_idle)
        self.folder_entry.config(state="readonly" if running else tk.NORMAL)
        self.port_entry.config(state="readonly" if running else tk.NORMAL)

    def append_log(self, message):
        def _ui():
            self.log.configure(state="normal")
            self.log.insert(tk.END, message + "\n")
            extra = int(self.log.index("end-1c").split(".")[0]) - LOG_MAX_LINES
            if extra > 0:
                self.log.delete("1.0", f"{extra + 1}.0")
            self.log.see(tk.END)
            self.log.configure(state="disabled")
        if threading.current_thread() is threading.main_thread():
            _ui()
        else:
            self.after(0, _ui)

    def on_close(self):
        self.stop_server()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
