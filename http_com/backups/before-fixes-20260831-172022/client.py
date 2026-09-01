# client.py
# 一体化 HTTP 客户端（上传 + 下载）
# 适配 python -m http.server（只读：仅下载）与 http_upload_server.py（可上传）
# 纯标准库、跨平台。界面更紧凑；默认服务器地址取本机局域网 IP；默认保存路径为脚本所在文件夹。

import threading
import urllib.request
import urllib.parse
import urllib.error
from html.parser import HTMLParser
from pathlib import Path
import mimetypes
import uuid
import socket
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import ipaddress
import struct

# ===================== 实用函数 =====================

def get_local_ip() -> str:
    """更稳健地获取本机局域网 IP（不会产生真实外连）。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # 不会真的发包，只为选择出站网卡
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        try:
            s.close()
        except Exception:
            pass
    return ip

def get_network_info():
    """获取本机网络信息（IP 和子网掩码）。"""
    try:
        local_ip = get_local_ip()
        if not local_ip or local_ip == "127.0.0.1":
            return None, None
        
        # 获取子网掩码
        import platform
        if platform.system() == "Windows":
            import subprocess
            result = subprocess.run(['ipconfig'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    # 查找包含 IPv4 地址的行
                    if 'IPv4' in line or 'IP Address' in line:
                        # 检查下一行是否包含我们的 IP
                        if i + 1 < len(lines) and local_ip in lines[i + 1]:
                            # 继续查找子网掩码
                            for j in range(i + 1, min(i + 10, len(lines))):
                                line_lower = lines[j].lower()
                                if 'subnet mask' in line_lower or '子网掩码' in lines[j]:
                                    # 提取子网掩码
                                    parts = lines[j].split(':')
                                    if len(parts) >= 2:
                                        mask_str = parts[-1].strip()
                                        try:
                                            # 转换为 CIDR 格式
                                            mask_parts = mask_str.split('.')
                                            if len(mask_parts) == 4:
                                                mask_int = sum(bin(int(x)).count('1') for x in mask_parts)
                                                return local_ip, f"/{mask_int}"
                                        except:
                                            pass
                                    break
                                # 如果遇到下一个适配器，停止搜索
                                if j + 1 < len(lines) and ('适配器' in lines[j + 1] or 'adapter' in lines[j + 1].lower()):
                                    break
        else:
            # Linux/Mac: 使用 ip 或 ifconfig
            import subprocess
            try:
                result = subprocess.run(['ip', 'addr', 'show'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if local_ip in line and '/' in line:
                            parts = line.split()
                            for part in parts:
                                if '/' in part and local_ip.split('.')[0] in part:
                                    return local_ip, part.split('/')[1]
            except:
                pass
            
            # 回退到 ifconfig
            try:
                result = subprocess.run(['ifconfig'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if local_ip in line and 'netmask' in line.lower():
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if 'netmask' in part.lower() and i+1 < len(parts):
                                    mask_hex = parts[i+1]
                                    try:
                                        mask_int = int(mask_hex, 16)
                                        mask_bin = bin(mask_int).count('1')
                                        return local_ip, f"/{mask_bin}"
                                    except:
                                        pass
            except:
                pass
        
        # 如果无法获取子网掩码，假设是 /24（255.255.255.0）
        return local_ip, "/24"
    except Exception:
        return None, None

def check_http_service(ip, port=8000, timeout=0.5):
    """检查指定 IP 的指定端口是否有 HTTP 服务。"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def scan_network_ips(local_ip, subnet_mask, port=8000, max_threads=50):
    """扫描同网段内运行 HTTP 服务的 IP 地址。"""
    found_ips = set()
    
    try:
        # 构建网络对象
        if '/' in str(subnet_mask):
            network_str = f"{local_ip}{subnet_mask}"
        else:
            # 如果是点分十进制格式，转换为 CIDR
            try:
                mask_int = sum(bin(int(x)).count('1') for x in subnet_mask.split('.'))
                network_str = f"{local_ip}/{mask_int}"
            except:
                network_str = f"{local_ip}/24"
        
        network = ipaddress.IPv4Network(network_str, strict=False)
        
        # 扫描网段内的 IP
        ip_list = list(network.hosts())
        
        def scan_ip(ip_str):
            # 跳过本机 IP
            if ip_str != local_ip and check_http_service(ip_str, port):
                found_ips.add(ip_str)
        
        # 使用线程池扫描
        threads = []
        # 限制扫描范围，避免扫描时间过长
        max_scan = min(254, len(ip_list))
        for ip in ip_list[:max_scan]:
            while len(threads) >= max_threads:
                threads = [t for t in threads if t.is_alive()]
                if len(threads) >= max_threads:
                    import time
                    time.sleep(0.1)
            
            t = threading.Thread(target=scan_ip, args=(str(ip),), daemon=True)
            t.start()
            threads.append(t)
        
        # 等待所有线程完成
        for t in threads:
            t.join(timeout=2)
            
    except Exception as e:
        pass
    
    return found_ips

def get_all_ips(scan_network=False) -> list:
    """获取所有可用的 IP 地址列表（包括本机 IP 和回环地址）。
    
    Args:
        scan_network: 是否扫描同网段的其他 IP（可能较慢）
    """
    ips = set()
    
    # 添加回环地址
    ips.add("127.0.0.1")
    
    # 获取本机局域网 IP
    try:
        local_ip = get_local_ip()
        if local_ip:
            ips.add(local_ip)
    except Exception:
        pass
    
    # 尝试获取主机名对应的所有 IP
    try:
        hostname = socket.gethostname()
        # 获取主机名对应的 IP 列表
        addrinfo_list = socket.getaddrinfo(hostname, None, socket.AF_INET)
        for addr_info in addrinfo_list:
            ip = addr_info[4][0]
            if ip and ip != "0.0.0.0":
                ips.add(ip)
    except Exception:
        pass
    
    # Linux/Mac: 尝试通过 hostname -I 获取所有接口 IP
    try:
        import platform
        if platform.system() != "Windows":
            import subprocess
            result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                for ip in result.stdout.strip().split():
                    if ip and ip != "0.0.0.0":
                        ips.add(ip)
    except Exception:
        pass
    
    # 扫描同网段的其他 IP（如果启用）
    if scan_network:
        try:
            local_ip, subnet_mask = get_network_info()
            if local_ip and subnet_mask:
                network_ips = scan_network_ips(local_ip, subnet_mask)
                ips.update(network_ips)
        except Exception:
            pass
    
    # 转换为列表并排序（127.0.0.1 在前，其他按字符串排序）
    ip_list = sorted(ips, key=lambda x: (x != "127.0.0.1", x))
    return ip_list if ip_list else ["127.0.0.1"]

SCRIPT_DIR = Path(__file__).resolve().parent

# ===================== 基础 HTTP & 解析 =====================

class LinkParser(HTMLParser):
    """稳健提取 <a href="..."> 与其可见文本，不折叠内部空白"""
    def __init__(self):
        super().__init__()
        self.links = []  # (href, text)
        self._in_a = False
        self._cur_href = None
        self._cur_text = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'a':
            self._in_a = True
            self._cur_href = dict(attrs).get('href')
            self._cur_text = []

    def handle_data(self, data):
        # 不 strip，保留文件名中的空格/特殊空白
        if self._in_a and data:
            self._cur_text.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == 'a' and self._in_a:
            text = "".join(self._cur_text).strip()  # 仅去掉首尾空白
            self.links.append((self._cur_href, text))
            self._in_a = False
            self._cur_href = None
            self._cur_text = []

def http_get(url, timeout=10, headers=None):
    hdrs = {"User-Agent": "HTTP-Client/1.0"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers


import json, urllib.parse

def list_dir(url):
    """返回目录项 [(name, href_abs, is_dir)]"""
    # ---- 尝试 JSON ----
    try:
        # 1) 先试 Accept 头；2) 再试 ?format=json
        raw, hdr = http_get(url, headers={"Accept": "application/json"})
        ctype = (hdr.get_content_type() or "").lower()
        if ctype == "application/json":
            obj = json.loads(raw.decode("utf-8", errors="replace"))
            items = []
            for it in obj.get("items", []):
                href_abs = urllib.parse.urljoin(url, it["href"])
                items.append((it["name"], href_abs, bool(it["is_dir"])))
            items.sort(key=lambda x: (not x[2], x[0].lower()))
            return items
        # 如果不是 JSON，继续走 HTML
    except Exception:
        # 再尝试在 URL 上加 ?format=json
        try:
            sep = "&" if ("?" in url) else "?"
            raw, hdr = http_get(url + f"{sep}format=json")
            if (hdr.get_content_type() or "").lower() == "application/json":
                obj = json.loads(raw.decode("utf-8", errors="replace"))
                items = []
                for it in obj.get("items", []):
                    href_abs = urllib.parse.urljoin(url, it["href"])
                    items.append((it["name"], href_abs, bool(it["is_dir"])))
                items.sort(key=lambda x: (not x[2], x[0].lower()))
                return items
        except Exception:
            pass

    # ---- 回落：解析 HTML ----
    html, _ = http_get(url)
    parser = LinkParser()
    parser.feed(html.decode(errors="replace"))  # 用 replace 防止丢字符

    items = []
    for href, text in parser.links:
        if not href or href.startswith("?") or href.startswith("#"):
            continue
        t = (text or "").strip()
        if t in ("Parent Directory", "../"):
            continue
        abs_url = urllib.parse.urljoin(url, href)
        is_dir = href.endswith("/")
        name_from_href = urllib.parse.unquote(href.rstrip("/").split("/")[-1])
        name = (t.rstrip("/") if t else name_from_href)
        items.append((name, abs_url, is_dir))

    uniq, seen = [], set()
    for it in items:
        key = (it[1], it[2])
        if key not in seen:
            uniq.append(it)
            seen.add(key)
    uniq.sort(key=lambda x: (not x[2], x[0].lower()))
    return uniq

def crawl_tree(base_url):
    """递归抓取目录下所有文件 URL 列表"""
    result = []
    def _walk(url):
        try:
            entries = list_dir(url)
        except Exception:
            result.append(url)
            return
        for _name, href, is_dir in entries:
            if is_dir:
                _walk(href)
            else:
                result.append(href)
    _walk(base_url)
    return result

def download_file(url, dest_path, progress_cb=None, timeout=30, chunk_size=128*1024):
    req = urllib.request.Request(url, headers={"User-Agent": "HTTP-Client/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = resp.getheader("Content-Length")
        total = int(total) if total and total.isdigit() else None
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        read_bytes = 0
        with open(dest_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                read_bytes += len(chunk)
                if progress_cb:
                    progress_cb(read_bytes, total)

def multipart_post(url, files, timeout=30, subpath=""):
    """上传本地文件列表到 url(+subpath)，使用 multipart/form-data"""
    if not url.endswith("/"):
        url += "/"
    if subpath:
        url += subpath.strip("/") + "/"

    boundary = uuid.uuid4().hex
    CRLF = "\r\n"
    body = []

    for p in files:
        p = Path(p)
        if not p.is_file():
            continue
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        body.append(f"--{boundary}{CRLF}".encode())
        head = (
            f'Content-Disposition: form-data; name="file"; filename="{p.name}"{CRLF}'
            f"Content-Type: {mime}{CRLF}{CRLF}"
        ).encode()
        body.append(head)
        body.append(p.read_bytes())
        body.append(CRLF.encode())

    body.append(f"--{boundary}--{CRLF}".encode())
    data = b"".join(body)

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(data)),
            "User-Agent": "HTTP-Client/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode(errors="ignore")

# ===================== GUI 应用 =====================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HTTP 传输助手（上传 + 下载）")
        # 初始更紧凑；用户可拉伸
        self.geometry("880x520")
        self.minsize(760, 420)

        # 默认值：服务器地址取本机 IP；保存到脚本所在目录
        self.available_ips = get_all_ips(scan_network=False)  # 初始不扫描，避免启动慢
        self.initial_ip_count = len(self.available_ips)  # 保存初始 IP 数量
        default_ip = self.available_ips[0] if self.available_ips else get_local_ip()
        self.current_url = tk.StringVar(value=f"http://{default_ip}:8000/")
        self.save_dir = tk.StringVar(value=str(SCRIPT_DIR))
        self.upload_subdir = tk.StringVar(value="")
        self.upload_to_current = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="就绪")

        self.history = []
        self.upload_files = []
        self.scanning = False

        self._build_ui()

    def _build_ui(self):
        # 顶部：地址栏 & 动作（网格，避免过宽）
        top = ttk.Frame(self, padding=(10,8,10,4))
        top.pack(fill=tk.X)
        ttk.Label(top, text="服务器：").grid(row=0, column=0, sticky="w")
        
        # 创建 IP 选项列表（格式：http://IP:8000/）
        ip_options = [f"http://{ip}:8000/" for ip in self.available_ips]
        # 使用可编辑的下拉框，既可以从列表选择，也可以手动输入
        self.url_combo = ttk.Combobox(top, textvariable=self.current_url, values=ip_options, width=40)
        self.url_combo.grid(row=0, column=1, sticky="we", padx=6)
        
        ttk.Button(top, text="扫描网络", command=self.scan_network).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="连接/刷新", command=self.connect).grid(row=0, column=3, padx=4)
        ttk.Button(top, text="后退", command=self.go_back).grid(row=0, column=4)
        top.grid_columnconfigure(1, weight=1)

        # 主体：左右分栏，可拖动
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # 左：浏览 & 下载
        left = ttk.Labelframe(paned, text="浏览 & 下载", padding=8)
        paned.add(left, weight=3)

        cols = ("name", "type", "url")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="extended", height=14)
        self.tree.heading("name", text="名称")
        self.tree.heading("type", text="类型")
        self.tree.heading("url",  text="URL")
        self.tree.column("name", width=260, anchor=tk.W)
        self.tree.column("type", width=70, anchor=tk.W)
        self.tree.column("url",  width=360, anchor=tk.W)
        self.tree.grid(row=0, column=0, columnspan=4, sticky="nsew")
        self.tree.bind("<Double-1>", self.on_double_click)

        sb_left = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        sb_left.grid(row=0, column=4, sticky="ns", padx=(6,0))
        self.tree.configure(yscrollcommand=sb_left.set)

        # 下载区
        ttk.Label(left, text="保存到：").grid(row=1, column=0, sticky="w", pady=(8,0))
        ttk.Entry(left, textvariable=self.save_dir).grid(row=1, column=1, sticky="we", padx=6, pady=(8,0))
        ttk.Button(left, text="选择…", command=self.choose_save_dir).grid(row=1, column=2, padx=2, pady=(8,0))
        self.btn_download_sel = ttk.Button(left, text="下载所选", command=self.download_selected)
        self.btn_download_sel.grid(row=1, column=3, padx=2, pady=(8,0))
        self.btn_download_all = ttk.Button(left, text="下载全部", command=self.download_all)
        self.btn_download_all.grid(row=1, column=4, padx=(2,0), pady=(8,0))

        left.grid_columnconfigure(1, weight=1)
        left.grid_rowconfigure(0, weight=1)

        # 右：上传
        right = ttk.Labelframe(paned, text="上传", padding=8)
        paned.add(right, weight=2)

        up_top = ttk.Frame(right)
        up_top.grid(row=0, column=0, columnspan=3, sticky="we")
        ttk.Button(up_top, text="添加文件…", command=self.add_upload_files).pack(side=tk.LEFT)
        ttk.Button(up_top, text="清空", command=self.clear_upload_files).pack(side=tk.LEFT, padx=6)

        self.listbox = tk.Listbox(right, height=11)
        self.listbox.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=6)
        sb_u = ttk.Scrollbar(right, command=self.listbox.yview)
        sb_u.grid(row=1, column=3, sticky="ns", padx=(6,0))
        self.listbox.config(yscrollcommand=sb_u.set)

        ttk.Checkbutton(right, text="上传到当前浏览路径", variable=self.upload_to_current).grid(row=2, column=0, sticky="w")
        ttk.Label(right, text="或 子目录：").grid(row=2, column=1, sticky="e", padx=(10,4))
        ttk.Entry(right, textvariable=self.upload_subdir, width=18).grid(row=2, column=2, sticky="we")
        self.btn_upload = ttk.Button(right, text="开始上传", command=self.do_upload)
        self.btn_upload.grid(row=3, column=2, sticky="e", pady=(8,0))

        right.grid_columnconfigure(0, weight=1)
        right.grid_columnconfigure(1, weight=0)
        right.grid_columnconfigure(2, weight=1)
        right.grid_rowconfigure(1, weight=1)

        # 底部：进度 & 状态
        bottom = ttk.Frame(self, padding=(10,0,10,10))
        bottom.pack(fill=tk.X)
        ttk.Label(bottom, text="总进度：").pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Label(bottom, textvariable=self.status, width=40).pack(side=tk.RIGHT, anchor="e")

    # ============ 浏览 ============

    def scan_network(self):
        """扫描同网段内运行 HTTP 服务的 IP 地址。"""
        if self.scanning:
            messagebox.showinfo("提示", "正在扫描中，请稍候...")
            return
        
        self.scanning = True
        self.status.set("正在扫描同网段 IP...")
        
        def worker():
            try:
                # 扫描网络
                new_ips = get_all_ips(scan_network=True)
                
                # 更新 IP 列表
                def update_ui():
                    self.available_ips = new_ips
                    ip_options = [f"http://{ip}:8000/" for ip in self.available_ips]
                    self.url_combo['values'] = ip_options
                    found_count = len(new_ips) - self.initial_ip_count
                    if found_count > 0:
                        self.status.set(f"扫描完成，找到 {found_count} 个同网段服务器")
                    else:
                        self.status.set("扫描完成，未找到其他服务器")
                    self.scanning = False
                
                self.after(0, update_ui)
            except Exception as e:
                def update_ui():
                    self.status.set(f"扫描失败：{e}")
                    self.scanning = False
                self.after(0, update_ui)
        
        threading.Thread(target=worker, daemon=True).start()

    def connect(self):
        url = self.current_url.get().strip()
        if not url:
            messagebox.showwarning("提示", "请输入服务器地址，例如：http://192.168.1.10:8000/")
            return
        if not url.endswith("/"):
            url += "/"
            self.current_url.set(url)
        if not self.history or self.history[-1] != url:
            self.history.append(url)
        self._load_dir(url)

    def go_back(self):
        if len(self.history) >= 2:
            self.history.pop()
            url = self.history[-1]
            self.current_url.set(url)
            self._load_dir(url)

    def _load_dir(self, url):
        self._set_busy(True, "正在加载目录…")
        def worker():
            try:
                items = list_dir(url)
                self._update_tree(items)
                self._log(f"加载完成：{url}")
            except Exception as e:
                self._error(f"加载失败：{e}")
            finally:
                self._set_busy(False)
        threading.Thread(target=worker, daemon=True).start()

    def _update_tree(self, items):
        def _ui():
            for i in self.tree.get_children():
                self.tree.delete(i)
            for name, href, is_dir in items:
                self.tree.insert("", tk.END, values=(name, "目录" if is_dir else "文件", href))
        self.after(0, _ui)

    def on_double_click(self, _evt):
        sel = self.tree.selection()
        if not sel:
            return
        name, typ, url = self.tree.item(sel[0])["values"]
        if typ == "目录":
            url = url if url.endswith("/") else url + "/"
            if not self.history or self.history[-1] != url:
                self.history.append(url)
            self.current_url.set(url)
            self._load_dir(url)
        else:
            if messagebox.askyesno("下载", f"下载文件：{name}?"):
                self._download_urls([url])

    def choose_save_dir(self):
        d = filedialog.askdirectory(initialdir=self.save_dir.get() or str(SCRIPT_DIR))
        if d:
            self.save_dir.set(d)

    # ============ 下载 ============

    def download_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("提示", "请先在列表中选择文件或文件夹。")
            return

        url_list = []         # 纯 URL 字符串
        folder_urls = []

        for iid in sel:
            name, typ, url = self.tree.item(iid)["values"]
            if typ == "目录":
                folder_urls.append(url)
            else:
                url_list.append(url)

        def scan_and_go():
            try:
                for base_url in folder_urls:
                    url_list.extend(crawl_tree(base_url))
            except Exception as e:
                self._error(f"扫描失败：{e}")
            finally:
                if url_list:
                    self._download_urls(url_list)

        if folder_urls:
            if not messagebox.askyesno("确认", f"将递归下载 {len(folder_urls)} 个文件夹中的所有文件，确认继续？"):
                return
            self._set_busy(True, "正在扫描目录…")
            threading.Thread(target=scan_and_go, daemon=True).start()
        else:
            if url_list:
                self._download_urls(url_list)

    def download_all(self):
        children = self.tree.get_children()
        if not children:
            return

        url_list = []
        folder_urls = []

        for iid in children:
            name, typ, url = self.tree.item(iid)["values"]
            if typ == "目录":
                folder_urls.append(url)
            else:
                url_list.append(url)

        def scan_and_go():
            try:
                for base_url in folder_urls:
                    url_list.extend(crawl_tree(base_url))
            except Exception as e:
                self._error(f"扫描失败：{e}")
            finally:
                if url_list:
                    self._download_urls(url_list)

        if folder_urls:
            if not messagebox.askyesno("确认", f"将递归下载当前目录下所有内容（包含 {len(folder_urls)} 个文件夹），继续？"):
                return
            self._set_busy(True, "正在扫描目录…")
            threading.Thread(target=scan_and_go, daemon=True).start()
        else:
            if url_list:
                self._download_urls(url_list)

    def _download_urls(self, urls):
        dest_root = Path(self.save_dir.get() or SCRIPT_DIR)
        dest_root.mkdir(parents=True, exist_ok=True)
        total = len(urls)
        if total == 0:
            return
        self.progress["value"] = 0
        self.progress["maximum"] = total
        self._set_busy(True, f"准备下载 {total} 个文件…")

        def to_local_path(url):
            try:
                u = urllib.parse.urlparse(url)
                path = urllib.parse.unquote(u.path.lstrip("/"))
                return dest_root / path
            except Exception:
                return dest_root / Path(url.split("/")[-1])

        def worker():
            ok = 0
            fail = 0
            for idx, url in enumerate(urls, 1):
                try:
                    dst = to_local_path(url)
                    def per_prog(rb, tb):
                        if tb:
                            pct = int(rb*100/max(1,tb))
                            self.status.set(f"下载中（{idx}/{total}）：{dst.name}  {pct}%")
                        else:
                            self.status.set(f"下载中（{idx}/{total}）：{dst.name}  {rb/1024:.1f} KB")
                    download_file(url, dst, progress_cb=per_prog)
                    ok += 1
                except Exception as e:
                    fail += 1
                    self._log(f"失败：{url} - {e}")
                finally:
                    self.after(0, lambda: self.progress.step(1))
            msg = f"下载完成：成功 {ok}，失败 {fail}。保存至 {dest_root}"
            self._log(msg)
            self._set_busy(False, msg)
            messagebox.showinfo("下载结果", msg)
        threading.Thread(target=worker, daemon=True).start()

    # ============ 上传 ============

    def add_upload_files(self):
        paths = filedialog.askopenfilenames(title="选择要上传的文件")
        for p in paths:
            self.upload_files.append(p)
            self.listbox.insert(tk.END, p)

    def clear_upload_files(self):
        self.upload_files.clear()
        self.listbox.delete(0, tk.END)

    def do_upload(self):
        if not self.upload_files:
            messagebox.showinfo("提示", "请先添加至少一个文件。")
            return
        base_url = self.current_url.get().strip()
        subdir = "" if self.upload_to_current.get() else self.upload_subdir.get().strip()

        self._set_busy(True, "正在上传…")
        def worker():
            try:
                code, text = multipart_post(base_url, self.upload_files, subpath=subdir)
                msg = f"上传完成，HTTP {code}"
                self._log(msg)
                self._set_busy(False, msg)
                messagebox.showinfo("上传结果", "上传成功！请在服务器目录刷新查看。")
            except urllib.error.HTTPError as e:
                self._set_busy(False, "上传失败")
                try:
                    detail = e.read(4096).decode(errors='ignore')
                except Exception:
                    detail = ""
                messagebox.showerror("上传失败", f"HTTP {e.code}\n{detail}")
            except Exception as e:
                self._set_busy(False, "上传失败")
                messagebox.showerror("上传失败", str(e))
        threading.Thread(target=worker, daemon=True).start()

    # ============ 辅助 ============

    def _set_busy(self, busy, msg=None):
        def _ui():
            st = tk.DISABLED if busy else tk.NORMAL
            self.btn_download_sel.config(state=st)
            self.btn_download_all.config(state=st)
            self.btn_upload.config(state=st)
            if msg:
                self.status.set(msg)
        self.after(0, _ui)

    def _log(self, msg):
        self.after(0, lambda: self.status.set(msg))

    def _error(self, msg):
        self.after(0, lambda: (self.status.set(msg), messagebox.showerror("错误", msg)))

if __name__ == "__main__":
    App().mainloop()
