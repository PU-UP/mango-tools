import argparse, io, os, sys, json, time, urllib.parse, socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from pathlib import Path
from email import policy
from email.parser import BytesParser
import tempfile

class UploadHandler(SimpleHTTPRequestHandler):
    # ---- 目录渲染：支持 JSON & HTML ----
    def list_directory(self, path):
        # 判断是否要返回 JSON
        u = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(u.query)
        wants_json = (
            ("format" in qs and any(v.lower() == "json" for v in qs["format"])) or
            ("application/json" in (self.headers.get("Accept") or ""))
        )

        # 构造目录项
        try:
            names = os.listdir(path)
        except OSError:
            self.send_error(404, "No permission to list directory")
            return None

        # 统一按照文件名排序（目录在前，名称升序）
        def _key(name):
            p = Path(path) / name
            is_dir = p.is_dir()
            return (not is_dir, name.lower())
        names.sort(key=_key)

        if wants_json:
            items = []
            for name in names:
                p = Path(path) / name
                st = p.stat()
                items.append({
                    "name": name,
                    "href": urllib.parse.quote(name) + ("/" if p.is_dir() else ""),
                    "is_dir": p.is_dir(),
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                })
            data = json.dumps({
                "cwd": os.path.abspath(path),
                "items": items,
            }, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return io.BytesIO(data)

        # 完整生成页面后再发送长度，避免添加上传表单后正文被截断。
        title = escape(urllib.parse.unquote(u.path))
        rows = []
        for name in names:
            suffix = "/" if (Path(path) / name).is_dir() else ""
            href = urllib.parse.quote(name) + suffix
            rows.append(f'<li><a href="{href}">{escape(name + suffix)}</a></li>')
        data = (
            f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>Directory listing for {title}</title></head><body>'
            f'<h1>Directory listing for {title}</h1>'
            '<hr><h3>Upload file</h3>'
            '<form enctype="multipart/form-data" method="post">'
            '<input name="file" type="file" multiple>'
            '<input type="submit" value="Upload"></form><hr><ul>'
            + "".join(rows) + '</ul><hr></body></html>'
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        return io.BytesIO(data)

    # ---- 简单上传实现（multipart/form-data） ----
    def do_POST(self):
        ctype = self.headers.get_content_type()
        if ctype != 'multipart/form-data':
            self.send_error(400, "Expected multipart/form-data")
            return

        boundary = self.headers.get_param('boundary')
        if not boundary:
            self.send_error(400, "No boundary")
            return

        try:
            length = int(self.headers.get('Content-Length', ''))
            if length <= 0 or self.headers.get('Transfer-Encoding'):
                raise ValueError("Invalid Content-Length")
            root = Path(self.directory).resolve()
            url_path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
            segments = [s for s in url_path.split('/') if s]
            if any(s in ('.', '..') or any(c in s for c in ('\\', ':', '\x00')) for s in segments):
                raise ValueError("Invalid upload directory")
            directory = root.joinpath(*segments).resolve()
            directory.relative_to(root)

            # ponytail: 保留整包内存解析；超大文件上传需改为流式 multipart。
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete upload")
            message = BytesParser(policy=policy.default).parsebytes(
                b'Content-Type: ' + self.headers['Content-Type'].encode('ascii')
                + b'\r\nMIME-Version: 1.0\r\n\r\n' + raw
            )
            if not message.is_multipart() or any(p.defects for p in message.walk()):
                raise ValueError("Invalid multipart body")
            files = []
            for part in message.iter_parts():
                filename = part.get_filename()
                if not filename:
                    continue
                filename = os.path.basename(filename.replace('\\', '/'))
                if not filename or filename in ('.', '..') or ':' in filename or '\x00' in filename:
                    raise ValueError("Invalid filename")
                target = (directory / filename).resolve()
                target.relative_to(root)
                body = part.get_payload(decode=True)
                if body is None:
                    raise ValueError("Invalid file body")
                files.append((target, body))
            if not files:
                raise ValueError("No files supplied")
        except (ValueError, UnicodeError):
            self.send_error(400, "Invalid upload data or path")
            return

        saved = 0
        try:
            directory.mkdir(parents=True, exist_ok=True)
            for target, body in files:
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(dir=directory, prefix='.upload-', delete=False) as f:
                        temp_path = Path(f.name)
                        f.write(body)
                    os.replace(temp_path, target)
                    saved += 1
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
        except OSError:
            self.send_error(500, "Cannot save uploaded file")
            return

        msg = f"Uploaded {saved} file(s)."
        data = f"<html><body><h3>{msg}</h3><a href='.'>Back</a></body></html>".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

def get_local_ip():
    """获取本机IP地址"""
    try:
        # 连接到一个远程地址来获取本机IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int, nargs="?", default=8000)
    ap.add_argument("--directory", default=None)
    args = ap.parse_args()
    if args.directory:
        os.chdir(args.directory)
    local_ip = get_local_ip()
    with ThreadingHTTPServer(("0.0.0.0", args.port), UploadHandler) as httpd:
        print(f"Current IP: {local_ip}")
        print(f"Serving (upload+list) on 0.0.0.0:{args.port} dir={os.getcwd()}")
        print(f"Access via: http://{local_ip}:{args.port}")
        httpd.serve_forever()
