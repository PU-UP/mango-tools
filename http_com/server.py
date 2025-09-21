import argparse, io, os, sys, json, time, urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from pathlib import Path

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

        # ---- HTML 模式：复用父类页面，并在顶部插入上传表单 ----
        r = super().list_directory(path)  # 返回 BytesIO
        extra = (b"<hr><h3>Upload file</h3>"
                 b"<form ENCTYPE='multipart/form-data' method='post'>"
                 b"<input name='file' type='file' multiple>"
                 b"<input type='submit' value='Upload'>"
                 b"</form><hr>")
        return io.BytesIO(extra + r.read())

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

        length = int(self.headers.get('content-length', 0))
        raw = self.rfile.read(length)

        boundary_bytes = b"--" + boundary.encode()
        parts = raw.split(boundary_bytes)
        saved = 0

        for part in parts:
            if not part or part in (b"--\r\n", b"\r\n"):
                continue
            # 每段形如：\r\nHeaders\r\n\r\nBODY\r\n
            if part.startswith(b"\r\n"):
                part = part[2:]
            head, sep, body = part.partition(b"\r\n\r\n")
            if not sep:
                continue
            # 去尾部结语
            if body.endswith(b"\r\n"):
                body = body[:-2]
            # 找文件名
            filename = None
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-disposition:"):
                    # 兼容不同浏览器的 filename 格式
                    segs = line.decode(errors="ignore").split(";")
                    for s in segs:
                        s = s.strip()
                        if s.startswith("filename="):
                            filename = s.split("=", 1)[1].strip().strip('"')
                            break
            if not filename:
                continue
            filename = os.path.basename(filename)
            with open(filename, "wb") as f:
                f.write(body)
            saved += 1

        msg = f"Uploaded {saved} file(s)."
        data = f"<html><body><h3>{msg}</h3><a href='.'>Back</a></body></html>".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int, nargs="?", default=8000)
    ap.add_argument("--directory", default=None)
    args = ap.parse_args()
    if args.directory:
        os.chdir(args.directory)
    with ThreadingHTTPServer(("0.0.0.0", args.port), UploadHandler) as httpd:
        print(f"Serving (upload+list) on 0.0.0.0:{args.port} dir={os.getcwd()}")
        httpd.serve_forever()
