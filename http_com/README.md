# http_com

可信局域网中的 HTTP 文件传输工具，包含可上传的服务端和图形客户端。程序仅使用 Python 标准库。

## Windows 快速启动

1. 确保已经安装带 Tkinter 的 Python 3。
2. 在要共享文件的电脑上双击 `启动服务器.bat`，选择文件夹后启动服务。窗口会显示当前 IP 和访问地址。
3. 在另一台电脑上双击 `启动客户端.bat`，输入或扫描服务端地址，然后浏览、上传或下载文件。

客户端默认把文件保存到本目录的 `downloads/`；该目录不会进入 Git。
图形服务端会把上次选择的文件夹和端口记在用户目录（Windows 下为 `%LOCALAPPDATA%\mango-tools\http_com\`），不会写入仓库。
请把共享目录选在本仓库以外，避免对方上传的文件出现在 git 变更中。
首次启动服务端时，若 Windows 防火墙弹出提示，需要允许专用网络访问。

## 命令行启动服务端

```powershell
python server.py 8000 --directory "D:\path\to\shared-folder"
```

## 命令行启动客户端

```powershell
python client.py
```

## 测试

```powershell
python -B -m unittest -v test_http_com
```

测试使用临时目录和本机回环连接，不会扫描真实局域网。修复背景、限制和旧版恢复方式见 [REPAIR_NOTES.md](REPAIR_NOTES.md)。

## 使用范围

没有身份验证或加密，只适合可信局域网。上传仍会把整个请求读入内存，避免一次上传过多超大文件。

## 变更记录

- 2026-09-11：新增图形服务端。可双击 `启动服务器.bat` 启动，显示本机 IP，并选择要共享的文件夹。
