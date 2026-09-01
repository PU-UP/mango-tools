# http_com

可信局域网中的 HTTP 文件传输工具，包含可上传的服务端和图形客户端。程序仅使用 Python 标准库。

## Windows 快速启动

1. 确保已经安装带 Tkinter 的 Python 3。
2. 双击 `启动客户端.bat`。
3. 输入或扫描服务端地址，然后浏览、上传或下载文件。

客户端默认把文件保存到本目录的 `downloads/`；该目录不会进入 Git。

## 启动服务端

在需要共享文件的电脑上运行：

```powershell
python server.py 8000 --directory "D:\path\to\shared-folder"
```

服务端会监听所有网卡，并显示局域网访问地址。Windows 防火墙首次询问时，需要允许专用网络访问。

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
