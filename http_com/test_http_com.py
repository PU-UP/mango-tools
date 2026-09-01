"""Local regression checks: python -B -m unittest -v test_http_com"""
import functools
import http.client
import socket
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch
from http.server import ThreadingHTTPServer

import client
import server


class QuietHandler(server.UploadHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/short':
            self.send_response(200)
            self.send_header('Content-Length', '100')
            self.end_headers()
            self.wfile.write(b'abc')
        else:
            super().do_GET()


class TransferTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='http-com-test-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.shared = self.root / 'shared'
        self.shared.mkdir()
        self.source = self.root / 'source'
        self.source.mkdir()
        self.httpd = ThreadingHTTPServer(
            ('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(self.shared)))
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.httpd.server_port}/'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.patch = patch.object(client.urllib.request, 'urlopen', opener.open)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def source_file(self, name='sample.bin', data=b'new data'):
        path = self.source / name
        path.write_bytes(data)
        return path

    def test_binary_unicode_semicolon_and_empty_roundtrip(self):
        payload = bytes(range(256)) * 600 + b'\r\n'
        sources = [self.source_file('data.bin', payload),
                   self.source_file('\u4e2d\u6587;a.txt', b'hello\r\n\r\n'),
                   self.source_file('empty.bin', b'')]
        status, _ = client.multipart_post(self.url, sources)
        self.assertEqual(status, 200)
        items = client.list_dir(self.url)
        self.assertEqual({x[0] for x in items}, {p.name for p in sources})
        for name, url, is_dir in items:
            dest = client.to_local_path(url, self.root / 'downloads')
            client.download_file(url, dest)
            self.assertEqual(dest.read_bytes(), (self.source / name).read_bytes())
        self.assertFalse(list(self.shared.glob('.upload-*')))

    def test_upload_existing_and_new_subdirectories(self):
        source = self.source_file()
        (self.shared / source.name).write_bytes(b'keep root')
        (self.shared / 'existing').mkdir()
        for subdir in ['existing', '\u5b50\u76ee\u5f55 with space/nested']:
            status, _ = client.multipart_post(self.url, [source], subpath=subdir)
            self.assertEqual(status, 200)
            self.assertEqual((self.shared / subdir / source.name).read_bytes(), b'new data')
        self.assertEqual((self.shared / source.name).read_bytes(), b'keep root')
        client.multipart_post(self.url + 'existing/', [source])
        self.assertEqual((self.shared / 'existing' / source.name).read_bytes(), b'new data')

    def test_html_and_head_lengths(self):
        (self.shared / 'last-file.txt').write_bytes(b'last')
        connection = http.client.HTTPConnection('127.0.0.1', self.httpd.server_port)
        self.addCleanup(connection.close)
        connection.request('GET', '/')
        response = connection.getresponse()
        body = response.read()
        self.assertEqual(len(body), int(response.getheader('Content-Length')))
        self.assertIn(b'last-file.txt', body)
        self.assertIn(b'multipart/form-data', body)
        self.assertTrue(body.endswith(b'</html>'))
        connection.request('HEAD', '/')
        response = connection.getresponse()
        self.assertEqual(int(response.getheader('Content-Length')), len(body))
        self.assertEqual(response.read(), b'')

    def test_incomplete_download_preserves_old_file(self):
        dest = self.root / 'downloads' / 'old.bin'
        dest.parent.mkdir()
        dest.write_bytes(b'old complete data')
        with self.assertRaises(OSError):
            client.download_file(self.url + 'short', dest)
        self.assertEqual(dest.read_bytes(), b'old complete data')
        self.assertEqual(list(dest.parent.iterdir()), [dest])
        dest.unlink()
        with self.assertRaises(OSError):
            client.download_file(self.url + 'short', dest)
        self.assertEqual(list(dest.parent.iterdir()), [])

    def test_download_write_failure_preserves_old_file(self):
        (self.shared / 'file.bin').write_bytes(b'complete')
        dest = self.root / 'old.bin'
        dest.write_bytes(b'original')
        with patch.object(client.os, 'replace', side_effect=PermissionError('locked')):
            with self.assertRaises(PermissionError):
                client.download_file(self.url + 'file.bin', dest)
        self.assertEqual(dest.read_bytes(), b'original')
        self.assertFalse(list(self.root.glob('.download-*')))

    def test_unsafe_paths_rejected(self):
        for path in ['/%2e%2e/escape', '/%2Fescape', '/C%3A/escape',
                     '/%5C%5Cserver/share', '/x/%2e%2e/escape', '/x%00.txt', '/']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                client.to_local_path(self.url.rstrip('/') + path, self.root / 'downloads')
        source = self.source_file()
        for path in ['%2e%2e/out', 'C%3A/out', '%5Cout']:
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as error:
                client.multipart_post(self.url + path + '/', [source])
            self.assertEqual(error.exception.code, 400)
        self.assertEqual(list(self.shared.iterdir()), [])

    def test_invalid_upload_does_not_overwrite(self):
        target = self.shared / 'old.bin'
        target.write_bytes(b'old')
        # Valid file part, but missing the terminating multipart boundary.
        body = b'--BOUND\r\nContent-Disposition: form-data; name="file"; filename="old.bin"\r\n\r\nnew\r\n'
        for data, length in [(body, str(len(body))), (b'', '-1')]:
            connection = http.client.HTTPConnection('127.0.0.1', self.httpd.server_port)
            try:
                connection.request('POST', '/', body=data, headers={
                    'Content-Type': 'multipart/form-data; boundary=BOUND', 'Content-Length': length})
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 400)
            finally:
                connection.close()
        self.assertEqual(target.read_bytes(), b'old')

    def test_short_upload_does_not_overwrite(self):
        target = self.shared / 'old.bin'
        target.write_bytes(b'old')
        with socket.create_connection(('127.0.0.1', self.httpd.server_port), timeout=3) as sock:
            sock.sendall(b'POST / HTTP/1.0\r\nContent-Type: multipart/form-data; boundary=B\r\nContent-Length: 100\r\n\r\nabc')
            sock.shutdown(socket.SHUT_WR)
            self.assertIn(b'400', sock.recv(4096).split(b'\r\n')[0])
        self.assertEqual(target.read_bytes(), b'old')

    def test_missing_upload_file_is_error(self):
        with self.assertRaises(FileNotFoundError):
            client.multipart_post(self.url, [self.source / 'missing'])
        with self.assertRaises(ValueError):
            client.multipart_post(self.url, [])

    def test_regular_http_server_compatibility(self):
        # HTML fallback remains usable with Python's standard read-only server.
        with patch.object(QuietHandler, 'list_directory', server.SimpleHTTPRequestHandler.list_directory):
            (self.shared / 'normal.txt').write_text('normal')
            self.assertEqual(client.list_dir(self.url)[0][0], 'normal.txt')


class ImmediateThread:
    def __init__(self, target, args=(), **kwargs):
        self.target, self.args = target, args
    def start(self):
        self.target(*self.args)
    def is_alive(self):
        return False
    def join(self, **kwargs):
        pass


class NetworkTests(unittest.TestCase):
    def test_windows_and_linux_masks(self):
        windows = 'Ethernet adapter:\n IPv4 Address : 192.168.10.42\n Subnet Mask : 255.255.0.0\n'
        linux = ' inet 192.168.10.42/16 brd 192.168.255.255 scope global eth0\n'
        for system, output in [('Windows', windows), ('Linux', linux)]:
            with self.subTest(system=system), patch.object(client, 'get_local_ip', return_value='192.168.10.42'), patch('platform.system', return_value=system), patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, stdout=output)):
                self.assertEqual(client.get_network_info(), ('192.168.10.42', '/16'))

    def test_scan_range_is_bounded_and_near_local_ip(self):
        for mask in ['16', '/16', '255.255.0.0', '/0', '/24']:
            scanned = []
            def check(ip, port):
                scanned.append(ip)
                return ip == '192.168.10.43'
            with self.subTest(mask=mask), patch.object(client.threading, 'Thread', ImmediateThread), patch.object(client, 'check_http_service', side_effect=check):
                result = client.scan_network_ips('192.168.10.42', mask)
                self.assertEqual(len(scanned), 253)
                self.assertTrue(all(ip.startswith('192.168.10.') for ip in scanned))
                self.assertEqual(result, {'192.168.10.43'})


class FakeTree:
    def selection(self):
        return ['folder']
    def get_children(self):
        return ['folder']
    def item(self, iid):
        return {'values': ('empty', '\u76ee\u5f55', 'http://localhost/empty/')}


class FakeApp:
    tree = FakeTree()
    def __init__(self):
        self.busy, self.errors, self.downloads = [], [], []
    def _set_busy(self, busy, *args):
        self.busy.append(busy)
    def _error(self, message):
        self.errors.append(message)
    def _download_urls(self, urls):
        self.downloads.extend(urls)
    def after(self, delay, callback):
        callback()


class GuiFlowTests(unittest.TestCase):
    def test_default_download_dir_is_not_the_script_dir(self):
        self.assertEqual(client.DOWNLOAD_DIR, client.SCRIPT_DIR / 'downloads')

    def test_empty_and_failed_folder_scan_restore_buttons(self):
        for method in [client.App.download_selected, client.App.download_all]:
            for error in [None, OSError('cannot list')]:
                app = FakeApp()
                with self.subTest(method=method.__name__, error=error), patch.object(client, 'crawl_tree', return_value=[], side_effect=error), patch.object(client.threading, 'Thread', ImmediateThread), patch.object(client.messagebox, 'askyesno', return_value=True):
                    method(app)
                self.assertEqual(app.busy, [True, False])
                self.assertEqual(bool(app.errors), error is not None)
                self.assertEqual(app.downloads, [])

    def test_directory_error_propagates(self):
        with patch.object(client, 'list_dir', side_effect=OSError('unreachable')):
            with self.assertRaises(OSError):
                client.crawl_tree('http://localhost/folder/')


if __name__ == '__main__':
    unittest.main()
