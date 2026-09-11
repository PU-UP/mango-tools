# clash_converter

版本：1.0.0

把 `vless://` 链接转成 Clash Meta 可用的 YAML。页面在本地浏览器中运行，不把节点内容发到任何服务器。

支持两种输入：

- Quantumult / Shadowrocket 风格：`vless://Base64(auto:uuid@host:port)?remarks=...&tls=1&peer=...&xtls=2&pbk=...&sid=...&fingerprint=chrome`
- 标准 Xray 风格：`vless://uuid@host:port?security=reality&sni=...&fp=chrome&pbk=...&sid=...&flow=xtls-rprx-vision#名称`

`xtls=2` 且带 `pbk` / `sid` 时，按 VLESS + Reality + `xtls-rprx-vision` 写出。

## 启动

双击 `启动转换器.bat`，或用浏览器直接打开 `index.html`。不需要安装 Python 或其他依赖。

## 使用

1. 把链接粘贴到左侧，每行一条。
2. 选择「完整配置」或「仅 proxies」。
3. 点「转换」，或按 `Ctrl+Enter`。
4. 复制 YAML，或下载为 `clash-meta.yaml`，再导入 Clash Meta / Clash Party / Clash Verge。

复制和下载在尚未转换时会先自动转换一次。

## 输出说明

完整配置包含：

- `mixed-port: 7890`
- `proxies` 节点列表
- `PROXY` 手动选择组、`AUTO` 延迟测试组
- `GEOIP,CN,DIRECT` 与 `MATCH,PROXY`

Clash 客户端需要自带 GeoIP 数据，否则 `GEOIP` 规则不会生效。

## 限制

- 只转换 `vless://`。`ss://`、`vmess://`、`trojan://` 等会记为失败行。
- Reality 节点默认带 `flow: xtls-rprx-vision`。若机场实际未开启 Vision，需要手工删掉该字段。
- 生成的是通用 Clash Meta 配置，不是某个客户端的完整偏好设置。

## 变更记录

- 1.0.0（2026-09-11）：首次发布。可视化转换 vless 链接为 Clash Meta YAML。
