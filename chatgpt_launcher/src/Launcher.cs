using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.Win32;

static class Launcher
{
    internal static readonly string[] ProxyKeys = { "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "WS_PROXY", "WSS_PROXY", "NO_PROXY" };
    internal const string Endpoint = "https://chatgpt.com/backend-api/codex/responses";
    [STAThread]
    static int Main(string[] args)
    {
        if (args.Contains("--self-test")) { SelfTest(); return 0; }
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        bool owner;
        using (var mutex = new Mutex(true, "Local\\MangoChatGPTLauncher", out owner))
        using (var wake = new EventWaitHandle(false, EventResetMode.AutoReset, "Local\\MangoChatGPTLauncherWake"))
        {
            if (!owner) { wake.Set(); return 0; }
            Application.Run(new LauncherWindow(wake));
        }
        return 0;
    }

    internal static Uri ProxyUri(string value)
    {
        Uri uri;
        if (!Uri.TryCreate(value.Trim(), UriKind.Absolute, out uri) || uri.Scheme != "http" ||
            !string.IsNullOrEmpty(uri.UserInfo) || uri.AbsolutePath != "/" || uri.Query != "" || uri.Fragment != "")
            throw new ArgumentException("请输入 HTTP 代理地址，例如 http://127.0.0.1:7890（不支持账号密码或 SOCKS 地址）。");
        return uri;
    }

    internal static ProcessStartInfo StartInfo(string exe, Uri proxy)
    {
        var info = new ProcessStartInfo(exe) { UseShellExecute = false, WorkingDirectory = Path.GetDirectoryName(exe) };
        foreach (string key in ProxyKeys) info.EnvironmentVariables.Remove(key);
        if (proxy != null)
        {
            info.EnvironmentVariables["HTTP_PROXY"] = proxy.AbsoluteUri;
            info.EnvironmentVariables["HTTPS_PROXY"] = proxy.AbsoluteUri;
            info.EnvironmentVariables["NO_PROXY"] = "localhost,127.0.0.1,::1";
        }
        return info;
    }

    internal static bool ReachableStatus(int status) { return status == 101 || status == 401 || status == 426; }
    internal static string Probe(Uri proxy)
    {
        ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
        var request = (HttpWebRequest)WebRequest.Create(Endpoint);
        request.Proxy = proxy == null ? null : new WebProxy(proxy);
        request.Timeout = 7000;
        request.ReadWriteTimeout = 7000;
        request.AllowAutoRedirect = false;
        request.Connection = "Upgrade";
        request.Headers["Upgrade"] = "websocket";
        request.Headers["Sec-WebSocket-Version"] = "13";
        request.Headers["Sec-WebSocket-Key"] = Convert.ToBase64String(Guid.NewGuid().ToByteArray());
        request.UserAgent = "MangoChatGPTLauncher/1.0";
        try
        {
            using (var response = (HttpWebResponse)request.GetResponse())
                return ReachableStatus((int)response.StatusCode) ? null : "HTTP " + (int)response.StatusCode;
        }
        catch (WebException ex)
        {
            using (var response = ex.Response as HttpWebResponse)
                return response != null && ReachableStatus((int)response.StatusCode) ? null :
                    response == null ? ex.Status.ToString() : "HTTP " + (int)response.StatusCode;
        }
    }

    internal static string FindApp()
    {
        // Query the installed package each time so Store updates do not break the path.
        string command = "Get-AppxPackage | Where-Object { $_.Name -in @('OpenAI.Codex','OpenAI.ChatGPT') } | ForEach-Object { $p = Join-Path $_.InstallLocation 'app\\ChatGPT.exe'; if (Test-Path -LiteralPath $p) { $p } }";
        var info = new ProcessStartInfo(Path.Combine(Environment.SystemDirectory, "WindowsPowerShell\\v1.0\\powershell.exe"),
            "-NoProfile -NonInteractive -EncodedCommand " + Convert.ToBase64String(System.Text.Encoding.Unicode.GetBytes(command)))
            { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true };
        using (var process = Process.Start(info))
        {
            string output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            string path = output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries).FirstOrDefault(File.Exists);
            if (path != null) return path;
        }
        return null;
    }

    internal static Process[] Running(string exe)
    {
        var matching = new System.Collections.Generic.List<Process>();
        foreach (var process in Process.GetProcessesByName(Path.GetFileNameWithoutExtension(exe)))
        {
            bool matches = false;
            try { matches = string.Equals(process.MainModule.FileName, exe, StringComparison.OrdinalIgnoreCase); }
            catch (System.ComponentModel.Win32Exception) { }
            catch (InvalidOperationException) { }
            if (matches) matching.Add(process); else process.Dispose();
        }
        return matching.ToArray();
    }

    internal static bool AnyRunning(string exe)
    {
        var processes = Running(exe);
        try { return processes.Length != 0; }
        finally { foreach (var process in processes) process.Dispose(); }
    }

    internal static string DetectProxy()
    {
        using (var key = Registry.CurrentUser.OpenSubKey("Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings"))
        {
            string raw = key == null ? "" : Convert.ToString(key.GetValue("ProxyServer"));
            if (!string.IsNullOrWhiteSpace(raw))
            {
                string part = raw.Split(';').FirstOrDefault(p => p.StartsWith("https=", StringComparison.OrdinalIgnoreCase)) ??
                              raw.Split(';').FirstOrDefault(p => p.StartsWith("http=", StringComparison.OrdinalIgnoreCase)) ?? raw;
                if (part.Contains("=")) part = part.Substring(part.IndexOf('=') + 1);
                try { return ProxyUri(part.Contains("://") ? part : "http://" + part).AbsoluteUri; } catch { }
            }
        }
        return "http://127.0.0.1:7890";
    }

    static void SelfTest()
    {
        var normal = StartInfo(Path.Combine(Environment.SystemDirectory, "notepad.exe"), null);
        foreach (string key in ProxyKeys) if (normal.EnvironmentVariables.ContainsKey(key)) throw new Exception("Default environment leaked " + key);
        var proxy = StartInfo(normal.FileName, ProxyUri("http://127.0.0.1:7890"));
        if (proxy.EnvironmentVariables["HTTPS_PROXY"] != "http://127.0.0.1:7890/") throw new Exception("Proxy not set");
        foreach (string invalid in new[] { "", "socks5://localhost:1080", "http://user:pass@localhost:7890", "http://localhost/path" })
        {
            bool rejected = false; try { ProxyUri(invalid); } catch (ArgumentException) { rejected = true; }
            if (!rejected) throw new Exception("Invalid proxy accepted");
        }
        if (!ReachableStatus(101) || !ReachableStatus(401) || ReachableStatus(403) || ReachableStatus(200)) throw new Exception("Probe classification");
    }
}

sealed class LauncherWindow : Form
{
    readonly TextBox proxy = new TextBox();
    readonly TextBox appPath = new TextBox();
    readonly Label state = new Label();
    readonly TextBox details = new TextBox();
    readonly Button start = new Button();
    readonly ComboBox selection = new ComboBox();
    readonly NotifyIcon tray = new NotifyIcon();
    readonly System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer();
    readonly EventWaitHandle wake;
    bool busy, quitting, launched;
    string observedProcess;
    bool inspecting;
    Icon modeIcon;
    readonly Icon trayBrand = new Icon(typeof(Launcher).Assembly.GetManifestResourceStream("ChatGPT.ico"));
    readonly Image brand = Image.FromStream(typeof(Launcher).Assembly.GetManifestResourceStream("ChatGPT.png"));

    Label FieldLabel(string text) { return new Label { Text = text, AutoSize = true, Margin = Padding.Empty, ForeColor = Color.FromArgb(70, 77, 71) }; }
    Panel InputFrame(TextBox input)
    {
        input.BorderStyle = BorderStyle.None; input.Dock = DockStyle.Fill; input.BackColor = Color.White;
        var panel = new Panel { Dock = DockStyle.Fill, BackColor = Color.White, Padding = new Padding(11, 9, 11, 7), Margin = new Padding(0, 2, 0, 12) };
        panel.Controls.Add(input);
        panel.Paint += delegate(object sender, PaintEventArgs e) { using (var pen = new Pen(Color.FromArgb(217, 221, 217))) e.Graphics.DrawRectangle(pen, 0, 0, panel.Width - 1, panel.Height - 1); };
        return panel;
    }
    [DllImport("user32.dll")] static extern bool DestroyIcon(IntPtr icon);

    public LauncherWindow(EventWaitHandle wakeEvent)
    {
        wake = wakeEvent;
        Text = "ChatGPT 连接助手";
        Font = new Font("Microsoft YaHei UI", 10);
        AutoScaleMode = AutoScaleMode.Dpi;
        ClientSize = new Size(640, 650);
        MinimumSize = new Size(656, 689);
        StartPosition = FormStartPosition.CenterScreen;
        BackColor = Color.FromArgb(250, 250, 248);
        ForeColor = Color.FromArgb(32, 35, 33);
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(32, 28, 32, 22), ColumnCount = 1, RowCount = 12 };
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        Controls.Add(layout);
        var header = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, Margin = Padding.Empty };
        header.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 62));
        header.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        var logo = new PictureBox { Image = brand, SizeMode = PictureBoxSizeMode.Zoom, Size = new Size(42, 42), Margin = new Padding(0, 6, 0, 0) };
        header.Controls.Add(logo, 0, 0);
        var titles = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.TopDown, WrapContents = false, Margin = Padding.Empty };
        titles.Controls.Add(new Label { Text = "ChatGPT 连接助手", AutoSize = true, Font = new Font(Font.FontFamily, 20, FontStyle.Bold), Margin = Padding.Empty });
        titles.Controls.Add(new Label { Text = "为桌面应用选择合适的连接方式", AutoSize = true, ForeColor = Color.FromArgb(95, 100, 96), Margin = new Padding(0, 5, 0, 0) });
        header.Controls.Add(titles, 1, 0); layout.Controls.Add(header, 0, 0);
        state.Font = new Font(Font.FontFamily, 10, FontStyle.Bold); state.AutoSize = true;
        state.Padding = new Padding(10, 5, 10, 5); state.Margin = new Padding(0, 8, 0, 12);
        layout.Controls.Add(state, 0, 1);
        layout.Controls.Add(FieldLabel("连接方式"), 0, 2);
        selection.DropDownStyle = ComboBoxStyle.DropDownList;
        selection.Items.AddRange(new object[] { "自动选择 · 优先默认连接", "仅使用默认连接", "仅使用指定代理" });
        selection.SelectedIndex = 0; selection.Dock = DockStyle.Top; selection.AccessibleName = "连接方式";
        selection.Margin = new Padding(0, 2, 0, 12); selection.FlatStyle = FlatStyle.Flat;
        layout.Controls.Add(selection, 0, 3);
        layout.Controls.Add(FieldLabel("备用代理地址"), 0, 4);
        proxy.Text = Launcher.DetectProxy(); proxy.AccessibleName = "备用 HTTP 代理地址";
        layout.Controls.Add(InputFrame(proxy), 0, 5);
        layout.Controls.Add(FieldLabel("桌面应用"), 0, 6);
        var appRow = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, Margin = new Padding(0, 2, 0, 12) };
        appRow.RowCount = 1; appRow.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        appRow.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100)); appRow.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 82));
        appPath.AccessibleName = "ChatGPT 程序路径";
        var appFrame = InputFrame(appPath); appFrame.Margin = Padding.Empty;
        appRow.Controls.Add(appFrame, 0, 0);
        var browse = new Button { Text = "更改…", Dock = DockStyle.Fill, FlatStyle = FlatStyle.Flat, Margin = new Padding(8, 0, 0, 0), BackColor = Color.White };
        browse.FlatAppearance.BorderColor = Color.FromArgb(217, 221, 217);
        browse.Click += delegate { using (var dialog = new OpenFileDialog { Filter = "ChatGPT 程序|ChatGPT.exe;Codex.exe", Title = "选择已安装的原版桌面应用" }) { if (dialog.ShowDialog(this) == DialogResult.OK) appPath.Text = dialog.FileName; } };
        appRow.Controls.Add(browse, 1, 0); layout.Controls.Add(appRow, 0, 7);
        start.Text = "检测并启动 ChatGPT"; start.Dock = DockStyle.Fill;
        start.Margin = new Padding(0, 7, 0, 14); start.FlatStyle = FlatStyle.Flat; start.FlatAppearance.BorderSize = 0;
        start.BackColor = Color.FromArgb(34, 48, 41); start.ForeColor = Color.White;
        start.Font = new Font(Font.FontFamily, 11, FontStyle.Bold);
        start.Paint += delegate(object sender, PaintEventArgs e) {
            e.Graphics.Clear(start.Enabled ? start.BackColor : Color.FromArgb(216, 223, 217));
            TextRenderer.DrawText(e.Graphics, start.Text, start.Font, start.ClientRectangle, start.Enabled ? Color.White : Color.FromArgb(65, 75, 67), TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter);
            if (start.Focused) ControlPaint.DrawFocusRectangle(e.Graphics, Rectangle.Inflate(start.ClientRectangle, -5, -5), Color.White, start.BackColor);
        };
        start.Click += async delegate { await Launch(); };
        layout.Controls.Add(start, 0, 8);
        layout.Controls.Add(FieldLabel("连接记录"), 0, 9);
        details.Multiline = true; details.ReadOnly = true; details.ScrollBars = ScrollBars.Vertical; details.Dock = DockStyle.Fill;
        details.BorderStyle = BorderStyle.None; details.BackColor = Color.FromArgb(241, 243, 239); details.ForeColor = Color.FromArgb(65, 72, 66); details.AccessibleName = "检测记录";
        var logFrame = new Panel { Dock = DockStyle.Fill, BackColor = details.BackColor, Padding = new Padding(12), Margin = new Padding(0, 3, 0, 12) };
        logFrame.Controls.Add(details); layout.Controls.Add(logFrame, 0, 10);
        layout.Controls.Add(new Label { Text = "绿点 默认连接    蓝点 指定代理    ·    悬停图标查看模式\n关闭窗口仅保留托盘图标；右键可完全退出助手。", AutoSize = true, ForeColor = Color.FromArgb(95, 100, 96), Font = new Font(Font.FontFamily, 9), Margin = Padding.Empty }, 0, 11);
        foreach (int height in new[] { 82, 50, 24, 48, 24, 52, 24, 54, 68, 25 }) layout.RowStyles.Add(new RowStyle(SizeType.Absolute, height));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100)); layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 44));
        var menu = new ContextMenuStrip();
        menu.Items.Add("打开连接助手", null, delegate { Reveal(); });
        menu.Items.Add("自动检测并启动 / 重启", null, async delegate { Reveal(); await Launch(); });
        menu.Items.Add("退出助手（保留 ChatGPT）", null, delegate { if (!busy) { quitting = true; Close(); } });
        tray.ContextMenuStrip = menu; tray.DoubleClick += delegate { Reveal(); }; tray.Visible = true;
        SetMode("?", "准备就绪 · 尚未选择模式", Color.DimGray);
        timer.Interval = 1500;
        timer.Tick += async delegate {
            if (wake.WaitOne(0)) Reveal();
            if (!busy) await RecognizeRunning();
        };
        timer.Start();
        Shown += async delegate {
            start.Enabled = false;
            try {
                string found = await Task.Run(() => Launcher.FindApp());
                if (IsDisposed) return;
                if (string.IsNullOrWhiteSpace(appPath.Text)) appPath.Text = found ?? "";
            }
            catch (Exception ex) { if (!IsDisposed) Log("查找应用失败：" + ex.Message); }
            finally { if (!IsDisposed) start.Enabled = true; }
            if (IsDisposed) return;
            Log(appPath.Text == "" ? "未自动找到应用，请选择 ChatGPT.exe。" : "已找到原版应用，正在识别现有进程的代理配置。");
            await RecognizeRunning();
        };
        FormClosing += delegate(object sender, FormClosingEventArgs e) { if (!quitting && e.CloseReason != CloseReason.WindowsShutDown && e.CloseReason != CloseReason.ApplicationExitCall) { e.Cancel = true; ShowInTaskbar = false; Hide(); } };
        FormClosed += delegate { timer.Stop(); tray.Visible = false; };
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
        if (disposing) { timer.Dispose(); tray.Visible = false; tray.Dispose(); if (modeIcon != null) modeIcon.Dispose(); brand.Dispose(); trayBrand.Dispose(); }
    }

    async Task RecognizeRunning()
    {
        if (inspecting || busy || IsDisposed || string.IsNullOrWhiteSpace(appPath.Text)) return;
        inspecting = true;
        try
        {
            string exe = appPath.Text.Trim();
            var processes = Launcher.Running(exe);
            try
            {
                if (processes.Length == 0)
                {
                    if (observedProcess != "none") { observedProcess = "none"; launched = false; SetMode("?", "应用未运行", Color.DimGray); start.Text = "检测并启动 ChatGPT"; }
                    return;
                }
                var main = processes.OrderBy(p => p.MainWindowHandle == IntPtr.Zero ? 1 : 0).ThenBy(p => p.StartTime).First();
                string identity = exe + "|" + main.Id + "|" + main.StartTime.ToUniversalTime().Ticks;
                if (identity == observedProcess) return;
                int id = main.Id;
                var environment = await Task.Run(() => ProcessMode.ReadProxy(id));
                if (IsDisposed || busy || exe != appPath.Text.Trim() || main.HasExited) return;
                bool usesProxy = ProcessMode.UsesProxy(environment);
                observedProcess = identity; launched = true;
                SetMode(usesProxy ? "P" : "D", usesProxy ? "代理模式 · 已识别" : "默认模式 · 已识别", usesProxy ? Color.RoyalBlue : Color.SeaGreen);
                start.Text = "重新检测并重启 ChatGPT";
                Log("已读取运行中应用的代理配置，无需重启。" + (usesProxy ? "检测到显式代理设置。" : "未设置代理环境变量，系统代理仍可能生效。"));
            }
            finally { foreach (var process in processes) process.Dispose(); }
        }
        catch (Exception)
        {
            if (!IsDisposed && !busy && observedProcess != "unreadable") { observedProcess = "unreadable"; launched = false; SetMode("?", "模式无法读取", Color.DimGray); Log("无法读取当前进程设置，未更改应用。可能因权限不同或进程正在退出。稍后自动重试。"); }
        }
        finally { inspecting = false; }
    }

    void Reveal() { ShowInTaskbar = true; Show(); WindowState = FormWindowState.Normal; Activate(); }
    void Log(string message) { details.AppendText(DateTime.Now.ToString("HH:mm:ss") + "  " + message + Environment.NewLine); }
    void SetMode(string letter, string label, Color color)
    {
        using (var bitmap = new Bitmap(64, 64))
        using (var g = Graphics.FromImage(bitmap))
        using (var brush = new SolidBrush(color))
        {
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            g.Clear(Color.Transparent); g.DrawIcon(trayBrand, new Rectangle(2, 2, 58, 58));
            if (letter != "?") { g.FillEllipse(Brushes.White, 44, 44, 18, 18); g.FillEllipse(brush, 47, 47, 12, 12); }
            IntPtr handle = bitmap.GetHicon();
            var next = (Icon)Icon.FromHandle(handle).Clone(); DestroyIcon(handle);
            var old = modeIcon; modeIcon = next; Icon = next; tray.Icon = next; if (old != null) old.Dispose();
        }
        state.Text = label; state.ForeColor = color; state.BackColor = Color.FromArgb(237, 240, 235); tray.Text = "ChatGPT 连接助手：" + label; Text = label + " — ChatGPT 连接助手";
    }

    async Task Launch()
    {
        if (busy) return;
        Process[] running = new Process[0];
        busy = true; start.Text = "正在检测连接…"; start.Enabled = false; proxy.Enabled = false; appPath.Enabled = false; selection.Enabled = false;
        try
        {
            string exe = appPath.Text.Trim();
            if (!File.Exists(exe)) throw new Exception("请先选择已安装的 ChatGPT.exe。");
            Log("正在检测连接路径…");
            int choice = selection.SelectedIndex;
            string directError = choice == 2 ? "用户选择代理" : await Task.Run(() => Launcher.Probe(null));
            Uri chosen = null;
            if (choice == 1 && directError != null) throw new Exception("默认路径未通过：" + directError + "。未改变运行中的应用。");
            if (directError != null)
            {
                Log("默认路径未通过：" + directError + "。检测备用代理…");
                chosen = Launcher.ProxyUri(proxy.Text);
                string proxyError = await Task.Run(() => Launcher.Probe(chosen));
                if (proxyError != null) throw new Exception("两条路径均未通过检测，保留正在运行的应用。代理结果：" + proxyError + "。请检查 VPN 和代理地址。");
            }
            Log("路径检测通过：" + (chosen == null ? "默认" : chosen.Authority) + "。这是未登录探测，不保证模型长连接成功。");
            running = Launcher.Running(exe);
            if (running.Length > 0)
            {
                if (MessageBox.Show(this, "需要重启 ChatGPT 才能应用模式。请结束任务、保存输入后再继续。现在正常退出并重启吗？", "切换连接", MessageBoxButtons.YesNo, MessageBoxIcon.Question, MessageBoxDefaultButton.Button2) != DialogResult.Yes) { Log("已取消，保留现有模式。"); return; }
                foreach (var process in running) { try { process.CloseMainWindow(); } catch (InvalidOperationException) { } }
                await Task.Run(() => { for (int i = 0; i < 16 && Launcher.AnyRunning(exe); i++) Thread.Sleep(500); });
                foreach (var process in running) process.Dispose();
                running = Launcher.Running(exe);
                if (running.Length > 0)
                {
                    if (MessageBox.Show(this, "应用未完全退出。强制退出会中断任务并可能丢失未保存的输入。是否强制退出？", "应用仍在运行", MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2) != DialogResult.Yes) { Log("未强制退出，请从应用托盘退出后重试。"); return; }
                    await Task.Run(() => {
                        foreach (var process in running)
                        {
                            try {
                                if (process.HasExited) continue;
                                using (var killer = Process.Start(new ProcessStartInfo(Path.Combine(Environment.SystemDirectory, "taskkill.exe"), "/PID " + process.Id + " /T /F") { UseShellExecute = false, CreateNoWindow = true })) killer.WaitForExit();
                            } catch (InvalidOperationException) { }
                        }
                    });
                    await Task.Delay(1500);
                    if (Launcher.AnyRunning(exe)) throw new Exception("应用仍未退出；未启动第二个实例，请手动退出后重试。");
                }
            }
            using (var started = Process.Start(Launcher.StartInfo(exe, chosen))) { }
            await Task.Delay(2000);
            if (!Launcher.AnyRunning(exe)) throw new Exception("启动后未找到应用进程，请检查应用安装。");
            launched = true;
            SetMode(chosen == null ? "D" : "P", chosen == null ? "默认模式" : "代理模式", chosen == null ? Color.SeaGreen : Color.RoyalBlue);
            Log("已启动：" + (chosen == null ? "未附加代理变量（系统代理仍可能生效）" : chosen.AbsoluteUri));
            tray.ShowBalloonTip(2500, "ChatGPT 已启动", state.Text + "；双击托盘图标查看。", ToolTipIcon.Info);
        }
        catch (Exception ex) { Log(ex.Message); MessageBox.Show(this, ex.Message, "未切换连接", MessageBoxButtons.OK, MessageBoxIcon.Information); }
        finally { foreach (var process in running) process.Dispose(); busy = false; start.Text = launched ? "重新检测并重启 ChatGPT" : "检测并启动 ChatGPT"; start.Enabled = true; proxy.Enabled = true; appPath.Enabled = true; selection.Enabled = true; }
    }
}
