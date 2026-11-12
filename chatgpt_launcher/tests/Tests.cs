using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Windows.Forms;

static class Tests
{
    [STAThread]
    static void Main(string[] args)
    {
        string key = "HTTPS_PROXY", original = Environment.GetEnvironmentVariable(key);
        try
        {
            Environment.SetEnvironmentVariable(key, "http://parent.invalid:9999");
            foreach (bool useProxy in new[] { false, true })
            {
                var info = Launcher.StartInfo(Path.Combine(Environment.SystemDirectory, "WindowsPowerShell\\v1.0\\powershell.exe"), useProxy ? Launcher.ProxyUri("http://127.0.0.1:7890") : null);
                info.Arguments = "-NoProfile -NonInteractive -Command \"[Console]::Write($env:HTTPS_PROXY)\"";
                info.CreateNoWindow = true; info.RedirectStandardOutput = true;
                using (var child = Process.Start(info))
                {
                    string output = child.StandardOutput.ReadToEnd(); child.WaitForExit();
                    if (child.ExitCode != 0 || output != (useProxy ? "http://127.0.0.1:7890/" : "")) throw new Exception("Child environment mismatch");
                }
                if (Environment.GetEnvironmentVariable(key) != "http://parent.invalid:9999") throw new Exception("Parent environment modified");
            }
            Console.WriteLine("PASS: child inheritance and parent isolation");
        }
        finally { Environment.SetEnvironmentVariable(key, original); }
        foreach (string folder in new[] { Environment.SystemDirectory, Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows), "SysWOW64") })
        foreach (bool configured in new[] { false, true })
        {
            string shell = Path.Combine(folder, "WindowsPowerShell\\v1.0\\powershell.exe");
            if (!File.Exists(shell)) continue;
            var info = Launcher.StartInfo(shell, configured ? Launcher.ProxyUri("http://127.0.0.1:7890") : null);
            info.Arguments = "-NoProfile -NonInteractive -Command \"[Console]::WriteLine('ready'); [Console]::ReadLine() | Out-Null\"";
            info.CreateNoWindow = true; info.RedirectStandardOutput = true; info.RedirectStandardInput = true;
            using (var child = Process.Start(info))
            {
                try {
                    if (child.StandardOutput.ReadLine() != "ready") throw new Exception("Fixture not ready");
                    var environment = ProcessMode.ReadProxy(child.Id);
                    if (ProcessMode.UsesProxy(environment) != configured) throw new Exception("Live process mode mismatch");
                    if (configured && environment["HTTPS_PROXY"] != "http://127.0.0.1:7890/") throw new Exception("Proxy value mismatch");
                } finally { child.StandardInput.WriteLine(); if (!child.WaitForExit(3000)) child.Kill(); }
            }
        }
        bool rejectedProcess = false;
        try { ProcessMode.ReadProxy(int.MaxValue); } catch (InvalidOperationException) { rejectedProcess = true; }
        if (!rejectedProcess) throw new Exception("Missing process must not be classified as default");
        Console.WriteLine("PASS: read live 32/64-bit default/proxy processes; missing process stays unknown");
        if (Array.IndexOf(args, "--inspect") >= 0)
        {
            string exe = Launcher.FindApp();
            if (exe == null) throw new Exception("App not found");
            var processes = Launcher.Running(exe);
            var main = System.Linq.Enumerable.First(System.Linq.Enumerable.OrderBy(processes, p => p.StartTime));
            Console.WriteLine("Running ChatGPT PID " + main.Id + ": " + (ProcessMode.UsesProxy(ProcessMode.ReadProxy(main.Id)) ? "P" : "D"));
            foreach(var process in processes) process.Dispose();
        }
        Application.EnableVisualStyles();
        using (var wake = new EventWaitHandle(false, EventResetMode.AutoReset))
        using (var window = new LauncherWindow(wake))
        {
            window.Show();
            Application.DoEvents();
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            var button = (Button)typeof(LauncherWindow).GetField("start", flags).GetValue(window);
            for (int i = 0; i < 60 && !button.Enabled; i++) { Application.DoEvents(); Thread.Sleep(100); }
            ((TextBox)typeof(LauncherWindow).GetField("appPath", flags).GetValue(window)).Text = @"C:\Program Files\WindowsApps\OpenAI.Codex\app\ChatGPT.exe";
            button.Enabled = true;
            var setMode = typeof(LauncherWindow).GetMethod("SetMode", flags);
            foreach (string mode in new[] { "?", "D", "P" })
            {
                setMode.Invoke(window, new object[] { mode, mode == "P" ? "代理模式" : mode == "D" ? "默认模式" : "准备就绪", mode == "P" ? Color.RoyalBlue : mode == "D" ? Color.SeaGreen : Color.DimGray });
                Application.DoEvents();
                using (var iconPreview = window.Icon.ToBitmap()) iconPreview.Save(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "icon-" + (mode == "?" ? "ready" : mode) + ".png"));
                using (var bitmap = new Bitmap(window.Width, window.Height))
                {
                    window.DrawToBitmap(bitmap, new Rectangle(0, 0, window.Width, window.Height));
                    bitmap.Save(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "preview-" + (mode == "?" ? "ready" : mode) + ".png"));
                }
            }
            window.Scale(new SizeF(1.5f, 1.5f));
            Application.DoEvents();
            using (var scaled = new Bitmap(window.Width, window.Height)) {
                window.DrawToBitmap(scaled, new Rectangle(0, 0, window.Width, window.Height));
                scaled.Save(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "preview-150.png"));
            }
            window.Close();
            var tray = (NotifyIcon)typeof(LauncherWindow).GetField("tray", flags).GetValue(window);
            if (window.IsDisposed || window.ShowInTaskbar || window.Visible || !tray.Visible) throw new Exception("Close must hide taskbar and retain tray");
            typeof(LauncherWindow).GetMethod("Reveal", flags).Invoke(window, null);
            if (!window.Visible || !window.ShowInTaskbar) throw new Exception("Tray must restore window");
            tray.ContextMenuStrip.Items[2].PerformClick();
            if (!window.IsDisposed || tray.Visible) throw new Exception("Exit must remove tray icon");

        }
        bool lateUiError = false;
        System.Threading.ThreadExceptionEventHandler handler = delegate { lateUiError = true; };
        Application.ThreadException += handler;
        try {
            using (var wake = new EventWaitHandle(false, EventResetMode.AutoReset))
            using (var closing = new LauncherWindow(wake)) {
                closing.Show(); Application.DoEvents();
                var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
                var tray = (NotifyIcon)typeof(LauncherWindow).GetField("tray", flags).GetValue(closing);
                tray.ContextMenuStrip.Items[2].PerformClick();
                if (!closing.IsDisposed) throw new Exception("Exit during discovery failed");
                for (int i = 0; i < 30; i++) { Application.DoEvents(); Thread.Sleep(100); }
            }
            if (lateUiError) throw new Exception("Discovery callback accessed a closed window");
        } finally { Application.ThreadException -= handler; }
        Console.WriteLine("PASS: exit during asynchronous application discovery");
        Console.WriteLine("PASS: form rendering");
        if (Array.IndexOf(args, "--network") >= 0)
        {
            Console.WriteLine("Installed application: " + (Launcher.FindApp() ?? "not found"));
            Console.WriteLine("Default path: " + (Launcher.Probe(null) ?? "reachable (unauthenticated)"));
            string address = Launcher.DetectProxy();
            Console.WriteLine("Proxy " + address + ": " + (Launcher.Probe(Launcher.ProxyUri(address)) ?? "reachable (unauthenticated)"));
        }
    }
}
