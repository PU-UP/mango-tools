using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

// Read only; never write process memory or save the environment block.
static class ProcessMode
{
    [DllImport("kernel32.dll", SetLastError = true)] static extern IntPtr OpenProcess(uint access, bool inherit, int id);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool IsWow64Process(IntPtr handle, out bool wow64);
    [DllImport("kernel32.dll", SetLastError = true)] static extern bool ReadProcessMemory(IntPtr handle, IntPtr address, byte[] buffer, IntPtr size, out IntPtr read);
    [DllImport("ntdll.dll")] static extern int NtQueryInformationProcess(IntPtr handle, int kind, byte[] data, int size, out int returned);

    static byte[] Read(IntPtr handle, long address, int size)
    {
        var data = new byte[size]; IntPtr read;
        if (!ReadProcessMemory(handle, new IntPtr(address), data, new IntPtr(size), out read) || read.ToInt64() != size)
            throw new InvalidOperationException("无法读取进程代理设置");
        return data;
    }
    static long Pointer(byte[] data, int offset, bool wide)
    { return wide ? BitConverter.ToInt64(data, offset) : BitConverter.ToUInt32(data, offset); }

    internal static Dictionary<string, string> ReadProxy(int id)
    {
        IntPtr handle = OpenProcess(0x0400 | 0x0010, false, id);
        if (handle == IntPtr.Zero) throw new InvalidOperationException("无权读取进程代理设置");
        try
        {
            bool wow;
            if (!IsWow64Process(handle, out wow)) throw new InvalidOperationException("无法确认进程架构");
            if (Environment.Is64BitOperatingSystem && !Environment.Is64BitProcess)
                throw new InvalidOperationException("请使用 64 位助手识别此进程");
            bool wide = Environment.Is64BitProcess && !wow;
            var info = new byte[wow ? IntPtr.Size : IntPtr.Size * 6]; int returned;
            if (NtQueryInformationProcess(handle, wow ? 26 : 0, info, info.Length, out returned) != 0)
                throw new InvalidOperationException("无法查询进程设置");
            long peb = Pointer(info, wow ? 0 : IntPtr.Size, Environment.Is64BitProcess);
            // shortcut: Windows process-parameter layout; unsupported layouts fail closed as unknown.
            long parameters = Pointer(Read(handle, peb + (wide ? 0x20 : 0x10), wide ? 8 : 4), 0, wide);
            long address = Pointer(Read(handle, parameters + (wide ? 0x80 : 0x48), wide ? 8 : 4), 0, wide);
            if (address == 0) throw new InvalidOperationException("进程环境尚未就绪");
            var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            var entry = new StringBuilder();
            for (int total = 0; total < 1024 * 1024;)
            {
                int size = (int)Math.Min(512, 4096 - (address % 4096));
                byte[] block = Read(handle, address, size);
                for (int i = 0; i < block.Length; i += 2)
                {
                    char c = (char)(block[i] | block[i + 1] << 8);
                    if (c != '\0') { entry.Append(c); continue; }
                    if (entry.Length == 0) return result;
                    string line = entry.ToString(); entry.Clear();
                    int equals = line.IndexOf('=');
                    if (equals <= 0) continue;
                    string key = line.Substring(0, equals);
                    if (Array.Exists(Launcher.ProxyKeys, k => k.Equals(key, StringComparison.OrdinalIgnoreCase)))
                        result[key] = line.Substring(equals + 1);
                }
                address += size; total += size;
            }
            throw new InvalidOperationException("进程环境超出读取范围");
        }
        finally { CloseHandle(handle); }
    }

    internal static bool UsesProxy(Dictionary<string, string> environment)
    {
        foreach (string key in Launcher.ProxyKeys)
        {
            string value;
            if (key != "NO_PROXY" && environment.TryGetValue(key, out value) && !string.IsNullOrWhiteSpace(value)) return true;
        }
        return false;
    }
}
