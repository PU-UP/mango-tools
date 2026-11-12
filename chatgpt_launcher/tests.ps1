param([switch]$Network)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\build.ps1" -OutputDirectory "$PSScriptRoot\work"
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path $compiler)) { $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe' }
& $compiler /nologo /target:exe /main:Tests /r:System.Windows.Forms.dll /r:System.Drawing.dll /r:System.Core.dll "/out:$PSScriptRoot\work\Tests.exe" "/resource:$PSScriptRoot\assets\ChatGPT.png,ChatGPT.png" "/resource:$PSScriptRoot\assets\ChatGPT.ico,ChatGPT.ico" "$PSScriptRoot\src\Launcher.cs" "$PSScriptRoot\src\ProcessMode.cs" "$PSScriptRoot\tests\Tests.cs"
if ($LASTEXITCODE -ne 0) { throw 'Test compilation failed' }
if ($Network) { & "$PSScriptRoot\work\Tests.exe" --network } else { & "$PSScriptRoot\work\Tests.exe" }
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
