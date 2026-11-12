param([string]$OutputDirectory = (Join-Path $PSScriptRoot "dist"))
$ErrorActionPreference = 'Stop'
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path $compiler)) { $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe' }
$output = $OutputDirectory
New-Item -ItemType Directory -Force $output | Out-Null
& $compiler /nologo /target:winexe /platform:anycpu /optimize+ /r:System.Windows.Forms.dll /r:System.Drawing.dll /r:System.Core.dll "/win32icon:$PSScriptRoot\assets\ChatGPT.ico" "/out:$output\ChatGPT-Connection-Helper.exe" "/resource:$PSScriptRoot\assets\ChatGPT.png,ChatGPT.png" "/resource:$PSScriptRoot\assets\ChatGPT.ico,ChatGPT.ico" "$PSScriptRoot\src\Launcher.cs" "$PSScriptRoot\src\ProcessMode.cs"
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
$test = Start-Process "$output\ChatGPT-Connection-Helper.exe" -ArgumentList '--self-test' -WindowStyle Hidden -Wait -PassThru
if ($test.ExitCode -ne 0) { throw 'Self-test failed' }
Copy-Item -LiteralPath "$PSScriptRoot\README.md" -Destination (Join-Path $output "README.md") -Force
Write-Output "Build and self-test passed: $output\ChatGPT-Connection-Helper.exe"
