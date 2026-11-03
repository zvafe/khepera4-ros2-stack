# Khepera IV (BRT K4) PowerShell Launcher
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

# Dynamically convert current workspace path to WSL Linux mount path
$normalizedPath = $PSScriptRoot.Replace('\', '/')
$wslPath = (wsl -d Ubuntu wslpath -a "$normalizedPath").Trim()

if (-not $wslPath) {
    Write-Error "Failed to resolve WSL path for '$PSScriptRoot'. Please ensure WSL Ubuntu is running."
    exit 1
}

$argsStr = ($ScriptArgs -join " ")
$wslCommand = "cd '$wslPath' && ./run.sh $argsStr"
wsl -d Ubuntu bash -c $wslCommand

