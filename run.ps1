# Khepera IV (BRT K4) PowerShell Launcher
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$wslCommand = "cd /mnt/c/Users/Andy/Desktop/khepera4-ros2-stack-main && ./run.sh " + ($ScriptArgs -join " ")
wsl -d Ubuntu bash -c $wslCommand
