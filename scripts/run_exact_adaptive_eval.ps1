param(
    [Parameter(Mandatory = $true)]
    [string]$Model,

    [string]$RepoPath = "..\AdaptiveAttackAgent_tmp",
    [string]$DataSetting = "base_subset",
    [int]$PerStrategy = 3,
    [switch]$SkipGeneration
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $projectRoot

$cmd = @(
    "python",
    "evaluation/run_exact_adaptive_eval.py",
    "--repo-path", $RepoPath,
    "--model", $Model,
    "--data-setting", $DataSetting,
    "--per-strategy", "$PerStrategy"
)

if ($SkipGeneration) {
    $cmd += "--skip-generation"
}

Write-Host "Running exact adaptive evaluation..." -ForegroundColor Cyan
Write-Host ($cmd -join " ") -ForegroundColor DarkGray

& $cmd[0] $cmd[1..($cmd.Length - 1)]
