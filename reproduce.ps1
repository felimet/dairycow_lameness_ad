param([string]$DataRoot = 'F:/cow-data/01_data', [string]$Out = 'results')
$ErrorActionPreference = 'Stop'
try {
    Set-Location -LiteralPath $PSScriptRoot
    # conda's Windows activation batch file requires an ASCII temporary path.
    $envInfo = conda env list --json | ConvertFrom-Json
    $cowEnv = $envInfo.envs | Where-Object { (Split-Path $_ -Leaf) -eq 'cowlame' } | Select-Object -First 1
    if (-not $cowEnv) { throw 'The cowlame conda environment is missing.' }
    $env:TEMP = Join-Path $cowEnv 'tmp'
    $env:TMP = $env:TEMP
    New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
    foreach ($script in @('make_cohort.py', 'run_main.py', 'run_ablations.py', 'make_report.py')) {
        conda run --no-capture-output -n cowlame python "scripts/$script" --data-root $DataRoot --out $Out
        if ($LASTEXITCODE -ne 0) { throw "$script failed with exit code $LASTEXITCODE" }
    }
} catch {
    Write-Error "Reproduction failed: $_"
    exit 1
}
