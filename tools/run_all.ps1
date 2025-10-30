param(
  [switch]$FreshLog,
  [switch]$Force,
  [double]$RvMax = 60,
  [int]$PollSec = 60,
  [string]$PMode,
  [string]$PLong,
  [string]$AllowOnly,
  [double]$PMin = 0.40,   # proxy selection threshold
  [switch]$Paper,         # <-- NEW: run executor in paper mode (default true)
  [switch]$Live          # <-- NEW: run executor in live mode (uses API keys)
)

$base = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path "$base\.."
Set-Location $root

# refuse to start if already running (unless -Force)
$existing = Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'tools\\live_(writer|proxy|executor).*\.py|tools\\live_proxy_loop\.ps1' }
if ($existing -and -not $Force) {
  Write-Host "[run_all] already running:"
  $existing | Select-Object ProcessId, CommandLine
  Write-Host "[run_all] use -Force or run .\tools\stop_all.ps1 first."
  exit 0
}

# PYTHONPATH so 'ml_dl' imports work
$env:PYTHONPATH = ($root.Path)

# artifacts (fill if missing)
if (-not $env:DL_TX_MODEL_PATH)   { $env:DL_TX_MODEL_PATH   = "model_artifacts\tx\dl_tx_latest.pt" }
if (-not $env:DL_TX_SCALER_PATH)  { $env:DL_TX_SCALER_PATH  = "model_artifacts\tx\scaler_latest.joblib" }
if (-not $env:DL_TCN_MODEL_PATH)  { $env:DL_TCN_MODEL_PATH  = "model_artifacts\tcn\dl_tcn_latest.pt" }
if (-not $env:DL_TCN_SCALER_PATH) { $env:DL_TCN_SCALER_PATH = "model_artifacts\tcn\scaler_latest.joblib" }
if (-not $env:DL_LSTM_MODEL_PATH) { $env:DL_LSTM_MODEL_PATH = "model_artifacts\lstm\dl_lstm_latest.pt" }
if (-not $env:DL_LSTM_SCALER_PATH){ $env:DL_LSTM_SCALER_PATH= "model_artifacts\lstm\scaler_latest.joblib" }

# writer gating knobs
if ($PMode)     { $env:DL_P_LONG_MODE = $PMode }     elseif (-not $env:DL_P_LONG_MODE) { $env:DL_P_LONG_MODE = "abs" }
if ($PLong)     { $env:DL_P_LONG      = $PLong }     elseif (-not $env:DL_P_LONG)      { $env:DL_P_LONG      = "0.458314" }
if ($AllowOnly) { $env:DL_ALLOW_ONLY  = $AllowOnly } elseif (-not $env:DL_ALLOW_ONLY)  { $env:DL_ALLOW_ONLY  = "1" }

# general knobs
if (-not $env:DL_MAX_LOOKBACK_PAD) { $env:DL_MAX_LOOKBACK_PAD = "6000" }
if (-not $env:DL_SYMBOLS)          { $env:DL_SYMBOLS          = "BTCUSDT,ETHUSDT" }
if (-not $env:DL_TIMEFRAME)        { $env:DL_TIMEFRAME        = "1m" }
if (-not $env:DL_SEQ_LEN)          { $env:DL_SEQ_LEN          = "128" }
if (-not $env:DL_LOG_DIR)          { $env:DL_LOG_DIR          = "logs" }

# fresh log header if requested (master only; signals file is dynamic)
if ($FreshLog) {
  'ts,p_meta,thr,mode,rv_mean,allow,kinds_used' | Out-File .\live_meta_log.csv -Encoding utf8
}

# ensure logs dir exists
New-Item -ItemType Directory -Path .\logs -Force | Out-Null

# start writer
$writer = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList "tools\live_writer.py" `
  -RedirectStandardOutput ".\logs\live_writer.out" `
  -RedirectStandardError  ".\logs\live_writer.err" `
  -WorkingDirectory $root `
  -PassThru -WindowStyle Hidden

# start proxy loop with PMin
$proxy = Start-Process -FilePath "powershell.exe" `
  -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","tools\live_proxy_loop.ps1","-RvMax",$RvMax,"-PollSec",$PollSec,"-PMin",$PMin `
  -RedirectStandardOutput ".\logs\live_proxy.out" `
  -RedirectStandardError  ".\logs\live_proxy.err" `
  -WorkingDirectory $root `
  -PassThru -WindowStyle Hidden

# start executor (paper by default unless -Live specified)
$execArgs = @("tools\live_executor.py",
              "--signals","logs\live_signals.csv",
              "--rv-max",$RvMax,
              "--plong",$env:DL_P_LONG,
              "--pmode",$env:DL_P_LONG_MODE,
              "--max-pos-usd","100",          # conservative canary
              "--max-symbols","1")

if ($Live) {
  $execArgs += "--live"
} else {
  $execArgs += "--paper"
}

$executor = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList $execArgs `
  -RedirectStandardOutput ".\logs\live_executor.out" `
  -RedirectStandardError  ".\logs\live_executor.err" `
  -WorkingDirectory $root `
  -PassThru -WindowStyle Hidden

Write-Host "[run_all] started writer PID=$($writer.Id), proxy-loop PID=$($proxy.Id), executor PID=$($executor.Id)"
Write-Host "[run_all] safe to close this window; processes keep running."
Write-Host "Logs: .\logs\live_writer.out, .\logs\live_proxy.out, .\logs\live_executor.out"
