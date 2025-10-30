# Stop python processes for writer/proxy/executor and any proxy loop ps1
$procs = Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'tools\\live_(writer|proxy|executor).*\.py|tools\\live_proxy_loop\.ps1' }

if ($procs) {
  foreach ($p in $procs) {
    try {
      Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
      Write-Host "[stop_all] stopped PID=$($p.ProcessId)"
    } catch {
      Write-Host "[stop_all] could not stop PID=$($p.ProcessId): $($_.Exception.Message)"
    }
  }
} else {
  Write-Host "[stop_all] nothing to stop."
}
