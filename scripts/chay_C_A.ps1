# Huong C (du tru dung luong) + A (huan luyen dai hon) tren Windows, PowerShell 5.1 tro len.
# Chay tu thu muc goc cua ma (noi co scripts\run.py), sau khi da kich hoat .venv:
#   powershell -ExecutionPolicy Bypass -File scripts\chay_C_A.ps1 -Buoc 1
# Buoc 1: quet ty le du tru R tren checkpoint cu (nhanh, khong huan luyen)
# Buoc 2: huan luyen tiep co du tru (A + C) cho cac seed
# Buoc 3: benchmark day du voi du tru (moi phuong phap dung cung R)
param(
    [int]$Buoc = 1,
    [string]$Topo = "real:geant2012",
    [string]$TopoDir = "real_geant2012",
    [double]$R = 0.3,
    [int]$Steps = 300000,
    [int]$SongSong = 3,
    [int[]]$Seeds = @(0,1,2,3,4,5,6,7,8,9)
)

$ErrorActionPreference = "Stop"
$py = Join-Path (Get-Location).Path ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = (Get-Command python).Source }
New-Item -ItemType Directory -Force -Path "logs\C" | Out-Null

$queue = New-Object System.Collections.ArrayList   # moi phan tu: @(ten_log, mang_tham_so)

function Add-Job([string]$log, [string[]]$a) { [void]$queue.Add(@($log, $a)) }

function Run-Queue {
    # Chay toi da $SongSong tien trinh cung luc, moi tien trinh ghi log rieng trong logs\C.
    $running = @()
    foreach ($item in $queue) {
        while (@($running | Where-Object { -not $_.HasExited }).Count -ge $SongSong) { Start-Sleep -Seconds 10 }
        $log = $item[0]; $a = $item[1]
        Write-Host (">>> " + ($a -join " "))
        $running += Start-Process -FilePath $py -ArgumentList $a -NoNewWindow -PassThru `
            -RedirectStandardOutput "logs\C\$log.out.log" -RedirectStandardError "logs\C\$log.err.log"
    }
    $running | ForEach-Object { $_.WaitForExit() }
}

$tag = "GNN-C" + [int]($R * 100)
if ($Buoc -eq 1) {
    foreach ($r in 0.1, 0.2, 0.3, 0.4, 0.5) {
        foreach ($s in 0..4) {
            Add-Job "screen_r$($r)_s$s" @("scripts\run.py", "bench", "--topo", $Topo, "--shield", "--reserve", "$r",
                "--loads", "0.75", "1.0", "1.25", "--seeds", "$s",
                "--heuristics", "Greedy-Latency", "Greedy-Energy",
                "--rl", "GNN-PPO=runs\$TopoDir\GNN-PPO_s{seed}",
                "--out", "results\C_screen_$TopoDir\r$($r)_s$s")
        }
    }
}
elseif ($Buoc -eq 2) {
    foreach ($s in $Seeds) {
        Add-Job "train_$($tag)_s$s" @("scripts\run.py", "train", "--topo", $Topo, "--reserve", "$R",
            "--encoder", "gine", "--seed", "$s", "--steps", "$Steps",
            "--lr", "1.5e-4", "--lambda-init", "12", "--lagrangian", "0.98", "--beta", "10", "--eta", "0",
            "--init-from", "runs\$TopoDir\GNN-PPO_s$s\last.pt",
            "--out", "runs\$TopoDir\$($tag)_s$s")
    }
}
elseif ($Buoc -eq 3) {
    foreach ($s in $Seeds) {
        Add-Job "bench_$($tag)_s$s" @("scripts\run.py", "bench", "--topo", $Topo, "--shield", "--reserve", "$R",
            "--seeds", "$s",
            "--rl", "GNN-MAPPO-C=runs\$TopoDir\$($tag)_s{seed}", "GNN-PPO=runs\$TopoDir\GNN-PPO_s{seed}",
            "--out", "results\E6_reserve_$TopoDir\r$($R)_s$s")
    }
}
Run-Queue
Write-Host "Xong buoc $Buoc. Log trong logs\C, ket qua trong results\."
