Get-ChildItem -Path ".\data\Replay Data" -Directory | ForEach-Object {
    $folder = $_.FullName
    $replay = Get-ChildItem -Path $folder -Filter "*.REPLAY" | Select-Object -First 1
    if ($replay) {
        $output = Join-Path $folder "replay_summary.json"
        .\rrrocket.exe -p $replay.FullName > $output
        Write-Host "Processed $($replay.Name) -> $output"
    } else {
        Write-Host "No .REPLAY file found in $folder"
    }
}
