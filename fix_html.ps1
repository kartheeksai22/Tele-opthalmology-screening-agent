$lines = Get-Content -Path 'index.html' -Encoding UTF8
$keep = $lines[0..1352] + $lines[1456..($lines.Length-1)]
$keep | Set-Content -Path 'index.html' -Encoding UTF8
Write-Host "Done. Total lines: $($keep.Length)"
