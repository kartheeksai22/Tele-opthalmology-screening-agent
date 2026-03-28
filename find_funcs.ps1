$content = Get-Content index.html -Encoding UTF8 -Raw
$matches = [regex]::Matches($content, '(?i)(streak|suggestion|Days\s+going|updateStreakDisplay|buildSuggest)')
foreach ($m in $matches | Select-Object -First 20) {
    $lineNum = ($content.Substring(0, $m.Index) -split "`n").Count
    $context = $content.Substring([Math]::Max(0, $m.Index-20), [Math]::Min(80, $content.Length - $m.Index + 20))
    Write-Host "Line ~$lineNum : $context"
}
