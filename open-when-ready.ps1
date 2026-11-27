$url = 'http://127.0.0.1:8767/'
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
        $response = Invoke-RestMethod -Uri "${url}api/health" -TimeoutSec 2
        if ($response.app -eq 'aethermap-road-demo') {
            Start-Process $url
            exit 0
        }
    } catch { }
    Start-Sleep -Milliseconds 500
}
exit 1
