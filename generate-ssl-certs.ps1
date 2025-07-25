# PowerShell script to generate self-signed SSL certificates for development
# Run this script to create SSL certificates for HTTPS

Write-Host "🔐 Generating self-signed SSL certificates..." -ForegroundColor Green

# Create SSL directory if it doesn't exist
$sslDir = ".\nginx\ssl"
if (!(Test-Path $sslDir)) {
    New-Item -ItemType Directory -Path $sslDir -Force | Out-Null
}

# Check if OpenSSL is available
$opensslPath = Get-Command openssl -ErrorAction SilentlyContinue
if (!$opensslPath) {
    Write-Host "❌ OpenSSL not found. Please install OpenSSL first." -ForegroundColor Red
    Write-Host "💡 You can install it using:" -ForegroundColor Yellow
    Write-Host "   - Chocolatey: choco install openssl" -ForegroundColor Yellow
    Write-Host "   - Or download from: https://slproweb.com/products/Win32OpenSSL.html" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🔄 Alternative: Use Windows built-in certificate generation..." -ForegroundColor Cyan
    
    # Generate certificate using PowerShell (Windows 10/Server 2016+)
    try {
        $cert = New-SelfSignedCertificate -DnsName "localhost", "127.0.0.1" -CertStoreLocation "cert:\LocalMachine\My" -KeyUsage DigitalSignature,KeyEncipherment -KeyAlgorithm RSA -KeyLength 2048 -Provider "Microsoft RSA SChannel Cryptographic Provider" -HashAlgorithm SHA256 -NotAfter (Get-Date).AddYears(1)
        
        # Export certificate to PEM format
        $certPath = ".\nginx\ssl\cert.pem"
        $keyPath = ".\nginx\ssl\key.pem"
        
        # Export certificate
        $certBytes = $cert.Export([System.Security.Cryptography.X509Certificates.X509ContentType]::Cert)
        $certPem = [System.Convert]::ToBase64String($certBytes, [System.Base64FormattingOptions]::InsertLineBreaks)
        "-----BEGIN CERTIFICATE-----`n$certPem`n-----END CERTIFICATE-----" | Out-File -FilePath $certPath -Encoding ASCII
        
        Write-Host "✅ Certificate generated using Windows PowerShell!" -ForegroundColor Green
        Write-Host "⚠️  Note: Private key export requires additional steps with Windows certificates." -ForegroundColor Yellow
        Write-Host "💡 For full functionality, please install OpenSSL and run this script again." -ForegroundColor Yellow
        
    } catch {
        Write-Host "❌ Failed to generate certificate: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
    exit 0
}

# Generate private key
Write-Host "📝 Generating private key..." -ForegroundColor Blue
& openssl genrsa -out ".\nginx\ssl\key.pem" 2048

# Generate certificate signing request
Write-Host "📋 Generating certificate signing request..." -ForegroundColor Blue
& openssl req -new -key ".\nginx\ssl\key.pem" -out ".\nginx\ssl\cert.csr" -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"

# Create extension file for SAN
$extFile = @"
[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
IP.1 = 127.0.0.1
IP.2 = ::1
"@

$extFile | Out-File -FilePath ".\nginx\ssl\cert.ext" -Encoding ASCII

# Generate self-signed certificate
Write-Host "🏆 Generating self-signed certificate..." -ForegroundColor Blue
& openssl x509 -req -in ".\nginx\ssl\cert.csr" -signkey ".\nginx\ssl\key.pem" -out ".\nginx\ssl\cert.pem" -days 365 -extensions v3_req -extfile ".\nginx\ssl\cert.ext"

# Clean up temporary files
Remove-Item ".\nginx\ssl\cert.csr" -ErrorAction SilentlyContinue
Remove-Item ".\nginx\ssl\cert.ext" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "✅ SSL certificates generated successfully!" -ForegroundColor Green
Write-Host "📁 Certificate files:" -ForegroundColor Cyan
Write-Host "   - Certificate: .\nginx\ssl\cert.pem" -ForegroundColor White
Write-Host "   - Private Key: .\nginx\ssl\key.pem" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  Note: These are self-signed certificates for development only." -ForegroundColor Yellow
Write-Host "   Your browser will show a security warning that you'll need to accept." -ForegroundColor Yellow
Write-Host ""
Write-Host "🚀 You can now start the services with: docker compose up -d" -ForegroundColor Green
Write-Host "🌐 Access Open WebUI at: https://localhost" -ForegroundColor Cyan
