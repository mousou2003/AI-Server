#!/bin/bash

# Script to generate self-signed SSL certificates for development
# Run this script to create SSL certificates for HTTPS

echo "🔐 Generating self-signed SSL certificates..."

# Create SSL directory if it doesn't exist
mkdir -p ./nginx/ssl

# Generate private key
openssl genrsa -out ./nginx/ssl/key.pem 2048

# Generate certificate signing request
openssl req -new -key ./nginx/ssl/key.pem -out ./nginx/ssl/cert.csr -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"

# Generate self-signed certificate
openssl x509 -req -in ./nginx/ssl/cert.csr -signkey ./nginx/ssl/key.pem -out ./nginx/ssl/cert.pem -days 365 -extensions v3_req -extfile <(echo "[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
IP.1 = 127.0.0.1
IP.2 = ::1")

# Set proper permissions
chmod 600 ./nginx/ssl/key.pem
chmod 644 ./nginx/ssl/cert.pem

# Clean up CSR file
rm ./nginx/ssl/cert.csr

echo "✅ SSL certificates generated successfully!"
echo "📁 Certificate files:"
echo "   - Certificate: ./nginx/ssl/cert.pem"
echo "   - Private Key: ./nginx/ssl/key.pem"
echo ""
echo "⚠️  Note: These are self-signed certificates for development only."
echo "   Your browser will show a security warning that you'll need to accept."
echo ""
echo "🚀 You can now start the services with: docker compose up -d"
echo "🌐 Access Open WebUI at: https://localhost"
