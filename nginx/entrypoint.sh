#!/bin/sh

CERT_DIR=/etc/nginx/certs
CERT=$CERT_DIR/cert.pem
KEY=$CERT_DIR/key.pem

mkdir -p $CERT_DIR

if [ ! -f "$CERT" ] || [ ! -f "$KEY" ]; then
  echo "Generating self-signed TLS certificate..."

  openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout $KEY \
  -out $CERT \
  -subj "/C=NL/ST=NH/L=Amsterdam/O=RAG/OU=Dev/CN=localhost"

fi

echo "Starting nginx..."

nginx -g "daemon off;"