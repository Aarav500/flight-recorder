#!/usr/bin/env bash
# One-time EC2 provisioning for Flight Recorder (Amazon Linux 2023).
# Copy to the instance and run as ec2-user:  bash setup_ec2.sh
set -euo pipefail

echo "[1/4] Installing Docker..."
sudo dnf -y update
sudo dnf -y install docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

echo "[2/4] Ensuring AWS CLI is present (needed to pull from ECR)..."
command -v aws >/dev/null 2>&1 || sudo dnf -y install awscli

echo "[3/4] Creating artifact storage at /opt/flight-recorder/runs..."
sudo mkdir -p /opt/flight-recorder/runs
sudo chown -R ec2-user:ec2-user /opt/flight-recorder

echo "[4/4] Opening host firewall for port 8000 (if firewalld is present)..."
if command -v firewall-cmd >/dev/null 2>&1; then
  sudo firewall-cmd --permanent --add-port=8000/tcp || true
  sudo firewall-cmd --reload || true
fi

cat <<'NOTE'

EC2 base setup complete. Before the first deploy, also ensure:

  * The instance SECURITY GROUP allows inbound TCP 8000 from your IP / 0.0.0.0/0.
    A host firewall alone does not expose the port — set this in the AWS console
    or:  aws ec2 authorize-security-group-ingress \
           --group-id <sg-id> --protocol tcp --port 8000 --cidr 0.0.0.0/0
  * The instance has an IAM role with AmazonEC2ContainerRegistryReadOnly so
    `docker pull` from ECR works without static credentials.
  * Re-login (or run `newgrp docker`) so the docker group membership applies.

After the first `git push` to main, the app is reachable at:
  http://<EC2_HOST>:8000
NOTE
