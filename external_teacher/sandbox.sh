#!/usr/bin/env bash
# Run the original SandboxFusion image bound to localhost; coding training/preparation use this endpoint.
set -euo pipefail
exec docker run --rm --name "aopd-sandbox-${USER}" \
  -p 127.0.0.1:8080:8080 volcengine/sandbox-fusion:server-20250609
