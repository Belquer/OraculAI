#!/usr/bin/env bash
set -euo pipefail

# render_deploy.sh
# Usage: export RENDER_API_KEY="..." GHCR_PAT="..." RENDER_SERVICE_ID="srv-..." && ./render_deploy.sh

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required. Install curl and retry."
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required. On macOS: brew install jq"
  exit 2
fi

: "Ensure env vars are set"
: ${RENDER_API_KEY:?Need to set RENDER_API_KEY}
: ${GHCR_PAT:?Need to set GHCR_PAT}
: ${RENDER_SERVICE_ID:?Need to set RENDER_SERVICE_ID}

API_BASE="https://api.render.com/v1"
# Allow override for the registry credential name so user can supply the existing one (e.g. GHCR_render)
: ${RENDER_CREDS_NAME:=ghcr-belquer}
CREDS_NAME="$RENDER_CREDS_NAME"
GHCR_SERVER="ghcr.io"
GHCR_USERNAME="Belquer"
DOCKER_REPO="ghcr.io/belquer/oraculai:latest"

echo "Looking for existing registry credential named '$CREDS_NAME'..."
create_resp=""
existing=$(curl -sS -H "Authorization: Bearer $RENDER_API_KEY" "$API_BASE/registry-credentials" | jq -r '.[] | select(.name=="'"$CREDS_NAME"'" ) | @json' ) || true
if [ -n "$existing" ] && [ "$existing" != "null" ]; then
  echo "Found existing credential with name $CREDS_NAME"
  cred_id=$(jq -r '.id' <<< "$existing")
else
  echo "Creating registry credential '$CREDS_NAME'..."
  create_resp=$(curl -sS -X POST -H "Authorization: Bearer $RENDER_API_KEY" -H "Content-Type: application/json" \
    -d '{"name":"'"$CREDS_NAME"'","server":"'"$GHCR_SERVER"'","username":"'"$GHCR_USERNAME"'","password":"'"$GHCR_PAT"'}' \
    "$API_BASE/registry-credentials")

  # Print response (prettified if possible)
  echo "Create response:"
  if jq -e . >/dev/null 2>&1 <<<"$create_resp"; then
    jq -r '.' <<< "$create_resp"
  else
    echo "$create_resp"
  fi

  cred_id=$(jq -r '.id // empty' <<< "$create_resp")
  if [ -z "$cred_id" ]; then
    echo "Failed to create registry credential. Response above. Exiting."
    exit 3
  fi
fi
echo "Using registry credential id: $cred_id"

echo "Patching service $RENDER_SERVICE_ID to use image $DOCKER_REPO and registry credential..."
patch_resp=$(curl -sS -X PATCH -H "Authorization: Bearer $RENDER_API_KEY" -H "Content-Type: application/json" \
  -d '{"dockerRepository":"'"$DOCKER_REPO"'","registryCredentialId":"'"$cred_id"'"}' \
  "$API_BASE/services/$RENDER_SERVICE_ID")

echo "Patch response:"
if jq -e . >/dev/null 2>&1 <<<"$patch_resp"; then
  jq -r '.' <<< "$patch_resp"
else
  echo "$patch_resp"
fi

# Trigger a deploy

echo "Triggering deploy..."
deploy_resp=$(curl -sS -X POST -H "Authorization: Bearer $RENDER_API_KEY" -H "Content-Type: application/json" \
  -d '{"serviceId":"'"$RENDER_SERVICE_ID"'"}' "$API_BASE/deploys")

echo "Deploy response:"
if jq -e . >/dev/null 2>&1 <<<"$deploy_resp"; then
  jq -r '.' <<< "$deploy_resp"
else
  echo "$deploy_resp"
fi

deploy_id=$(jq -r '.id // empty' <<< "$deploy_resp")
if [ -z "$deploy_id" ]; then
  echo "Deploy trigger failed. Exiting."
  exit 4
fi

echo "Deploy started: $deploy_id"

echo "Polling deploy status for up to 5 minutes..."
end=$((SECONDS + 300))
while [ $SECONDS -lt $end ]; do
  dstatus=$(curl -sS -H "Authorization: Bearer $RENDER_API_KEY" "$API_BASE/deploys/$deploy_id" | jq -r '.status // empty') || true
  echo "Status: $dstatus"
  if [ "$dstatus" = "success" ] || [ "$dstatus" = "failed" ] || [ "$dstatus" = "cancelled" ]; then
    break
  fi
  sleep 5
done

# Final health check
SERVICE_URL=$(curl -sS -H "Authorization: Bearer $RENDER_API_KEY" "$API_BASE/services/$RENDER_SERVICE_ID" | jq -r '.serviceDetails.url // .url // empty')
if [ -n "$SERVICE_URL" ]; then
  echo "Service URL: $SERVICE_URL"
  echo "Polling $SERVICE_URL/health for index_present..."
  end2=$((SECONDS + 300))
  while [ $SECONDS -lt $end2 ]; do
    h=$(curl -sS "$SERVICE_URL/health" || true)
    echo "health: $h"
    if echo "$h" | jq -e '.index_present == true' >/dev/null 2>&1; then
      echo "Index present. Done."
      exit 0
    fi
    sleep 5
  done
  echo "Timeout waiting for index_present in /health."
else
  echo "Could not read service URL from Render API response."
fi

exit 0