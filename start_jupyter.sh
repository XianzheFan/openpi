set -a
source .env
set +a
LOCAL_PORT=${1:-9118}
PARTITION=${2:-interactive}
submit_job --gpu 8 --tasks_per_node 1 --nodes 1 -n jupyter_lab --image $TRAINING_IMAGE_PATH \
        --logroot work_dirs/jupyter_lab_log \
        --email_mode never \
        --partition $PARTITION \
        --duration 0 \
        --dependent_clones 0 \
        -c "bash tools/nv_cluster_tools/jupyter_script_on_server.sh"
sleep 60

# Path to the IP address file
IP_FILE=$OPENPI_CODEBASE_PATH/.server_ip_eagle.txt

# Check if the file exists
if [ ! -f "$IP_FILE" ]; then
  echo "Error: IP address file not found at $IP_FILE"
  exit 1
fi

# Read the IP address from the file
SERVER_IP=$(cat "$IP_FILE" | tr -d '[:space:]')

# Validate IP address format (optional)
if [[ ! $SERVER_IP =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: Invalid IP address format in $IP_FILE"
  exit 1
fi

# Target port and local listening port
TARGET_PORT=8888

# Print information
echo "Setting up socat to forward local port $LOCAL_PORT to $SERVER_IP:$TARGET_PORT"

# Start socat
socat TCP-LISTEN:$LOCAL_PORT,fork TCP:$SERVER_IP:$TARGET_PORT 