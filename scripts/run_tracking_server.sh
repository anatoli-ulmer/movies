PWD=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLRUN_DIR="$SCRIPT_DIR/../mlruns"

# Create the mlruns directory if it doesn't exist
mkdir -p "$MLRUN_DIR"

cd $SCRIPT_DIR

mlflow server \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-store-uri file://$MLRUN_DIR \
  --default-artifact-root file://$MLRUN_DIR \
  --serve-artifacts

cd $PWD