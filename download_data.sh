#!/bin/bash
# download_data.sh - Fetch and unpack large dataset from Dropbox into a local directory
# for run it ./download_data.sh

# Exit the script immediately if any command fails (e.g., network error or missing unzip)
set -e

# Local directory where data will be downloaded and extracted
DATA_DIR="data"

# Name of the Working Dir (local) ZIP file to store the downloaded data
WD_ZIP_FILE_NAME="DataSetsForTesting.zip"

# Dropbox direct download link (ensure it ends with dl=1 to force download)
DROPBOX_URL="https://www.dropbox.com/scl/fi/hgxzlc4g0ur8hg6fnr0jc/ejemplospythonaq.zip?rlkey=a4eugk0n85ug6xlay1t3zheom&dl=1"

echo "📁 Preparing download directory: $DATA_DIR"
# Create the data directory if it doesn't exist
mkdir -p "$DATA_DIR"

echo "⬇️  Downloading dataset from Dropbox..."
# Use curl with -L to follow redirects (Dropbox requires this)
# Save the output to $DATA_DIR/$WD_ZIP_FILE_NAME
curl -L "$DROPBOX_URL" -o "$DATA_DIR/$WD_ZIP_FILE_NAME"

echo "📦 Unzipping dataset into $DATA_DIR..."
# Extract the contents of the ZIP file into the data directory quietly (-q)
unzip -q "$DATA_DIR/$WD_ZIP_FILE_NAME" -d "$DATA_DIR"

echo "🧹 Cleaning up temporary ZIP file..."
# Delete the downloaded ZIP file to save space
rm "$DATA_DIR/$WD_ZIP_FILE_NAME"

echo "✅ Data successfully downloaded and extracted to $DATA_DIR"