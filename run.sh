#!/bin/bash

# ==============================================================================
# WORKSHOP: From Notebook to Cloud
# Entrypoint Script for Satellite Monitoring PoC
# ==============================================================================

# 1. Path Configurations (Reflecting Docker structure)
ROOT="/app"
VERSION="6"

# 2. Model Parameters (Bands and Indices used in the U-Net)
OPTICAL_BANDS="swir1,nir,red,green"
OPTICAL_INDICES="NDVI,MNDWI"


# 3. Fixed Date for Proof of Concept (Workshop)
# For the workshop PoC, we focus on a specific date to ensure consistency
START_DATE="2026-01-13"
END_DATE="2026-01-14"

# Format for folder naming (YYYYMMDD)
FOLDER_DATE=$(echo "$START_DATE" | sed 's/-//g')

# Define input and output paths
MOSAIC_PATH=$ROOT/daily_mosaic/$FOLDER_DATE
OUTPUT_PATH=$ROOT/output/$FOLDER_DATE

echo "--------------------------------------------------------"
echo " STARTING INFERENCE - GDE WORKSHOP"
echo " Target Date: $START_DATE"
echo " Output Folder: $OUTPUT_PATH"
echo "--------------------------------------------------------"

# 4. Python Execution
# -u: Forces unbuffered output (essential for real-time Cloud Logging)
python3 -u $ROOT/prediction.py "$VERSION" "$OPTICAL_BANDS" "$OPTICAL_INDICES" "$MOSAIC_PATH" "$OUTPUT_PATH" "$START_DATE" "$END_DATE" "$ROOT"

echo "--------------------------------------------------------"
echo "Script finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "--------------------------------------------------------"