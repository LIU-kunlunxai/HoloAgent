#!/bin/bash
set -e

#========================================================
# HMSG 离线建图脚本（SSH 断开安全）
# 用法: bash run_mapping.sh [config_name]
# 示例: bash run_mapping.sh semantic_scene_reconstruction_recorded
#========================================================

CONFIG_NAME="${1:-semantic_scene_reconstruction_recorded}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_${TIMESTAMP}.log"

echo "========================================="
echo "  HMSG Mapping"
echo "  Config : ${CONFIG_NAME}"
echo "  Log    : ${LOG_FILE}"
echo "========================================="

cd "${SCRIPT_DIR}"

nohup python -u application/semantic_scene_reconstrucion_offline/semantic_scene_reconstruction.py \
    --config-name="${CONFIG_NAME}" \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "PID: ${PID}"
echo "查看日志: tail -f ${LOG_FILE}"
echo "查看进程: ps -p ${PID}"
