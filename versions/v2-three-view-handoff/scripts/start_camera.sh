#!/usr/bin/env bash
set -eo pipefail

usage() {
    echo "Usage: bash scripts/start_camera.sh wrist|tripod|third CAMERA_SERIAL" >&2
    exit 2
}

[[ $# -eq 2 ]] || usage
case "$1" in
    wrist|tripod|third) ;;
    *) usage ;;
esac
serial=$2
[[ $serial =~ ^[0-9]+$ ]] || usage
source "$(dirname -- "${BASH_SOURCE[0]}")/lab_env.sh"
exec ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=perception "camera_name:=$1" "serial_no:=_$serial" \
    rgb_camera.color_profile:=1280x720x15 depth_module.depth_profile:=1280x720x15 \
    enable_infra1:=false enable_infra2:=false align_depth.enable:=true \
    enable_sync:=true pointcloud.enable:=false
