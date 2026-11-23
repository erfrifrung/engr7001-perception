# Source this file in each terminal running the lab camera, TF, WP3 or RViz processes.
WP3_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-67}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$WP3_ROOT/ros2_ws/src/wp3_live_perception/config/fastdds_shm_profile.xml"
export PYTHONPATH="$WP3_ROOT:${PYTHONPATH:-}"
