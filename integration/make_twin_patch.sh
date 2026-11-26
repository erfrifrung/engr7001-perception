#!/usr/bin/env bash
# Rebuild integration/twin_bridge_v31.patch against an UNPATCHED copy of Hazem's digital_twin_v5, then check it.
#   TWIN=~/digital_twin_v5 bash integration/make_twin_patch.sh
# Files carried by the patch (WP3-side changes only):
#   ros/wp3_scene_bridge.py, ros/wp3_link.py        <- integration/
#   ros/start_perception.sh, twin.sh                <- integration/twin_v5_files/  (measured table frame by default)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TWIN:?set TWIN to an unpatched digital_twin_v5 directory}"
W=$(mktemp -d); mkdir -p "$W/a/ros" "$W/b/ros"
for f in ros/wp3_scene_bridge.py ros/wp3_link.py ros/start_perception.sh twin.sh; do
  cp "$TWIN/$f" "$W/a/$f"
done
cp "$HERE/wp3_scene_bridge.py" "$HERE/wp3_link.py" "$W/b/ros/"
cp "$HERE/twin_v5_files/ros/start_perception.sh" "$W/b/ros/"; cp "$HERE/twin_v5_files/twin.sh" "$W/b/"
(cd "$W" && git diff --no-index a b > "$HERE/twin_bridge_v31.patch") || true
sed -i -e 's#a/a/#a/#g' -e 's#b/b/#b/#g' "$HERE/twin_bridge_v31.patch"     # --no-index prefixes the dir names
C=$(mktemp -d); cp -r "$TWIN/." "$C/"
(cd "$C" && git apply --check "$HERE/twin_bridge_v31.patch" && git apply "$HERE/twin_bridge_v31.patch")
cmp "$C/ros/wp3_scene_bridge.py" "$HERE/wp3_scene_bridge.py"
cmp "$C/ros/start_perception.sh" "$HERE/twin_v5_files/ros/start_perception.sh"
cmp "$C/twin.sh" "$HERE/twin_v5_files/twin.sh"
echo "patch OK: $(grep -c '^diff --git' "$HERE/twin_bridge_v31.patch") files, applies to $TWIN and reproduces integration/"
rm -rf "$W" "$C"
