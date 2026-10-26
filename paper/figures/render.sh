#!/usr/bin/env bash
# Usage: render.sh <svg> <out_prefix>   -> <out_prefix>.png (600 dpi) and <out_prefix>.pdf (vector)
set -euo pipefail
SVG=$(realpath "$1"); OUT=$2
read -r W H < <(python3 -c "import re,sys;v=re.search(r'viewBox=\"0 0 ([\d.]+) ([\d.]+)\"',open(sys.argv[1]).read());print(v[1],v[2])" "$SVG")
TMP=$(dirname "$SVG")/_wrap.html
cat > "$TMP" <<EOF
<!doctype html><html><head><meta charset="utf-8"><style>
@page { size: ${W}pt ${H}pt; margin: 0 }
html,body { margin:0; padding:0; background:white }
svg { display:block; width:${W}pt; height:${H}pt }
</style></head><body>
EOF
cat "$SVG" >> "$TMP"
echo "</body></html>" >> "$TMP"
PXW=$(python3 -c "import math;print(math.ceil($W*96/72))"); PXH=$(python3 -c "import math;print(math.ceil($H*96/72))")
google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars --allow-file-access-from-files \
  --force-device-scale-factor=6.25 --window-size=$PXW,$PXH --screenshot="$(realpath -m "$OUT").png" "file://$TMP" 2>/dev/null
google-chrome --headless=new --disable-gpu --no-sandbox --allow-file-access-from-files \
  --no-pdf-header-footer --print-to-pdf="$(realpath -m "$OUT").pdf" "file://$TMP" 2>/dev/null
ls -l "$OUT".png "$OUT".pdf
