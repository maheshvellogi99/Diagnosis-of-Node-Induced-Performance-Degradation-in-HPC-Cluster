#!/usr/bin/env bash
# Phase 6 kit setup (D115). RUN ON THE APPROVED TARGET MACHINE ONLY, never on the development Mac.
#
#   HPAS_COMMIT=<40-hex commit> STREAM_SHA256=<sha256 of stream.c> STREAM_ARRAY_SIZE=<n> \
#       bash cluster/setup_target.sh
#
# 1. Refuses unless the host is x86-64 Linux and its hostname is in cluster/approved_hosts.txt.
# 2. Clones HPAS (https://github.com/peaclab/HPAS, BSD-3) into cluster/third_party/HPAS and checks out
#    HPAS_COMMIT; records it in cluster/third_party/HPAS_COMMIT.txt. This git clone is the kit's only
#    network call. HPAS is NOT built here: build it by hand following its README at that commit and
#    write the exact commands into cluster/HPAS_BUILD.md (the build steps are not known in advance).
# 3. STREAM is not downloaded by this script: place McCalpin's stream.c at cluster/third_party/stream.c
#    yourself; the script checks it against STREAM_SHA256 and refuses on a mismatch.
# 4. Builds bin/cpu_work, bin/pchase and bin/stream with gcc -O2.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

fail() { echo "REFUSED: $*" >&2; exit 1; }

[ "$(uname -s)" = "Linux" ] || fail "OS is $(uname -s); Linux only"
[ "$(uname -m)" = "x86_64" ] || fail "architecture is $(uname -m); x86-64 only (HPAS membw)"
host="$(hostname)"; short="${host%%.*}"
grep -v '^\s*#' "$HERE/approved_hosts.txt" 2>/dev/null | grep -qx -e "$host" -e "$short" \
    || fail "host $host is not in cluster/approved_hosts.txt"
[[ "${HPAS_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || fail "set HPAS_COMMIT to a full 40-hex commit"
[[ "${STREAM_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || fail "set STREAM_SHA256"
[[ "${STREAM_ARRAY_SIZE:-}" =~ ^[0-9]+$ ]] || fail "set STREAM_ARRAY_SIZE (arrays well above the caches)"

mkdir -p "$HERE/third_party" "$HERE/bin"
if [ ! -d "$HERE/third_party/HPAS/.git" ]; then
    git clone https://github.com/peaclab/HPAS "$HERE/third_party/HPAS"
fi
git -C "$HERE/third_party/HPAS" checkout --detach "$HPAS_COMMIT"
[ "$(git -C "$HERE/third_party/HPAS" rev-parse HEAD)" = "$HPAS_COMMIT" ] || fail "HPAS checkout"
echo "$HPAS_COMMIT" > "$HERE/third_party/HPAS_COMMIT.txt"

[ -f "$HERE/third_party/stream.c" ] || fail "place stream.c at cluster/third_party/stream.c"
echo "$STREAM_SHA256  $HERE/third_party/stream.c" | sha256sum -c - || fail "stream.c checksum"

gcc -O2 -o "$HERE/bin/cpu_work" "$HERE/src/cpu_work.c"
gcc -O2 -o "$HERE/bin/pchase" "$HERE/src/pchase.c"
gcc -O2 -DSTREAM_ARRAY_SIZE="$STREAM_ARRAY_SIZE" -o "$HERE/bin/stream" "$HERE/third_party/stream.c"
echo "built: $(ls "$HERE/bin")"
echo "next: build HPAS per its README, record the commands in cluster/HPAS_BUILD.md, fill the UNKNOWN"
echo "      values in cluster/campaign.json, commit, then run the dry run before anything else."
