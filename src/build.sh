#!/bin/bash
# Build unified pgml.so from the four C modules.
set -euo pipefail
cd "$(dirname "$0")/c"
if command -v pg_config >/dev/null 2>&1; then
  HDR="$(pg_config --includedir-server)"
else
  HDR=/tmp/pgdev/usr/include/postgresql/17/server
fi
if [ ! -f "$HDR/postgres.h" ]; then
  echo "PostgreSQL server headers not found at $HDR" >&2
  echo "Install postgresql-server-dev or unpack the headers and set HDR." >&2
  exit 1
fi
gcc -fPIC -shared -O2 -I"$HDR" pgml.c pgclust.c pgreg.c pgpca.c \
  -o pgml.so -lm -lpthread -Wl,--allow-multiple-definition
echo "built $(pwd)/pgml.so"
