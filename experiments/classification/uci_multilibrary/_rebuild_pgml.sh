#!/bin/bash
# Rebuild pgml.so inside the docker container and register seeded forest overloads.
set -euo pipefail
SRC=/tmp/pgml_build
rm -rf "$SRC"
mkdir -p "$SRC"
# Mount is read-only; copy sources to a writable location.
cp /pgml/src/c/*.c "$SRC"/
cp /pgml/src/c/*.h "$SRC"/ 2>/dev/null || true
cp /pgml/src/sql/pgml--1.0.sql "$SRC"/
cd "$SRC"
INC="$(pg_config --includedir-server)"
LIBDIR="$(pg_config --pkglibdir)"
gcc -fPIC -shared -O2 -I"$INC" -I"$SRC" pgml.c pgclust.c pgreg.c pgpca.c -o pgml.so -lm -lpthread -Wl,--allow-multiple-definition
cp -f pgml.so "$LIBDIR/pgml.so"
sha256sum "$LIBDIR/pgml.so"
# Install / replace seeded overloads (5-arg symbols keep names; 6-arg are new).
psql -U postgres -d ml_test -v ON_ERROR_STOP=1 <<'SQL'
CREATE OR REPLACE FUNCTION pgml_train_forest(float4[][], int[], int, int, int)
RETURNS bytea AS 'pgml', 'pgml_train_forest' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_train_forest(float4[][], int[], int, int, int, int)
RETURNS bytea AS 'pgml', 'pgml_train_forest_seeded' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_train_forest_fast(float4[][], int[], int, int, int)
RETURNS bytea AS 'pgml', 'pgml_train_forest_fast' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION pgml_train_forest_fast(float4[][], int[], int, int, int, int)
RETURNS bytea AS 'pgml', 'pgml_train_forest_fast_seeded' LANGUAGE C STRICT;
SELECT proname, pronargs FROM pg_proc WHERE proname LIKE 'pgml_train_forest%' ORDER BY 1, 2;
SQL
echo "rebuild_ok"
