# PGXS build for the unified pgml extension (requires postgresql-server-dev).
# Combines all four C modules (classification, clustering, regression, PCA)
# into a single pgml.so. --allow-multiple-definition dedups the four identical
# PG_MODULE_MAGIC symbols.
EXTENSION = pgml
MODULE_big = pgml
OBJS = pgml.o pgclust.o pgreg.o pgpca.o
DATA = pgml--1.0.sql
SHLIB_LINK += -lm

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
