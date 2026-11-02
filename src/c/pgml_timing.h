/* Lightweight stage timing for pgml train wrappers.
 * Emits a single NOTICE line that experiments parse:
 *   PGML_STAGE_TIMING convert_ms=... native_ms=... serialize_ms=...
 */
#ifndef PGML_TIMING_H
#define PGML_TIMING_H

#include "postgres.h"
#include "portability/instr_time.h"

typedef struct PgmlStageTimer
{
	instr_time t0;
	instr_time t1;
	double convert_ms;
	double native_ms;
	double serialize_ms;
} PgmlStageTimer;

static inline void
pgml_stage_begin(PgmlStageTimer *st)
{
	st->convert_ms = 0;
	st->native_ms = 0;
	st->serialize_ms = 0;
	INSTR_TIME_SET_CURRENT(st->t0);
}

static inline double
pgml_stage_lap_ms(PgmlStageTimer *st)
{
	INSTR_TIME_SET_CURRENT(st->t1);
	INSTR_TIME_SUBTRACT(st->t1, st->t0);
	double ms = INSTR_TIME_GET_MILLISEC(st->t1);
	INSTR_TIME_SET_CURRENT(st->t0);
	return ms;
}

static inline void
pgml_stage_emit(const PgmlStageTimer *st)
{
	elog(NOTICE,
		 "PGML_STAGE_TIMING convert_ms=%.6f native_ms=%.6f serialize_ms=%.6f",
		 st->convert_ms, st->native_ms, st->serialize_ms);
}

#endif /* PGML_TIMING_H */
