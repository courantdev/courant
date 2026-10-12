"""Verified records (TRD REC-1, REC-2; PRD 2.0 P2-REC-1).

A record is one run's answer in a form that travels: the specification, the verdict, who produced
and who requested the run, the Courant version, and the full VER-14 manifest. It re-derives byte
for byte, as a pack does (`check_record`), and a dataset pipeline admits only the records whose
verdict is an `accept` judged against a locked specification (`admit`).

    from courant.record import build_record, check_record, admit
"""

from courant.record.filter import Admission, admit, filter_records
from courant.record.record import (
    SCHEMA,
    RecordError,
    build_record,
    check_record,
    read_record,
    write_record,
)

__all__ = ["SCHEMA", "Admission", "RecordError", "admit", "build_record", "check_record",
           "filter_records", "read_record", "write_record"]
