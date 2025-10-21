"""Batch export job orchestration helpers.

Phase 2 introduces a synchronous job runner that executes export work items
serially, tracks structured results, and surfaces progress updates via an
optional callback. The implementation intentionally keeps the execution
model simple so future phases can extend it with background workers or
parallelism without changing the public contract.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from threading import Event, Lock
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Sequence, Dict, Any
import traceback

__all__ = [
    "ExportResult",
    "Job",
    "JobItem",
    "JobState",
    "JobStatus",
]

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    """Lifecycle state for a job."""

    PENDING = "pending"
    RUNNING = "running"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


@dataclass(frozen=True)
class ExportResult:
    """Structured outcome for a single export item."""

    item_id: str
    ok: bool
    output_path: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - simple data normalisation
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class JobItem:
    """Callable work unit executed by :class:`Job.start`."""

    item_id: str
    execute: Callable[[], Optional[Mapping[str, Any]]]
    output_path: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - simple data normalisation
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class JobStatus:
    """Snapshot of the current job state and accumulated results."""

    state: JobState
    total: int
    completed: int
    succeeded: int
    failed: int
    cancelled: bool
    results: Mapping[str, ExportResult]
    current: Optional[str] = None

    def __post_init__(self) -> None:  # pragma: no cover - simple data normalisation
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))


class Job:
    """Serial batch export runner.

    Parameters
    ----------
    mode:
        Descriptive export mode (for logging/metadata only).
    items:
        Sequence of :class:`JobItem` entries.
    marker_set:
        The marker set name associated with the export.
    output_dir:
        Directory where generated files are persisted.
    file_format:
        File extension for rendered outputs.
    overrides:
        Optional mapping capturing configuration overrides applied to every
        item (for example DPI or downsample factor).
    progress_callback:
        Optional callable receiving :class:`JobStatus` snapshots after each
        item completes and when the job finishes.
    """

    def __init__(
        self,
        *,
        mode: str,
        items: Sequence[JobItem],
        marker_set: str,
        output_dir: str,
        file_format: str,
        overrides: Optional[Mapping[str, Any]] = None,
        progress_callback: Optional[Callable[[JobStatus], None]] = None,
    ) -> None:
        self.mode = mode
        self.marker_set = marker_set
        self.output_dir = output_dir
        self.file_format = file_format
        self.overrides = MappingProxyType(dict(overrides or {}))
        self._items = tuple(items)
        self._total = len(self._items)
        self._progress_callback = progress_callback

        self._state = JobState.PENDING
        self._results: "OrderedDict[str, ExportResult]" = OrderedDict()
        self._lock = Lock()
        self._cancel_event = Event()
        self._current: Optional[str] = None
        self._succeeded = 0
        self._failed = 0

    def cancel(self) -> None:
        """Request cancellation for the job.

        Cancellation is cooperative; the currently running item is allowed to
        finish. Subsequent items will be skipped once the request is observed.
        """

        self._cancel_event.set()

    def start(self) -> JobStatus:
        """Execute the job synchronously and return the final status."""

        with self._lock:
            if self._state is not JobState.PENDING:
                raise RuntimeError("Job has already been started")
            self._state = JobState.RUNNING

        if self._cancel_event.is_set():
            with self._lock:
                self._state = JobState.CANCELLED
                self._current = None
            final_status = self.status()
            self._notify_progress(final_status)
            return final_status

        logger.info(
            "Starting export job mode=%s marker_set=%s total_items=%d",
            self.mode,
            self.marker_set,
            self._total,
        )

        for item in self._items:
            if self._cancel_event.is_set():
                break

            with self._lock:
                self._current = item.item_id

            try:
                payload = item.execute()
            except Exception as exc:  # pragma: no cover - exercised in tests
                self._failed += 1
                result = ExportResult(
                    item_id=item.item_id,
                    ok=False,
                    output_path=item.output_path,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                    metadata=item.metadata,
                )
                logger.exception("Export job item failed: item_id=%s", item.item_id)
            else:
                self._succeeded += 1
                result = self._coerce_success_result(item, payload)

            with self._lock:
                self._results[item.item_id] = result

            self._notify_progress()

        with self._lock:
            if self._cancel_event.is_set() and len(self._results) < self._total:
                self._state = JobState.CANCELLED
            else:
                self._state = JobState.COMPLETED
            self._current = None

        final_status = self.status()
        logger.info(
            "Export job finished state=%s succeeded=%d failed=%d",
            final_status.state,
            final_status.succeeded,
            final_status.failed,
        )
        self._notify_progress(final_status)
        return final_status

    def status(self) -> JobStatus:
        """Return a snapshot of the current job status."""

        with self._lock:
            results_snapshot = dict(self._results)
            state = self._state
            succeeded = self._succeeded
            failed = self._failed
            current = self._current
            total = self._total
            cancelled = state is JobState.CANCELLED

        completed = len(results_snapshot)
        return JobStatus(
            state=state,
            total=total,
            completed=completed,
            succeeded=succeeded,
            failed=failed,
            cancelled=cancelled,
            results=results_snapshot,
            current=current,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coerce_success_result(
        self,
        item: JobItem,
        payload: Optional[Mapping[str, Any]],
    ) -> ExportResult:
        metadata: Dict[str, Any] = dict(item.metadata)
        output_path = item.output_path

        if payload:
            if "output_path" in payload:
                output_path = str(payload["output_path"])
            metadata_payload = payload.get("metadata")
            if isinstance(metadata_payload, Mapping):
                metadata.update(metadata_payload)

        return ExportResult(
            item_id=item.item_id,
            ok=True,
            output_path=output_path,
            metadata=metadata,
        )

    def _notify_progress(self, status: Optional[JobStatus] = None) -> None:
        callback = self._progress_callback
        if callback is None:
            return
        if status is None:
            status = self.status()
        try:
            callback(status)
        except Exception:  # pragma: no cover - defensive logging only
            logger.exception("Progress callback raised an exception")


def __getattr__(name: str) -> Any:
    """Lazy attribute re-export to support ``from job import Job`` style usage."""

    if name in __all__:
        return globals()[name]
    raise AttributeError(name)
