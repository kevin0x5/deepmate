"""Workspace JSONL store for scheduled jobs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from deepmate.cron.model import CronJob, job_path
from deepmate.storage.atomic import file_lock


@dataclass(frozen=True, slots=True)
class CronJobLoadIssue:
    """One user-editable JSONL line that could not be loaded as a cron job."""

    line_number: int
    message: str


@dataclass(frozen=True, slots=True)
class CronJobLoadReport:
    """Loaded cron jobs plus recoverable JSONL diagnostics."""

    jobs: tuple[CronJob, ...]
    issues: tuple[CronJobLoadIssue, ...]


@dataclass(frozen=True, slots=True)
class _CronJobLine:
    raw: str
    job: CronJob | None = None
    issue: CronJobLoadIssue | None = None


class CronJobStore:
    """Read and update one workspace's cron/jobs.jsonl file."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.path = job_path(self.workspace)

    def ensure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def load(self) -> tuple[CronJob, ...]:
        if not self.path.exists():
            return ()
        return self.load_report().jobs

    def load_report(self) -> CronJobLoadReport:
        if not self.path.exists():
            return CronJobLoadReport(jobs=(), issues=())
        lines = self._read_lines_unlocked()
        return CronJobLoadReport(
            jobs=tuple(line.job for line in lines if line.job is not None),
            issues=tuple(line.issue for line in lines if line.issue is not None),
        )

    def _read_lines_unlocked(self) -> tuple[_CronJobLine, ...]:
        if not self.path.exists():
            return ()
        lines: list[_CronJobLine] = []
        for line_number, raw in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw.strip()
            if not line or line.startswith("#"):
                lines.append(_CronJobLine(raw=raw))
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                lines.append(
                    _CronJobLine(
                        raw=raw,
                        issue=CronJobLoadIssue(
                            line_number=line_number,
                            message=f"invalid JSON: {exc.msg}",
                        ),
                    )
                )
                continue
            if not isinstance(payload, dict):
                lines.append(
                    _CronJobLine(
                        raw=raw,
                        issue=CronJobLoadIssue(
                            line_number=line_number,
                            message="expected a JSON object",
                        ),
                    )
                )
                continue
            try:
                job = CronJob.from_mapping(payload)
            except (TypeError, ValueError) as exc:
                lines.append(
                    _CronJobLine(
                        raw=raw,
                        issue=CronJobLoadIssue(
                            line_number=line_number,
                            message=f"invalid cron job: {exc}",
                        ),
                    )
                )
                continue
            if not job.id:
                lines.append(
                    _CronJobLine(
                        raw=raw,
                        issue=CronJobLoadIssue(
                            line_number=line_number,
                            message="missing job id",
                        ),
                    )
                )
                continue
            if not job.job.prompt:
                lines.append(
                    _CronJobLine(
                        raw=raw,
                        issue=CronJobLoadIssue(
                            line_number=line_number,
                            message="missing job.prompt",
                        ),
                    )
                )
                continue
            lines.append(_CronJobLine(raw=raw, job=job))
        return tuple(lines)

    def get(self, job_id: str) -> CronJob:
        clean = job_id.strip()
        for job in self.load():
            if job.id == clean:
                return job
        raise ValueError(f"cron job not found: {clean}")

    def save(self, job: CronJob) -> None:
        with file_lock(self.path):
            lines = list(self._read_lines_unlocked())
            replaced = False
            output_lines: list[str] = []
            for line in lines:
                if line.job is not None and line.job.id == job.id and not replaced:
                    output_lines.append(_job_line(job))
                    replaced = True
                    continue
                output_lines.append(line.raw)
            if not replaced:
                output_lines.append(_job_line(job))
            self._write_lines_unlocked(output_lines)

    def remove(self, job_id: str) -> bool:
        clean = job_id.strip()
        with file_lock(self.path):
            lines = list(self._read_lines_unlocked())
            removed = False
            output_lines: list[str] = []
            for line in lines:
                if line.job is not None and line.job.id == clean:
                    removed = True
                    continue
                output_lines.append(line.raw)
            if not removed:
                return False
            self._write_lines_unlocked(output_lines)
            return True

    def _write_lines_unlocked(self, lines: Iterable[str]) -> None:
        content = "\n".join(lines)
        if content:
            content += "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, self.path)


def _job_line(job: CronJob) -> str:
    return json.dumps(job.to_mapping(), ensure_ascii=False, separators=(",", ":"))
