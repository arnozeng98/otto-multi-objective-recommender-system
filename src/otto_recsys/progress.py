from __future__ import annotations

from typing import Protocol, Self

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ProgressReporter(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> None: ...

    def start_stage(
        self,
        step: int,
        total_steps: int,
        name: str,
        *,
        total: int | None,
        unit: str,
    ) -> None: ...

    def advance(self, amount: int = 1) -> None: ...

    def finish(self, status: str) -> None: ...


class NullProgress:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def start_stage(
        self,
        step: int,
        total_steps: int,
        name: str,
        *,
        total: int | None,
        unit: str,
    ) -> None:
        return None

    def advance(self, amount: int = 1) -> None:
        return None

    def finish(self, status: str) -> None:
        return None


class RichPipelineProgress:
    def __init__(self) -> None:
        self.console = Console(stderr=True)
        self.progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[unit]}"),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
            transient=False,
        )
        self.task_id: TaskID | None = None

    def __enter__(self) -> Self:
        self.progress.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.progress.stop()

    def start_stage(
        self,
        step: int,
        total_steps: int,
        name: str,
        *,
        total: int | None,
        unit: str,
    ) -> None:
        self.task_id = self.progress.add_task(
            f"Step {step}/{total_steps}  {name}",
            total=total,
            unit=unit,
        )

    def advance(self, amount: int = 1) -> None:
        if self.task_id is not None:
            self.progress.advance(self.task_id, amount)

    def finish(self, status: str) -> None:
        if self.task_id is None:
            return
        task = self.progress.tasks[self.task_id]
        if task.total is not None:
            self.progress.update(self.task_id, completed=task.total)
        self.progress.update(
            self.task_id,
            description=f"{task.description} [green]{status}[/green]",
        )


def create_progress(enabled: bool) -> ProgressReporter:
    return RichPipelineProgress() if enabled else NullProgress()
