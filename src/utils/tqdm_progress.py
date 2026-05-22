from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


def _coerce_mininterval(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        mininterval = float(value)
    except (TypeError, ValueError):
        return None
    if mininterval <= 0:
        return None
    return mininterval


class _TqdmConsole:
    def log(self, msg: str) -> None:
        if tqdm is None:
            print(msg, flush=True)
        else:
            tqdm.write(msg)


class TqdmProgress:
    def __init__(self, mininterval_sec: Optional[float] = None) -> None:
        self._bars: Dict[int, Any] = {}
        self._next_task_id = 0
        self._mininterval = _coerce_mininterval(mininterval_sec)
        self.console = _TqdmConsole()

    def __enter__(self) -> "TqdmProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def add_task(self, description: str, total: Optional[int] = None) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        if tqdm is None:
            self._bars[task_id] = None
            return task_id
        kwargs: Dict[str, Any] = {
            "total": total,
            "desc": description,
            "leave": True,
            "position": task_id,
            "dynamic_ncols": True,
        }
        if self._mininterval is not None:
            kwargs["mininterval"] = self._mininterval
        self._bars[task_id] = tqdm(**kwargs)
        return task_id

    def advance(self, task_id: int, advance: int = 1) -> None:
        bar = self._bars.get(task_id)
        if bar is None:
            return
        bar.update(advance)

    def update(
        self,
        task_id: int,
        total: Optional[int] = None,
        completed: Optional[int] = None,
        description: Optional[str] = None,
    ) -> None:
        bar = self._bars.get(task_id)
        if bar is None:
            return
        if description is not None:
            bar.set_description(str(description))
        if total is not None:
            bar.total = total
        if completed is not None:
            bar.n = completed
        bar.refresh()

    def close(self) -> None:
        for bar in self._bars.values():
            if bar is not None:
                bar.close()
        self._bars.clear()


def build_tqdm_progress(mininterval_sec: Optional[float] = None) -> TqdmProgress:
    return TqdmProgress(mininterval_sec=mininterval_sec)
