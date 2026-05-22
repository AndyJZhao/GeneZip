import hydra
import logging
import math
import numbers
import wandb
from codetiming import Timer
from contextlib import ContextDecorator
from datetime import datetime
from functools import wraps
from humanfriendly import format_timespan
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import pretty_repr
from types import SimpleNamespace
import rich


class _FormattedMetric:
    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text

    __str__ = __repr__


class RankFilter(logging.Filter):
    """Filter logs so only rank==0 records are emitted."""

    def __init__(self, rank=0):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        return self.rank == 0


logger = rich_logger = logging.getLogger()
rich_handler = RichHandler(
    rich_tracebacks=False,
    tracebacks_suppress=[hydra],
    console=Console(width=165),
    enable_link_path=False,
)
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[rich_handler],
)


class ExpLogger:
    """
    Minimal logger with rich console output and optional wandb plumbing.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.use_wandb = cfg.use_wandb
        self.rank = cfg.get("rank", 0)

        # Initialize rich logger
        self.logger = rich_logger
        self.console = rich_handler.console
        self.logger.setLevel(getattr(logging, cfg.logging.level.upper()))

        for f in self.logger.filters[:]:
            if isinstance(f, RankFilter):
                self.logger.removeFilter(f)
        self.logger.info(f"Initializing Logger at Rank={self.rank}")
        self.logger.addFilter(RankFilter(self.rank))

        pass_func = lambda *args, **kwargs: None
        self.rule = self.console.rule if self.rank == 0 else pass_func
        self.print = rich.print if self.rank == 0 else pass_func
        self.pprint = rich.pretty.pprint if self.rank == 0 else pass_func

        self.info = self.logger.info
        self.critical = self.logger.critical
        self.warning = self.logger.warning
        self.debug = self.logger.debug
        self.error = self.logger.error
        self.exception = self.logger.exception

        self.results = {}
        self.summary = {}

    @staticmethod
    def _coerce_metric_value(value):
        if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
            try:
                return value.item()
            except Exception:
                return value
        return value

    @classmethod
    def _format_metric_value(cls, value):
        value = cls._coerce_metric_value(value)
        if isinstance(value, bool):
            return value
        if isinstance(value, numbers.Real):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return value
            if not math.isfinite(numeric_value):
                return _FormattedMetric(str(value))
            abs_value = abs(numeric_value)
            if abs_value >= 1e6 or (0.0 < abs_value < 1e-6):
                return _FormattedMetric(f"{numeric_value:.4e}")
            return _FormattedMetric(f"{numeric_value:.4f}")
        return value

    def clean_metrics_for_display(self, metrics):
        if not isinstance(metrics, dict):
            return metrics
        return {key: self._format_metric_value(value) for key, value in metrics.items()}

    def log(self, *args, level="", **kwargs):
        if self.rank == 0:
            self.logger.log(getattr(logging, level.upper(), logging.INFO), *args, **kwargs)

    def log_metrics(self, metrics, step=None, level="info", use_pretty_repr=False):
        """Log metrics to Wandb if enabled, otherwise stdout."""
        if not isinstance(metrics, dict):
            return
        payload = dict(metrics)
        if step is not None:
            payload.setdefault("step", step)
        if self.use_wandb and self.rank == 0:
            wandb.log(payload, step=step)
        if (not self.use_wandb or self.cfg.logging.log_wandb_metric_to_stdout) and self.rank == 0:
            display_payload = self.clean_metrics_for_display(payload)
            self.log(pretty_repr(display_payload) if use_pretty_repr else display_payload, level=level)
        for key, value in payload.items():
            self.results.setdefault(key, []).append(value)

    def load_and_log_previous_metrics(self, previous_results, wandb_log=True):
        if self.results:
            self.warning(f"The current result is {self.results}. Overwriting with previous results.")
        self.results = self._normalize_results(previous_results)

        if wandb_log and self.use_wandb and self.results and self.rank == 0:
            max_steps = self.cfg.get("max_steps", 1e9)
            for metrics in self._iter_result_dicts(self.results):
                if metrics.get("step", 0) > max_steps:
                    break
                wandb.log(metrics)
            last_metrics = self._last_result(self.results)
            if last_metrics:
                self.critical(f"Resumed from previous checkpoints {last_metrics}")

    def update_summary(self, summary):
        if not isinstance(summary, dict):
            return
        self.summary.update(summary)
        if self.use_wandb and self.rank == 0:
            wandb.summary.update(summary)

    def wandb_summary_update(self, result):
        self.update_summary(result)

    def wandb_config_update(self, config_updates):
        if self.use_wandb and self.rank == 0:
            wandb.config.update(config_updates, allow_val_change=True)

    def save_file_to_wandb(self, file, base_path, policy="now", **kwargs):
        if self.use_wandb and self.rank == 0:
            wandb.save(file, base_path=base_path, policy=policy, **kwargs)

    @property
    def experiment(self):
        if not hasattr(self, "_experiment"):
            if self.use_wandb and self.rank == 0:
                self._experiment = wandb.run
            else:
                self._experiment = SimpleNamespace(log=self.logger.info, id=self.cfg.uid, name=self.cfg.alias)
        return self._experiment

    def wandb_finish(self, result=None):
        if self.use_wandb and self.rank == 0:
            wandb.summary.update(result or {})
            wandb.finish()

    @staticmethod
    def _normalize_results(results):
        if results is None:
            return {}
        if isinstance(results, dict):
            return {k: list(v) if isinstance(v, list) else [v] for k, v in results.items()}
        if isinstance(results, list):
            normalized = {}
            for entry in results:
                if not isinstance(entry, dict):
                    continue
                for key, value in entry.items():
                    normalized.setdefault(key, []).append(value)
            return normalized
        return {}

    @staticmethod
    def _iter_result_dicts(results):
        if not isinstance(results, dict) or not results:
            return []
        lengths = [len(v) for v in results.values() if isinstance(v, list)]
        if not lengths:
            return []
        n = min(lengths)
        dicts = []
        for idx in range(n):
            item = {}
            for key, values in results.items():
                if isinstance(values, list) and idx < len(values):
                    item[key] = values[idx]
            dicts.append(item)
        return dicts

    @staticmethod
    def _last_result(results):
        if not isinstance(results, dict):
            return {}
        last = {}
        for key, values in results.items():
            if isinstance(values, list):
                if values:
                    last[key] = values[-1]
            else:
                last[key] = values
        return last


class timer(ContextDecorator):
    def __init__(self, name=None, log_func=logger.info):
        self.name = name
        self.log_func = log_func
        self.timer = Timer(name=name, logger=None)

    def __enter__(self):
        self.timer.start()
        self.log_func(f"Started {self.name} at {get_cur_time()}")
        return self

    def __exit__(self, *exc):
        elapsed_time = self.timer.stop()
        formatted_time = format_timespan(elapsed_time)
        self.log_func(f"Finished {self.name} at {get_cur_time()}, running time = {formatted_time}.")
        return False

    def __call__(self, func):
        self.name = self.name or func.__name__

        @wraps(func)
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator


def get_cur_time(timezone=None, t_format="%m-%d %H:%M:%S"):
    return datetime.fromtimestamp(int(datetime.now().timestamp()), timezone).strftime(t_format)
