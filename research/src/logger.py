import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict


class Logger:
    """
    Универсальный логгер:
    - выводит события в CLI
    - пишет структурированные события в JSON Lines файл

    Формат JSON: 1 событие = 1 строка (jsonl)
    """

    def __init__(
        self,
        json_path: str,
        *,
        name: str = "app",
        flush: bool = True,
        cli: bool = True
    ):
        """
        json_path: путь до .json или .jsonl файла
        name: имя логгера (пишется в каждую запись)
        flush: делать flush после каждой записи
        cli: печатать ли в stdout
        """
        self.name = name
        self.flush = flush
        self.cli = cli

        self.json_path = Path(json_path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

        # Открываем файл в append-режиме
        self._fh = open(self.json_path, "a", encoding="utf-8")

    # -------------------------
    # Public API
    # -------------------------

    def log(self, event: str, **payload):
        """
        Универсальный метод логирования
        """
        record = self._make_record(event, payload)

        if self.cli:
            self._print_cli(record)

        self._write_json(record)

    def info(self, event: str, **payload):
        self.log(event, level="INFO", **payload)

    def warning(self, event: str, **payload):
        self.log(event, level="WARNING", **payload)

    def error(self, event: str, **payload):
        self.log(event, level="ERROR", **payload)

    # -------------------------
    # Internals
    # -------------------------

    def _make_record(self, event: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ts": time.time(),
            "datetime": datetime.utcnow().isoformat(),
            "logger": self.name,
            "event": event,
            **payload
        }

    def _print_cli(self, record: Dict[str, Any]):
        ts = record["datetime"]
        lvl = record.get("level", "INFO")
        event = record["event"]

        rest = {
            k: v
            for k, v in record.items()
            if k not in {"ts", "datetime", "logger", "event", "level"}
        }

        msg = f"[{ts}] [{self.name}] [{lvl}] {event}"
        if rest:
            msg += " | " + ", ".join(f"{k}={v}" for k, v in rest.items())

        print(msg, file=sys.stdout)

    def _write_json(self, record: Dict[str, Any]):
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.flush:
            self._fh.flush()

    def close(self):
        self._fh.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
