"""Gravação da transcrição em arquivos de log diários."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# Falante: até 40 caracteres sem ":" (nomes do people.txt, P1..PN ou "?").
_LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (?:([^:\[\]]{1,40}): )?(.*)$")


def format_line(ts: float, text: str, speaker: str | None = None) -> str:
    if speaker:
        text = f"{speaker}: {text}"
    return f"[{time.strftime(TIME_FORMAT, time.localtime(ts))}] {text}"


@dataclass
class LogLine:
    ts: float
    speaker: str | None
    text: str

    def __str__(self) -> str:
        return format_line(self.ts, self.text, self.speaker)


def parse_line(line: str) -> LogLine | None:
    """Inverso de `format_line`; None se a linha não está no formato do log."""
    m = _LINE.match(line.rstrip("\n"))
    if not m:
        return None
    ts = time.mktime(time.strptime(m.group(1), TIME_FORMAT))
    return LogLine(ts, m.group(2), m.group(3))


class TranscriptLog:
    """Escreve `[AAAA-MM-DD HH:MM:SS] [falante: ]texto` em `<dir>/<AAAA-MM-DD>.log`, um arquivo por dia."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._day: str | None = None
        self._file = None

    def _file_for(self, ts: float):
        day = time.strftime("%Y-%m-%d", time.localtime(ts))
        if day != self._day:
            self.close()
            self._file = open(self.directory / f"{day}.log", "a", encoding="utf-8")
            self._day = day
        return self._file

    def write(self, ts: float, text: str, speaker: str | None = None) -> None:
        line = format_line(ts, text, speaker)
        f = self._file_for(ts)
        f.write(line + "\n")
        f.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
