"""Gravação da transcrição em arquivos de log diários."""

from __future__ import annotations

import time
from pathlib import Path


def format_line(ts: float, text: str, speaker: str | None = None) -> str:
    if speaker:
        text = f"{speaker}: {text}"
    return f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}] {text}"


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
