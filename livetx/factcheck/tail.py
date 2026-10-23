"""Leitura incremental do log da transcrição, com retomada pelo estado salvo."""

from __future__ import annotations

import json
from pathlib import Path

from ..logger import LogLine, parse_line


class LogTail:
    """Segue um arquivo de log, ou o log mais recente (`AAAA-MM-DD.log`) de um diretório.

    `read()` devolve as linhas completas novas; `commit()` grava até onde já foi processado,
    para que um reinício não repita nem perca linhas.
    """

    def __init__(self, source: Path, state_path: Path, from_start: bool = False):
        self.source = Path(source)
        self.state_path = Path(state_path)
        self.file: Path | None = None
        self.offset = 0
        self._read_to = 0
        state = None if from_start else self._load_state()
        if state and self._resolve(state["file"]).exists() and self._resolve(state["file"]).name == state["file"]:
            self.file = self._resolve(state["file"])
            self.offset = self._read_to = state["offset"]
        elif not from_start:
            # Primeira execução sem --from-start: começa do fim (só o que for dito daqui em diante).
            self.file = self._latest()
            if self.file:
                self.offset = self._read_to = self.file.stat().st_size

    def _resolve(self, name: str) -> Path:
        return self.source / name if self.source.is_dir() else self.source

    def _latest(self) -> Path | None:
        if not self.source.is_dir():
            return self.source if self.source.exists() else None
        logs = sorted(self.source.glob("????-??-??.log"))
        return logs[-1] if logs else None

    def _load_state(self) -> dict | None:
        try:
            return json.loads(self.state_path.read_text())
        except (FileNotFoundError, ValueError):
            return None

    def read(self) -> list[LogLine]:
        if self.file is None:
            self.file = self._latest()
            if self.file is None:
                return []
        lines = self._read_complete()
        latest = self._latest()
        if not lines and latest and latest != self.file:
            # Virada do dia: o arquivo antigo acabou e há um mais novo.
            self.commit()
            self.file, self.offset, self._read_to = latest, 0, 0
            lines = self._read_complete()
        return lines

    def _read_complete(self) -> list[LogLine]:
        try:
            with open(self.file, "rb") as f:
                f.seek(self._read_to)
                data = f.read()
        except FileNotFoundError:
            return []
        end = data.rfind(b"\n") + 1  # ignora a última linha se ainda estiver sendo escrita
        self._read_to += end
        out = []
        for raw in data[:end].decode("utf-8", errors="replace").splitlines():
            if (line := parse_line(raw)) is not None:
                out.append(line)
        return out

    def commit(self) -> None:
        if self.file is None:
            return
        self.offset = self._read_to
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({"file": self.file.name, "offset": self.offset}))
