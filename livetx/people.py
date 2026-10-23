"""Nomes dos falantes informados pelo operador em people.txt (`P1:Marcos`)."""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path

log = logging.getLogger(__name__)

_LINE = re.compile(r"^\s*(P\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)


def parse(text: str) -> tuple[dict[str, str], list[str]]:
    """Devolve (rótulo → nome, linhas inválidas). Ignora linhas vazias e comentários (#)."""
    names: dict[str, str] = {}
    invalid: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if m := _LINE.match(line):
            names[m.group(1).upper()] = m.group(2)
        else:
            invalid.append(line.strip())
    return names, invalid


class PeopleMap:
    """Relê o arquivo a cada `interval` segundos, quando ele muda, numa thread de fundo."""

    def __init__(self, path: str | Path, interval: float = 10.0):
        self.path = Path(path)
        self.interval = interval
        self._names: dict[str, str] = {}
        self._stamp: tuple[int, int] | None = None
        self._warned: set[str] = set()
        self._stop = threading.Event()
        self.reload()
        self._thread = threading.Thread(target=self._run, daemon=True, name="people")
        self._thread.start()

    def name(self, label: str) -> str:
        return self._names.get(label.upper(), label)

    def reload(self) -> None:
        try:
            st = os.stat(self.path)
            stamp = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            stamp = None
        if stamp == self._stamp:
            return
        self._stamp = stamp

        names: dict[str, str] = {}
        if stamp is not None:
            names, invalid = parse(self.path.read_text(encoding="utf-8"))
            for line in invalid:
                if line not in self._warned:
                    self._warned.add(line)
                    log.warning("%s: linha ignorada (esperado 'P1:Nome'): %r", self.path, line)
        if names != self._names:
            log.info("nomes atualizados: %s",
                     ", ".join(f"{k}→{v}" for k, v in sorted(names.items())) or "nenhum")
        self._names = names  # troca atômica: quem lê vê o mapa antigo ou o novo, nunca parcial

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.reload()
            except Exception as exc:  # noqa: BLE001 - erro de leitura não pode derrubar a transcrição
                log.warning("falha ao ler %s: %s", self.path, exc)

    def stop(self) -> None:
        self._stop.set()
