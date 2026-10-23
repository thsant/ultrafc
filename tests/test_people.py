import os
import time

from livetx.people import PeopleMap, parse


def test_parse_tolerates_spaces_comments_and_case():
    names, invalid = parse("P1:Marcos\n  p2 :  Claudio Silva \n\n# P3:ninguém\nlixo\n")
    assert names == {"P1": "Marcos", "P2": "Claudio Silva"}
    assert invalid == ["lixo"]


def test_missing_file_keeps_labels(tmp_path):
    people = PeopleMap(tmp_path / "people.txt", interval=60)
    assert people.name("P1") == "P1"
    people.stop()


def test_reloads_when_file_changes(tmp_path):
    path = tmp_path / "people.txt"
    people = PeopleMap(path, interval=0.05)
    path.write_text("P1:Marcos\n", encoding="utf-8")
    deadline = time.time() + 2
    while people.name("P1") == "P1" and time.time() < deadline:
        time.sleep(0.02)
    assert people.name("P1") == "Marcos" and people.name("P2") == "P2"

    path.write_text("P1:Marcos\nP2:Claudio\n", encoding="utf-8")
    os.utime(path, ns=(time.time_ns(), time.time_ns() + 1))
    deadline = time.time() + 2
    while people.name("P2") == "P2" and time.time() < deadline:
        time.sleep(0.02)
    assert people.name("P2") == "Claudio"

    path.unlink()
    deadline = time.time() + 2
    while people.name("P1") != "P1" and time.time() < deadline:
        time.sleep(0.02)
    assert people.name("P1") == "P1"
    people.stop()
