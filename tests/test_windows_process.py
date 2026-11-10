"""Hidden venv launches without rendering or probing hardware."""
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import windows_process
from windows_process import hidden_venv_command


def test_hidden_venv_uses_windowless_redirector_preserving_arguments(tmp_path):
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    python = scripts / "python.exe"
    pythonw = scripts / "pythonw.exe"
    pythonw.touch()
    command = [str(python), "-X", "utf8", "worker.py", "--serve"]
    assert hidden_venv_command(command, 0x08000004) == [str(pythonw), *command[1:]]
    assert command[0] == str(python)
    assert hidden_venv_command(command, 0x200) is command
    assert hidden_venv_command(tuple(command), 0x08000000) == [str(pythonw), *command[1:]]


def test_hidden_venv_missing_windowless_launcher_reports_repair(tmp_path):
    with pytest.raises(FileNotFoundError, match="repair"):
        hidden_venv_command([str(tmp_path / "Scripts" / "python.exe"), "worker.py"], 0x08000000)


@pytest.mark.parametrize("command", [["python", "worker.py"], ["tool.exe"], "python worker.py", []])
def test_other_commands_are_unchanged(command):
    assert hidden_venv_command(command, 0x08000000) is command


def test_non_venv_python_and_existing_pythonw_are_unchanged(tmp_path):
    for path in (tmp_path / "python.exe", tmp_path / "Scripts" / "pythonw.exe"):
        command = [str(path), "worker.py"]
        assert hidden_venv_command(command, 0x08000000) is command


def test_owned_spawn_hides_venv_while_preserving_logs_and_attach_before_resume(tmp_path, monkeypatch):
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    pythonw = scripts / "pythonw.exe"
    pythonw.touch()
    calls = []
    events = []
    process = SimpleNamespace(_handle=1234)

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    resume = Mock(side_effect=lambda handle: events.append("resume") or 0)
    monkeypatch.setattr(windows_process.subprocess, "Popen", popen)
    monkeypatch.setattr(windows_process.ctypes, "WinDLL",
                        lambda *args: SimpleNamespace(NtResumeProcess=resume), raising=False)
    job = windows_process.OwnedJob.__new__(windows_process.OwnedJob)
    job.attach = lambda child: events.append("attach")
    result = job.spawn([str(scripts / "python.exe"), "-X", "utf8", "worker.py"],
                       creationflags=0x08000200, stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    command, kwargs = calls[0]
    assert command == [str(pythonw), "-X", "utf8", "worker.py"]
    assert kwargs["creationflags"] == 0x08000204
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stdout"] == subprocess.PIPE and kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["text"] and kwargs["encoding"] == "utf-8"
    assert events == ["attach", "resume"]
    assert result is process and process._studio_job is job
