import threading

from aiwf.web.tabs import settings as settings_tab


def test_current_process_restart_command_uses_active_python_and_argv(monkeypatch):
    monkeypatch.setattr(settings_tab.sys, "executable", r"C:\AIWF\venv\Scripts\python.exe")
    monkeypatch.setattr(settings_tab.sys, "argv", [r"F:\AIWF_Studio\webui.py", "--port", "7861"])

    executable, argv = settings_tab._current_process_restart_command()

    assert executable == r"C:\AIWF\venv\Scripts\python.exe"
    assert argv == [r"C:\AIWF\venv\Scripts\python.exe", r"F:\AIWF_Studio\webui.py", "--port", "7861"]


def test_schedule_backend_restart_executes_once(monkeypatch):
    monkeypatch.setattr(settings_tab.sys, "executable", r"C:\AIWF\venv\Scripts\python.exe")
    monkeypatch.setattr(settings_tab.sys, "argv", [r"F:\AIWF_Studio\webui.py"])
    settings_tab._BACKEND_RESTART_REQUESTED = False
    done = threading.Event()
    calls = []

    def fake_exec(executable, argv):
        calls.append((executable, argv))
        done.set()

    try:
        first = settings_tab._schedule_backend_restart(
            delay_seconds=0,
            exec_fn=fake_exec,
            sleep_fn=lambda _delay: None,
        )
        second = settings_tab._schedule_backend_restart(
            delay_seconds=0,
            exec_fn=fake_exec,
            sleep_fn=lambda _delay: None,
        )

        assert first is True
        assert second is False
        assert done.wait(1)
        assert calls == [
            (
                r"C:\AIWF\venv\Scripts\python.exe",
                [r"C:\AIWF\venv\Scripts\python.exe", r"F:\AIWF_Studio\webui.py"],
            )
        ]
    finally:
        settings_tab._BACKEND_RESTART_REQUESTED = False
