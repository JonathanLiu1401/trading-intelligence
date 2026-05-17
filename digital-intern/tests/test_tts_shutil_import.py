"""Regression: notifier.tts must import `shutil`.

`_speak_kokoro` calls `shutil.which("ffplay")` / `shutil.which("aplay")`
to pick an audio player. The module previously never imported `shutil`,
so every Kokoro playback raised `NameError: name 'shutil' is not defined`.
The bug was masked because `_speak_kokoro` wraps its body in a broad
`except Exception` that only *prints* the error — so this test drives the
playback branch with everything else mocked and asserts the player branch
runs (subprocess invoked) with no NameError surfacing in stdout.

No network, no real audio: kokoro_onnx/numpy are faked, the WAV write is
mocked, and subprocess.run is mocked.
"""
from __future__ import annotations

import sys
from unittest import mock

import pytest

from notifier import tts


def test_module_imports_shutil():
    # The actual fix: the name must resolve at module scope.
    assert hasattr(tts, "shutil"), "notifier.tts is missing `import shutil`"


@pytest.mark.parametrize(
    "which_map, expected_player",
    [
        ({"ffplay": "/usr/bin/ffplay"}, "ffplay"),   # line ~112 branch
        ({"aplay": "/usr/bin/aplay"}, "aplay"),       # line ~117 branch
    ],
)
def test_speak_kokoro_player_branch_no_nameerror(
    which_map, expected_player, capsys
):
    fake_kokoro_mod = mock.MagicMock()
    fake_kokoro_mod.Kokoro.return_value.create.return_value = (
        mock.MagicMock(),  # samples
        24000,             # sample_rate
    )

    # KOKORO_MODEL/VOICES are Path instances whose .exists() is read-only,
    # so swap the module attributes for mocks that report the files present.
    fake_model = mock.MagicMock()
    fake_model.exists.return_value = True
    fake_voices = mock.MagicMock()
    fake_voices.exists.return_value = True

    def fake_which(name):
        return which_map.get(name)

    with mock.patch.dict(
        sys.modules,
        {"kokoro_onnx": fake_kokoro_mod, "numpy": mock.MagicMock()},
    ), mock.patch.object(tts, "KOKORO_MODEL", fake_model), \
            mock.patch.object(tts, "KOKORO_VOICES", fake_voices), \
            mock.patch.object(tts.tempfile, "NamedTemporaryFile") as ntf, \
            mock.patch.object(tts.wave, "open"), \
            mock.patch("shutil.which", side_effect=fake_which), \
            mock.patch.object(tts.subprocess, "run") as run:
        ntf.return_value.__enter__.return_value.name = "/tmp/_tts_test_fake.wav"

        # Must not raise; the broad except in _speak_kokoro would otherwise
        # convert a NameError into a printed line.
        tts._speak_kokoro("hello markets")

    out = capsys.readouterr().out

    # The shutil-using branch was reached and passed cleanly.
    run.assert_called_once()
    assert expected_player in run.call_args.args[0][0]

    # No NameError (or any Kokoro error) leaked through the broad except.
    assert "NameError" not in out
    assert "not defined" not in out
    assert "Kokoro error" not in out
    assert "Kokoro playback complete" in out
