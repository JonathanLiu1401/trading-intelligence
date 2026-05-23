"""batch_runner.PIPELINE wiring guard.

Two of the standalone analytics tools (``junk_source_detector``,
``source_lead_time``) previously sat outside the hourly batch pipeline —
they had to be invoked by hand or never ran at all, which made their
outputs perpetually stale. Pin them as PIPELINE members so a future edit
that drops them is caught at test time.

Also guards the structural shape of PIPELINE so a malformed entry (wrong
arity, non-Path output) is rejected before the cron tries to spawn it.
"""
from __future__ import annotations

from pathlib import Path

from analytics import batch_runner


def test_junk_source_detector_in_pipeline():
    """The fixed CLI tool must be scheduled — otherwise the junk-source
    report drifts to permanent stale."""
    modules = [m for m, _, _ in batch_runner.PIPELINE]
    assert "analytics.junk_source_detector" in modules


def test_source_lead_time_in_pipeline():
    modules = [m for m, _, _ in batch_runner.PIPELINE]
    assert "analytics.source_lead_time" in modules


def test_pipeline_entries_well_formed():
    """Every entry is (str module, Path output, int threshold). The cron
    loop in run() assumes this shape; a bad entry would raise at parse time
    of the first iteration."""
    for entry in batch_runner.PIPELINE:
        assert len(entry) == 3, f"PIPELINE entry has wrong arity: {entry!r}"
        module, output_path, threshold = entry
        assert isinstance(module, str) and module.startswith("analytics."), (
            f"module name must be qualified analytics.X: {module!r}"
        )
        assert isinstance(output_path, Path), (
            f"output path must be a Path, got {type(output_path).__name__}: "
            f"{output_path!r}"
        )
        assert isinstance(threshold, int) and threshold > 0, (
            f"threshold must be a positive int, got {threshold!r}"
        )


def test_pipeline_module_names_unique():
    """Duplicate entries would run the same module twice per cycle and
    contend on the same output file. Cheap to pin."""
    modules = [m for m, _, _ in batch_runner.PIPELINE]
    assert len(modules) == len(set(modules)), (
        f"PIPELINE has duplicate modules: {[m for m in modules if modules.count(m) > 1]}"
    )


def test_pipeline_output_paths_unique():
    """Two modules writing to the same file would corrupt _is_fresh's mtime
    check (whichever ran last would let the other be wrongly skipped)."""
    outputs = [p for _, p, _ in batch_runner.PIPELINE]
    assert len(outputs) == len(set(outputs)), (
        f"PIPELINE has duplicate output paths: "
        f"{[p for p in outputs if outputs.count(p) > 1]}"
    )


def test_junk_source_detector_output_matches_script():
    """The output path in PIPELINE must match the OUT_PATH the script
    actually writes to — otherwise _is_fresh checks a never-written file
    and every cycle re-runs the module needlessly."""
    from analytics import junk_source_detector
    pipeline_paths = {m: p for m, p, _ in batch_runner.PIPELINE}
    assert (
        pipeline_paths["analytics.junk_source_detector"]
        == junk_source_detector.OUT_PATH
    )


def test_source_lead_time_output_matches_script():
    from analytics import source_lead_time
    pipeline_paths = {m: p for m, p, _ in batch_runner.PIPELINE}
    assert (
        pipeline_paths["analytics.source_lead_time"]
        == source_lead_time.OUT
    )
