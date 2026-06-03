"""Regression tests for the live portfolio equity chart UI.

The dashboard must keep the operator-facing dollar net-worth chart visible
while still offering deposit-adjusted return mode. A prior deposit-accounting
fix made the math correct but effectively hid the absolute total-value curve.
"""

from pathlib import Path


DASHBOARD = (
    Path(__file__).resolve().parents[1]
    / "paper_trader"
    / "dashboard.py"
)


def _source() -> str:
    return DASHBOARD.read_text(encoding="utf-8")


def test_live_portfolio_defaults_to_net_worth_mode():
    src = _source()

    assert 'let eqChartMode = "value";' in src
    assert 'id="eq-mode-value"' in src
    assert '$ net worth' in src
    assert "Net worth ($ total value)" in src
    assert 'label: isValueMode ? "Net worth $" : "Portfolio %"' in src


def test_net_worth_mode_shows_capital_basis_not_fake_gain():
    src = _source()

    assert "const totalValues = filtered.map" in src
    assert "const capitalBasis = filtered.map" in src
    assert 'label: "Capital basis $"' in src
    assert "deposit_adjusted_return_pct" in src
    assert "basisLegend.style.display = isValueMode" in src
