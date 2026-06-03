"""Tests for the browser chat page surface.

The chat page should stay focused on conversation. Portfolio/equity charts
belong on the Paper Trader page, not above the chat stream.
"""

from dashboard.web_server import create_app


def test_chat_page_does_not_embed_paper_trader_return_graph():
    client = create_app().test_client()

    resp = client.get("/chat")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    assert "Paper Trader Return Graph" not in html
    assert "return-chart" not in html
    assert "/trader/api/equity-tail" not in html
    assert "Chart.js" not in html
