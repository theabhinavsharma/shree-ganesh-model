from src.ingest.events.nse import _classify_event_category
from src.ingest.events.nse import _looks_like_promoter_buying


def test_event_classifier_maps_key_announcement_types() -> None:
    assert _classify_event_category("financial results for quarter ended december 2024") == "results"
    assert _classify_event_category("receipt of order worth rs 500 crore") == "order_win"
    assert _classify_event_category("usfda approval received for product") == "approval"
    assert _classify_event_category("release of pledge by promoter group") == "pledge_change"


def test_promoter_buying_detector_requires_promoter_context() -> None:
    assert _looks_like_promoter_buying("promoter acquisition through open market purchase")
    assert not _looks_like_promoter_buying("company acquisition approved by board")
