import json
from laptop50m.infrastructure.plot_curve import loss_curve_svg


def test_loss_curve_svg_has_one_point_per_eval(tmp_path):
    log = tmp_path / "log.jsonl"
    recs = [{"step": 10, "loss": 6.0}, {"step": 250, "loss": 5.0, "val_loss": 5.5, "wikitext103_val_loss": 6.5},
            {"step": 500, "loss": 4.0, "val_loss": 4.5, "wikitext103_val_loss": 5.2}]
    log.write_text("\n".join(json.dumps(r) for r in recs))
    svg = loss_curve_svg(str(log))
    assert svg.startswith("<svg") and svg.count("<polyline") == 2
    assert "FineWeb-Edu val" in svg and "WikiText-103 val" in svg
