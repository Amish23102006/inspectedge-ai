import shutil

import pytest

from inspectedge.data.mvtec import (
    find_leakage,
    index_category,
    integrity_report,
    load_split,
    write_manifest,
)


def test_index_counts_and_masks(mvtec_tree):
    s = index_category(mvtec_tree, "bottle", 0.2, seed=42)
    rep = integrity_report(s)
    assert rep["counts"] == {
        "test/anomalous": 2,
        "test/normal": 3,
        "train/normal": 8,
        "val/normal": 2,
    }
    assert rep["defect_types"] == ["broken_large"]
    assert len(rep["missing_masks"]) == 1 and rep["missing_masks"][0].endswith("001.png")


def test_split_is_deterministic_and_disjoint(mvtec_tree):
    a = index_category(mvtec_tree, "bottle", 0.2, seed=42)
    b = index_category(mvtec_tree, "bottle", 0.2, seed=42)
    assert a == b
    train = {x.path for x in a if x.split == "train"}
    val = {x.path for x in a if x.split == "val"}
    assert train and val and not train & val
    assert all(x.label == 0 for x in a if x.split in {"train", "val"})  # normal-only training


def test_no_leakage_in_clean_tree(mvtec_tree):
    assert find_leakage(index_category(mvtec_tree, "bottle", 0.2, 42)) == []


def test_leakage_detected_when_train_file_copied_to_test(mvtec_tree):
    cat = mvtec_tree / "bottle"
    shutil.copy(cat / "train/good/000.png", cat / "test/good/999.png")
    leaks = find_leakage(index_category(mvtec_tree, "bottle", 0.2, 42))
    assert len(leaks) == 1 and "test" in leaks[0]["splits"]


def test_missing_category_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        index_category(tmp_path, "bottle", 0.2, 42)


def test_load_split_shapes_and_manifest(mvtec_tree, tmp_path):
    s = index_category(mvtec_tree, "bottle", 0.2, 42)
    x, y, m, chosen = load_split(s, "test", 64)
    assert x.shape == (5, 64, 64, 3) and x.dtype.name == "uint8"
    assert y.sum() == 2 and m.shape == (5, 64, 64) and m.any()
    write_manifest(s, tmp_path / "m.csv")
    assert len((tmp_path / "m.csv").read_text().splitlines()) == len(s) + 1
