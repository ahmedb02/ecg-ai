import pytest

pytest.importorskip("pandas")
pytest.importorskip("wfdb")

from ecg_ai.data.ptbxl import (
    SCP_CODE_TO_FINDING, _scp_codes_to_findings, find_ptbxl_root, infer_evolution_label,
)


def test_scp_codes_to_findings_splits_global_and_regional():
    global_findings, regional_findings = _scp_codes_to_findings(
        {"IMI": 100.0, "STACH": 80.0, "NOT_A_REAL_CODE": 50.0}
    )
    assert global_findings == ["sinus_tach"]
    assert regional_findings == [("old_mi", "inferior")]


def test_scp_codes_to_findings_respects_likelihood_threshold():
    global_findings, _ = _scp_codes_to_findings({"STACH": 20.0}, likelihood_threshold=50.0)
    assert global_findings == []


def test_find_ptbxl_root_raises_when_missing():
    with pytest.raises(FileNotFoundError):
        find_ptbxl_root(["/definitely/not/a/real/path"])


@pytest.mark.parametrize(
    "baseline,latest,expected",
    [
        (set(), {"stemi"}, "new_or_evolving_stemi"),
        ({"nstemi_ischemia"}, {"stemi"}, "worsening_ischemia"),
        ({"stemi"}, {"old_mi"}, "interval_infarct_evolution"),
        ({"lbbb"}, set(), "resolved_conduction_abnormality"),
        (set(), {"afib"}, "new_arrhythmia"),
        (set(), set(), "no_significant_change"),
        ({"nstemi_ischemia"}, set(), "resolving_ischemia"),
        (set(), {"nstemi_ischemia"}, "worsening_ischemia"),
    ],
)
def test_infer_evolution_label(baseline, latest, expected):
    assert infer_evolution_label(baseline, latest) == expected


def test_scp_code_mapping_only_uses_valid_finding_keys():
    from ecg_ai.clinical.labels import FINDING_INDEX, REGIONS

    for code, (finding_key, region) in SCP_CODE_TO_FINDING.items():
        assert finding_key in FINDING_INDEX, f"{code} maps to unknown finding key {finding_key}"
        if region is not None:
            assert region in REGIONS, f"{code} maps to unknown region {region}"
