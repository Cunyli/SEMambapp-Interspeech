import copy

import pytest
import torch

from model.avqi_components import AVQI_COMPONENT_NAMES
from model.avqi_route_c import six_active_bidirectional_gap_losses
from model.avqi_training_reference import training_row_with_same_formula_reference


def source_row(role="train"):
    return dict(training_role=role, target_components={n: float(i+1) for i, n in enumerate(AVQI_COMPONENT_NAMES)})


def clean_values():
    return {n: None if n == "shimmer_db" else float(i+2) for i, n in enumerate(AVQI_COMPONENT_NAMES)}


def test_same_formula_identity_removes_spurious_clean_loss_without_changing_exact_targets():
    row = source_row()
    original = copy.deepcopy(row)
    fixed = training_row_with_same_formula_reference(row, clean_values())
    actual_clean = torch.tensor([[fixed["target_components"][n] for n in AVQI_COMPONENT_NAMES]])
    exact = torch.tensor([[row["target_components"][n] for n in AVQI_COMPONENT_NAMES]])
    old_loss = six_active_bidirectional_gap_losses(actual_clean, exact, torch.zeros(6), torch.ones(6))
    fixed_loss = six_active_bidirectional_gap_losses(actual_clean, actual_clean, torch.zeros(6), torch.ones(6))
    assert old_loss.sum() > 0
    assert torch.count_nonzero(fixed_loss) == 0
    assert row == original and fixed["exact_target_components"] == original["target_components"]
    assert fixed["target_components"]["shimmer_db"] == row["target_components"]["shimmer_db"]


@pytest.mark.parametrize("role", ["validation", "evaluation"])
def test_heldout_rows_cannot_supply_training_references(role):
    with pytest.raises(ValueError, match="train rows only"):
        training_row_with_same_formula_reference(source_row(role), clean_values())


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_invalid_reference_stops_before_training(invalid):
    values = clean_values()
    values["tilt"] = invalid
    with pytest.raises(ValueError, match="nonfinite"):
        training_row_with_same_formula_reference(source_row(), values)


def test_legacy_db_reference_cannot_replace_exact_candidate_e_target():
    values = clean_values()
    values["shimmer_db"] = 999.
    with pytest.raises(ValueError, match="Candidate-E"):
        training_row_with_same_formula_reference(source_row(), values)
