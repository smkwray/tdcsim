from pathlib import Path

import pandas as pd
import pytest

from sim_engine import run_simulation
from test_cbo_engine_integration import (
    _build_temp_forecast_inputs,
    _minimal_engine_params,
    _opening_tips_portfolio,
    _write_macro_path,
)


def test_tips_deflation_floor_is_named_stock_to_cash_journal_leg(
    tmp_path: Path,
) -> None:
    """Adjusted stock 80 plus the named floor top-up 20 must fund cash 100."""

    paths = _build_temp_forecast_inputs(
        tmp_path,
        cbo_public_debt_target_bil=205.0,
        public_nonmarketable_bil=100.0,
        definition_residual_bil=25.0,
        pre_issuance_controlled_debt_bil=100.0,
        signed_primary_flow_bil=0.0,
    )
    paths["macro_forecast_path_file"] = _write_macro_path(
        tmp_path,
        cpi_u_index=80.0,
    )
    opening = _opening_tips_portfolio(face_value=100.0, reference_cpi=100.0)
    opening.loc[:, "MaturityDate"] = pd.Timestamp("2026-09-25")
    opening.loc[:, "AdjustedPrincipal"] = 80.0
    opening.loc[:, "IndexRatio"] = 0.8
    params = _minimal_engine_params(paths, opening_controlled_debt_bil=0.0)
    params["initial_bonds_df"] = opening
    params["tips_params"] = {
        "cpi_start_level": 80.0,
        "cpi_annual_inflation": 0.0,
        "ref_cpi_lag_months": 0,
        "default_real_coupon_rate": 0.0,
    }

    results, _ = run_simulation(
        params,
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    principal = pd.DataFrame(
        results.attrs["handoff_tables"]["tdcsim_period_principal_flows"]
    )
    maturity = principal.loc[
        principal["instrument_type"].eq("TIPS")
        & principal["redemption_type"].eq("scheduled_maturity")
    ].iloc[0]
    payments = pd.DataFrame(
        results.attrs["handoff_tables"]["tdcsim_period_payment_flows"]
    )
    floor = payments.loc[
        payments["payment_type"].eq("tips_deflation_floor_topup")
    ]

    assert "adjusted_principal_stock_removed_bil" in maturity.index, (
        "TIPS maturity journal omits adjusted-principal stock removal"
    )
    assert maturity["adjusted_principal_stock_removed_bil"] == pytest.approx(80.0)
    assert maturity["cash_paid_bil"] == pytest.approx(100.0)
    assert len(floor) == 1
    assert floor.iloc[0]["accounting_basis"] == "stock_to_cash_bridge"
    assert floor.iloc[0]["amount_bil"] == pytest.approx(20.0)
    assert floor.iloc[0]["is_additive_to_cash_total"] in (False, 0)
    assert (
        maturity["adjusted_principal_stock_removed_bil"]
        + floor.iloc[0]["amount_bil"]
    ) == pytest.approx(maturity["cash_paid_bil"])
