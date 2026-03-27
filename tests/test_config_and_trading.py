import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from csv_gen import generate_initial_portfolio
from simulation_core import execute_preference_trades, run_simulation
from tdc_shared import BOND_PORTFOLIO_COLS, PORTFOLIO_DTYPES
from tdc_validation import validate_config, validate_events, validate_sector_preferences


def empty_portfolio_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES)


def base_issuance_profile():
    return {
        'bills': {
            'category_cutoff_years': 1.0,
            'target_percentage_of_remainder': 1.0,
            'maturities': [1.0],
            'maturity_distribution': [1.0],
        },
        'notes': {
            'category_cutoff_years': 10.0,
            'target_percentage_of_remainder': 0.0,
            'maturities': [2.0],
            'maturity_distribution': [1.0],
        },
        'bonds': {
            'category_cutoff_years': 999.0,
            'target_percentage_of_remainder': 0.0,
            'maturities': [20.0],
            'maturity_distribution': [1.0],
        },
        'remainder_maturity_years': 1.0,
    }


def full_holder_prefs(private_bill_weight=1.0):
    return {
        'Banks': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
        'Private': {'bills_pct': private_bill_weight, 'notes_pct': 1.0, 'bonds_pct': 1.0},
        'CB': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
        'Foreign': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
    }


def minimal_params() -> dict:
    return {
        'initial_values': {'reserves': 1000.0, 'tdc_level': 0.0, 'tga': 50.0},
        'tga_params': {'target_balance': 100.0, 'floor': 0.0},
        'fiscal_params': {
            'initial_weekly_spending': 0.0,
            'initial_weekly_taxes': 0.0,
            'spending_growth_qtr': 0.0,
            'tax_growth_qtr': 0.0,
        },
        'other_flows': {'reserve_transfer': 0.0, 'cb_net_expense': 0.0, 'money_minting_transfers': 0.0},
        'treasury_issuance_profile': base_issuance_profile(),
        'yield_curve': {
            'use_static': True,
            'years': [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
            'rates': [0.04, 0.045, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        },
        'sector_preferences': full_holder_prefs(),
        'simulation_period': {'enable_preference_trading': False},
        'initial_bonds_df': empty_portfolio_df(),
        'events': [],
    }


def first_active_period(results: pd.DataFrame) -> pd.Series:
    assert len(results.index) >= 2
    return results.iloc[1]


def make_bond_row(**overrides):
    row = {col: pd.NA for col in BOND_PORTFOLIO_COLS}
    defaults = {
        'BondID': 1,
        'SecurityType': 'Fixed',
        'IssueDate': pd.Timestamp('2025-01-01'),
        'MaturityDate': pd.Timestamp('2027-01-01'),
        'OriginalMaturityYears': 2.0,
        'FaceValue': 100.0,
        'CouponRate': 0.04,
        'HolderType': 'Banks',
        'Status': 'Active',
        'MaturityCategory': 'notes',
        'OriginalPrincipal': 100.0,
        'AdjustedPrincipal': 100.0,
        'ReferenceCPI_Issue': 0.0,
        'IndexRatio': 1.0,
        'FixedSpread': 0.0,
        'AccruedInterest_FRN': 0.0,
        'BenchmarkRate_FRN': 0.0,
        'LastAccrualDate': pd.Timestamp('2025-03-01'),
        'IssuePriceRatio': 1.0,
        'IssueProceeds': 100.0,
        'IssueYieldAtIssue': 0.04,
        'TimeToMaturity': pd.NA,
        'DiscountYield': pd.NA,
        'CleanPrice': pd.NA,
        'AccruedInterest': pd.NA,
        'DirtyValue': pd.NA,
        'DirtyPriceRatio': pd.NA,
    }
    defaults.update(overrides)
    row.update(defaults)
    return row


def test_shipped_config_validates():
    cfg_path = Path(__file__).resolve().parent.parent / 'tdc_config.yaml'
    with cfg_path.open('r') as f:
        cfg = yaml.safe_load(f)
    errors = validate_config(cfg)
    assert errors == []


def test_bill_issuance_uses_discounted_proceeds():
    params = minimal_params()
    results, _ = run_simulation(params, '2025-01-01', '2025-01-19', freq='W', scenario_name='bill_discount_test')
    period1 = first_active_period(results)

    assert period1['AuctionProceeds'] == pytest.approx(50.0, rel=1e-6)
    assert period1['NewDebtIssued'] > period1['AuctionProceeds']
    assert period1['IssueDiscountCost_Period'] > 0.0
    assert period1['IssueDiscountCost_Period'] == pytest.approx(
        period1['NewDebtIssued'] - period1['AuctionProceeds'], rel=1e-6
    )
    assert results.attrs['run_metadata']['validation_status'] == 'passed'



def test_pre_t0_event_is_applied_before_first_week():
    baseline = minimal_params()
    baseline_results, _ = run_simulation(baseline, '2025-01-01', '2025-01-19', freq='W', scenario_name='baseline')

    shocked = minimal_params()
    shocked['events'] = [
        {
            'date': '2025-01-03',
            'actions': [
                {'parameter_path': 'tga_params.target_balance', 'new_value': 200.0},
            ],
        }
    ]
    shocked_results, _ = run_simulation(shocked, '2025-01-01', '2025-01-19', freq='W', scenario_name='pre_t0_event')

    assert first_active_period(baseline_results)['AuctionProceeds'] == pytest.approx(50.0, rel=1e-6)
    assert first_active_period(shocked_results)['AuctionProceeds'] == pytest.approx(150.0, rel=1e-6)



def test_validate_events_rejects_unsupported_paths():
    errors = validate_events([
        {
            'date': '2025-01-10',
            'actions': [{'parameter_path': 'foo.bar', 'new_value': 1}],
        }
    ])
    assert errors
    assert 'Unsupported parameter block' in errors[0] or 'foo.bar' in errors[0]



def test_validate_sector_preferences_rejects_non_unit_columns():
    bad_prefs = {
        'Banks': {'bills_pct': 0.4, 'notes_pct': 0.0, 'bonds_pct': 0.0},
        'Private': {'bills_pct': 0.4, 'notes_pct': 1.0, 'bonds_pct': 1.0},
        'CB': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
        'Foreign': {'bills_pct': 0.1, 'notes_pct': 0.0, 'bonds_pct': 0.0},
    }
    errors = validate_sector_preferences(bad_prefs, issuance_profile=base_issuance_profile())
    assert errors
    assert any("bills_pct" in err for err in errors)



def test_explicit_auction_and_secondary_preferences_can_differ():
    params = minimal_params()
    params.pop('sector_preferences', None)
    params['auction_absorption_preferences'] = {
        'Banks': {'bills_pct': 1.0, 'notes_pct': 1.0, 'bonds_pct': 1.0},
        'Private': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
        'CB': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
        'Foreign': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
    }
    params['secondary_target_preferences'] = {
        'Banks': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
        'Private': {'bills_pct': 1.0, 'notes_pct': 1.0, 'bonds_pct': 1.0},
        'CB': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
        'Foreign': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
    }
    params['simulation_period']['enable_preference_trading'] = False

    results, _ = run_simulation(params, '2025-01-01', '2025-01-19', freq='W', scenario_name='split_prefs')
    period1 = first_active_period(results)

    assert period1['DebtHeld_Banks'] > 0.0
    assert period1['DebtHeld_DomesticNonBanks'] == pytest.approx(0.0, abs=1e-8)
    assert results.attrs['run_metadata']['uses_legacy_sector_preferences_for_auction'] is False
    assert results.attrs['run_metadata']['uses_legacy_sector_preferences_for_secondary'] is False



def test_frn_partial_trade_preserves_accrued_interest():
    portfolio = pd.DataFrame([
        make_bond_row(
            BondID=101,
            SecurityType='FRN',
            HolderType='Banks',
            CouponRate=0.0,
            FixedSpread=0.002,
            BenchmarkRate_FRN=0.04,
            AccruedInterest_FRN=2.0,
            LastAccrualDate=pd.Timestamp('2025-03-01'),
            IssueYieldAtIssue=0.0,
        ),
        make_bond_row(BondID=102, SecurityType='Fixed', HolderType='Banks', CouponRate=0.04),
        make_bond_row(BondID=103, SecurityType='Fixed', HolderType='Private', CouponRate=0.04),
    ], columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES, errors='ignore')

    prefs = {
        'Banks': {'bills_pct': 0.0, 'notes_pct': 0.75, 'bonds_pct': 0.0, 'tips_pct': 0.0, 'frn_pct': 0.25},
        'Private': {'bills_pct': 0.0, 'notes_pct': 0.5, 'bonds_pct': 0.0, 'tips_pct': 0.0, 'frn_pct': 0.5},
        'CB': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0, 'tips_pct': 0.0, 'frn_pct': 0.0},
        'Foreign': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0, 'tips_pct': 0.0, 'frn_pct': 0.0},
        'FedInternal': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0, 'tips_pct': 0.0, 'frn_pct': 0.0},
        'TrustFunds': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0, 'tips_pct': 0.0, 'frn_pct': 0.0},
    }

    traded, _ = execute_preference_trades(
        portfolio,
        pd.Timestamp('2025-06-15'),
        [0.25, 0.5, 1.0, 2.0, 5.0],
        [0.04, 0.041, 0.042, 0.043, 0.045],
        prefs,
        {
            'bills': {'category_cutoff_years': 1.0},
            'notes': {'category_cutoff_years': 10.0},
            'bonds': {'category_cutoff_years': 999.0},
        },
        'frn_trade_test',
    )

    frn_rows = traded[(traded['SecurityType'] == 'FRN') & (traded['BondID'] == 101)].sort_values('HolderType').reset_index(drop=True)
    assert len(frn_rows) == 2
    assert frn_rows['FaceValue'].sum() == pytest.approx(100.0, rel=1e-6)
    assert frn_rows['AccruedInterest_FRN'].sum() == pytest.approx(2.0, rel=1e-6)
    assert set(frn_rows['HolderType'].astype(str)) == {'Banks', 'Private'}
    for _, row in frn_rows.iterrows():
        assert row['AccruedInterest_FRN'] == pytest.approx(2.0 * row['FaceValue'] / 100.0, rel=1e-5)
        assert row['AccruedInterest_FRN'] > 0.0



def test_generated_portfolio_populates_issue_cash_fields():
    df = generate_initial_portfolio(
        {
            'target_public_marketable_wam': 3.0,
            'wam_targeting_iterations': 1,
            'random_seed': 123,
            'target_face_values_billions': {
                'Banks': 0.05,
                'Private_Marketable': 0.10,
                'Private_NonMarketable': 0.0,
                'CB': 0.0,
                'Foreign': 0.0,
                'FedInternal': 0.0,
                'TrustFunds_NonMarketable': 0.0,
            },
        },
        '2025-01-01',
    )

    assert not df.empty
    assert {'IssuePriceRatio', 'IssueProceeds', 'IssueYieldAtIssue'}.issubset(df.columns)
    bill_mask = (
        (df['SecurityType'] == 'Fixed')
        & (df['CouponRate'] <= 1.0e-12)
        & (df['OriginalMaturityYears'] <= 1.0 + 1.0e-12)
    )
    assert bill_mask.any()
    assert (df.loc[bill_mask, 'IssueProceeds'] < df.loc[bill_mask, 'FaceValue']).all()
