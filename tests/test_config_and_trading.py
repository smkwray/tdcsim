import copy
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
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


def cbo_phase0_blocks() -> dict:
    return {
        'funding_rule': {
            'mode': 'cbo_public_debt_target',
            'target_scope': 'federal_debt_held_by_public',
            'controlled_scope': 'treasury_marketable_debt_held_by_public',
            'stock_basis': 'principal_adjusted_tips',
            'target_enforcement': 'every_period',
            'negative_required_issuance_action': 'error',
            'target_tolerance_bil': 0.000001,
        },
        'baseline_input_paths': {
            'source_contract_file': 'data/forecast_inputs/source_contract.json',
            'source_fixture_file': 'data/forecast_inputs/source_fixtures.csv',
            'cbo_fiscal_baseline_file': 'data/forecast_inputs/tdcsim_cbo_fiscal_baseline.csv',
            'current_fy_splice_file': 'data/forecast_inputs/tdcsim_current_fy_splice.csv',
            'debt_stock_path_file': 'data/forecast_inputs/tdcsim_debt_stock_path.csv',
            'primary_deficit_path_file': 'data/forecast_inputs/tdcsim_primary_deficit_path.csv',
            'operating_cash_path_file': 'data/forecast_inputs/tdcsim_operating_cash_path.csv',
            'cash_reconciliation_residual_file': 'data/forecast_inputs/tdcsim_cash_reconciliation_residual.csv',
            'macro_forecast_path_file': 'data/forecast_inputs/tdcsim_macro_forecast_path.csv',
            'yield_curve_surface_file': 'data/forecast_inputs/tdcsim_yield_curve_surface.csv',
            'fiscal_incidence_policy_file': 'data/forecast_inputs/tdcsim_fiscal_incidence_policy.csv',
            'net_interest_bridge_file': 'data/forecast_inputs/tdcsim_net_interest_bridge.csv',
            'holder_absorption_path_file': 'data/forecast_inputs/tdcsim_holder_absorption_path.csv',
        },
        'data_vintage': {
            'forecast_name': 'cbo_2026_02_baseline',
            'forecast_publication_date': '2026-02-11',
            'actuals_available_as_of': '2026-04-30',
            'opening_state_date': '2026-04-30',
            'fiscal_actuals_through': '2026-03-31',
            'allow_lookahead': False,
            'lookahead_policy': 'production_no_future_actuals',
        },
        'operating_cash_policy': {
            'mode': 'explicit_path',
            'fallback_mode': 'scalar_indexed_anchor',
            'operating_cash_definition': 'tga_only',
            'inflation_index': 'cbo_cpi_u',
            'inflation_scalar': 1.0,
            'required_sensitivity_scalars': [0.0],
            'floor_bil': None,
            'enforcement': 'target_with_cash_reconciliation_residual',
        },
        'public_debt_bridge': {
            'mode': 'latest_actual_constant_nominal_by_component',
            'claim_boundary': 'full_public_debt_after_bridge',
            'missing_bridge_action': 'error',
            'treasury_only_allowed': False,
        },
        'fiscal_baseline': {
            'primary_deficit_mode': 'hard',
            'net_interest_mode': 'nonbinding_validation_check',
            'total_deficit_mode': 'identity_check',
            'current_fy_splice': 'required_when_start_after_fy_begin',
            'cb_remittance_cash_treatment': 'memo_only_zero_tga_effect',
        },
        'fiscal_incidence_policy': {
            'mode': 'explicit_scenario_assumption',
            'incidence_basis': 'signed_net_primary_proxy',
            'du_share': 0.99,
            'ru_share': 0.01,
            'foreign_share': 0.0,
            'other_share': 0.0,
            'required_sensitivities': [
                {'du_share': 1.0, 'ru_share': 0.0},
                {'du_share': 0.95, 'ru_share': 0.05},
            ],
        },
        'budget_interest': {
            'cbo_comparison_role': 'nonbinding_validation_check',
            'scope_status': 'incomplete',
            'residual_policy': 'warning_only_until_scope_complete',
            'calibration_mode': 'none',
            'warning_threshold': {'absolute_bil': 10, 'percent': 1.0, 'combination': 'max'},
            'red_threshold': {'absolute_bil': 25, 'percent': 2.5, 'combination': 'max'},
            'allowed_calibration_modes': [
                'none',
                'opening_book_reconciliation',
                'named_component_bridge',
                'bounded_curve_shape_fit',
                'reporting_only_cbo_overlay',
            ],
            'modeled_interest_affects_cash_and_issuance': True,
            'cbo_reported_interest_affects_cash_and_issuance': False,
        },
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


@pytest.mark.parametrize('config_name', [
    'tdc_config.yaml',
    'tdc_config_optional.yaml',
    'tdc_config_ratewall_source_backed.yaml',
])
def test_shipped_configs_validate_with_root_key_allowlist(config_name):
    cfg_path = Path(__file__).resolve().parent.parent / config_name
    with cfg_path.open('r') as f:
        cfg = yaml.safe_load(f)
    errors = validate_config(cfg)
    assert errors == []


def test_cbo_phase0_config_sketch_blocks_validate_without_runtime_semantics():
    cfg = cbo_phase0_blocks()
    cfg['simulation_period'] = {
        'start_date': '2026-04-30',
        'end_date': '2036-09-30',
        'frequency': 'W',
        'insert_control_dates': True,
        'control_dates': ['09-30'],
        'interval_convention': '[start, end)',
    }
    assert validate_config(cfg) == []


def test_cbo_phase0_unknown_root_key_rejected_by_validate_config():
    errors = validate_config({'funding_rul': {'mode': 'cbo_public_debt_target'}})
    assert any('Configuration root contains unknown keys' in e and 'funding_rul' in e for e in errors)


@pytest.mark.parametrize('block_name', [
    'funding_rule',
    'baseline_input_paths',
    'data_vintage',
    'operating_cash_policy',
    'public_debt_bridge',
    'fiscal_baseline',
    'fiscal_incidence_policy',
    'budget_interest',
])
def test_cbo_phase0_unknown_nested_key_rejected_in_each_new_block(block_name):
    cfg = {block_name: dict(cbo_phase0_blocks()[block_name])}
    cfg[block_name]['typo_key'] = 'bad'
    errors = validate_config(cfg)
    assert any(block_name in e and 'typo_key' in e for e in errors), errors


def test_cbo_phase0_unknown_deep_nested_key_rejected():
    cfg = {'budget_interest': dict(cbo_phase0_blocks()['budget_interest'])}
    cfg['budget_interest']['warning_threshold'] = {
        'absolute_bil': 10,
        'percent': 1.0,
        'combination': 'max',
        'typo_threshold': True,
    }
    errors = validate_config(cfg)
    assert any('budget_interest.warning_threshold' in e and 'typo_threshold' in e for e in errors), errors


def test_cbo_phase0_new_blocks_allowed_in_scenario_and_group_overrides():
    config = {
        'scenario_groups': [
            {
                'group_name': 'CBO admission',
                'overrides': {'data_vintage': cbo_phase0_blocks()['data_vintage']},
                'scenarios': [
                    {
                        'name': 'Debt target',
                        'overrides': {'funding_rule': cbo_phase0_blocks()['funding_rule']},
                    }
                ],
            }
        ],
    }
    assert validate_config(config) == []


def test_cbo_phase0_group_override_unknown_key_rejected():
    errors = validate_config({
        'scenario_groups': [
            {
                'group_name': 'Bad group override',
                'overrides': {'funding_rul': {'mode': 'cbo_public_debt_target'}},
                'scenarios': [{'name': 'baseline'}],
            }
        ],
    })
    assert any('unknown override keys' in e.lower() and 'funding_rul' in e for e in errors), errors


def test_cbo_phase0_admitted_blocks_are_runtime_inert_in_cash_mode():
    baseline = minimal_params()
    cbo_admitted = minimal_params()
    cbo_admitted.update(cbo_phase0_blocks())
    cbo_admitted['funding_rule'] = {'mode': 'cash_tga_target'}

    baseline_results, _ = run_simulation(baseline, '2025-01-01', '2025-01-19', freq='W', scenario_name='baseline')
    cbo_results, _ = run_simulation(cbo_admitted, '2025-01-01', '2025-01-19', freq='W', scenario_name='cbo_admitted')

    assert first_active_period(cbo_results)['AuctionProceeds'] == pytest.approx(
        first_active_period(baseline_results)['AuctionProceeds'], rel=1e-6
    )
    assert cbo_results.attrs['run_metadata']['validation_status'] == 'passed'


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


@pytest.mark.parametrize('counterparty', ['Banks', 'CB', 'Foreign', 'FedInternal', 'TrustFunds'])
@pytest.mark.parametrize('private_role, expected_sign', [('seller', 1), ('buyer', -1)])
def test_private_ru_secondary_trade_deposit_sign_matches_du_ru_identity(counterparty, private_role, expected_sign):
    """Private-as-seller to any RU raises DU deposits; Private-as-buyer from any RU lowers them."""
    seller = 'Private' if private_role == 'seller' else counterparty
    buyer = counterparty if private_role == 'seller' else 'Private'
    portfolio = pd.DataFrame([
        make_bond_row(
            BondID=201,
            HolderType=seller,
            FaceValue=100.0,
            CouponRate=0.04,
            IssueDate=pd.Timestamp('2025-01-01'),
            MaturityDate=pd.Timestamp('2027-01-01'),
            OriginalMaturityYears=2.0,
            MaturityCategory='notes',
        ),
        make_bond_row(
            BondID=202,
            HolderType=buyer,
            FaceValue=100.0,
            CouponRate=0.0,
            IssueDate=pd.Timestamp('2025-01-01'),
            MaturityDate=pd.Timestamp('2025-07-01'),
            OriginalMaturityYears=0.5,
            MaturityCategory='bills',
            IssueYieldAtIssue=0.04,
            IssuePriceRatio=0.98,
            IssueProceeds=98.0,
        ),
    ], columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES, errors='ignore')
    prefs = {
        holder: {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0, 'tips_pct': 0.0, 'frn_pct': 0.0}
        for holder in ['Banks', 'CB', 'Foreign', 'FedInternal', 'TrustFunds', 'Private']
    }
    prefs[buyer]['notes_pct'] = 1.0

    _, impact = execute_preference_trades(
        portfolio,
        pd.Timestamp('2025-03-01'),
        [0.25, 0.5, 1.0, 2.0, 5.0],
        [0.04, 0.041, 0.042, 0.043, 0.045],
        prefs,
        {
            'bills': {'category_cutoff_years': 1.0},
            'notes': {'category_cutoff_years': 10.0},
            'bonds': {'category_cutoff_years': 999.0},
        },
        f'private_{private_role}_{counterparty}',
    )

    assert impact['deposit_change'] * expected_sign > 0.0



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


def test_config_derived_generated_portfolio_uses_config_curve_and_maturities():
    base_config = minimal_params()
    base_config['treasury_issuance_profile'] = {
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
    base_config['yield_curve'] = {
        'use_static': True,
        'years': [0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
        'rates': [0.02, 0.03, 0.05, 0.055, 0.06, 0.065],
    }
    base_config['sector_preferences'] = {
        'Banks': {'bills_pct': 1.0, 'notes_pct': 1.0, 'bonds_pct': 1.0},
        'Private': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
        'CB': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
        'Foreign': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
        'FedInternal': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
        'TrustFunds': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0},
    }

    df = generate_initial_portfolio(
        {
            'generation_method': 'config_derived',
            'target_public_marketable_wam': 1.0,
            'wam_targeting_iterations': 1,
            'random_seed': 7,
            'target_face_values_billions': {
                'Banks': 0.05,
                'Private_Marketable': 0.0,
                'Private_NonMarketable': 0.0,
                'CB': 0.0,
                'Foreign': 0.0,
                'FedInternal': 0.0,
                'TrustFunds_NonMarketable': 0.0,
            },
        },
        '2025-01-01',
        base_config=base_config,
    )

    assert not df.empty
    assert set(df['OriginalMaturityYears'].astype(float).unique()) == {1.0}
    assert ((df['IssueYieldAtIssue'] - 0.05).abs() < 1e-6).all()


def test_legacy_generator_ignores_optional_base_config_when_generation_method_legacy():
    gen_config = {
        'generation_method': 'legacy',
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
    }
    base_config = minimal_params()
    base_config['yield_curve']['rates'] = [0.10] * len(base_config['yield_curve']['rates'])

    legacy_plain = generate_initial_portfolio(gen_config, '2025-01-01')
    legacy_with_base = generate_initial_portfolio(gen_config, '2025-01-01', base_config=base_config)
    pdt.assert_frame_equal(legacy_plain.reset_index(drop=True), legacy_with_base.reset_index(drop=True), check_dtype=False)


def test_disabled_optional_blocks_preserve_simulation_results():
    baseline = minimal_params()
    disabled = copy.deepcopy(baseline)
    disabled['rate_sensitive_demand'] = {
        'enabled': False,
        'min_multiplier': 0.25,
        'auction': {},
        'secondary': {},
    }
    disabled['financing_cost_options'] = {'include_tips_inflation_accretion': False}

    baseline_results, baseline_portfolio = run_simulation(
        baseline, '2025-01-01', '2025-03-01', freq='W', scenario_name='baseline_disabled_compare'
    )
    disabled_results, disabled_portfolio = run_simulation(
        disabled, '2025-01-01', '2025-03-01', freq='W', scenario_name='disabled_optional_compare'
    )

    pdt.assert_frame_equal(baseline_results, disabled_results, check_dtype=False, check_like=False)
    pdt.assert_frame_equal(
        baseline_portfolio.reset_index(drop=True),
        disabled_portfolio.reset_index(drop=True),
        check_dtype=False,
        check_like=False,
    )
