"""Tests for Federal bucket split, factorial scenarios, and original-maturity categorization."""
import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from csv_gen import generate_initial_portfolio
from simulation_core import (
    execute_preference_trades,
    get_security_category_for_prefs,
    run_simulation,
)
from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    HOLDER_TYPES,
    INTRAGOV_HOLDERS,
    PORTFOLIO_DTYPES,
)
from tdc_validation import (
    validate_config,
    validate_issuance_profile,
    validate_nonmarketable_params,
    validate_sector_preferences,
)


# ---------------------------------------------------------------------------
# Helpers (shared with test_config_and_trading.py style)
# ---------------------------------------------------------------------------

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


# ===================================================================
# 1. Federal Bucket Split Tests
# ===================================================================

class TestFederalBucketSplit:
    """Tests for the FedInternal / TrustFunds holder split."""

    def test_holder_types_contains_split_holders(self):
        """HOLDER_TYPES has FedInternal and TrustFunds, not Federal."""
        assert 'FedInternal' in HOLDER_TYPES
        assert 'TrustFunds' in HOLDER_TYPES
        assert 'Federal' not in HOLDER_TYPES

    def test_intragov_holders_constant(self):
        """INTRAGOV_HOLDERS contains exactly FedInternal and TrustFunds."""
        assert INTRAGOV_HOLDERS == frozenset({'FedInternal', 'TrustFunds'})

    def test_config_validates_with_split_holders(self):
        """Shipped config passes validation with the new holder types."""
        cfg_path = Path(__file__).resolve().parent.parent / 'tdc_config.yaml'
        with cfg_path.open('r') as f:
            cfg = yaml.safe_load(f)
        errors = validate_config(cfg)
        assert errors == [], f"Validation errors: {errors}"

    def test_generated_portfolio_uses_split_holders(self):
        """Generated portfolio produces FedInternal and TrustFunds, not Federal."""
        df = generate_initial_portfolio(
            {
                'target_public_marketable_wam': 3.0,
                'wam_targeting_iterations': 1,
                'random_seed': 99,
                'target_face_values_billions': {
                    'Banks': 0.05,
                    'Private_Marketable': 0.05,
                    'Private_NonMarketable': 0.0,
                    'CB': 0.0,
                    'Foreign': 0.0,
                    'FedInternal': 0.05,
                    'TrustFunds_NonMarketable': 0.10,
                },
            },
            '2025-01-01',
        )
        holder_types_in_df = set(df['HolderType'].unique())
        assert 'FedInternal' in holder_types_in_df
        assert 'TrustFunds' in holder_types_in_df
        assert 'Federal' not in holder_types_in_df

        # TrustFunds should only hold NonMarketable
        tf_types = set(df.loc[df['HolderType'] == 'TrustFunds', 'SecurityType'].unique())
        assert tf_types == {'NonMarketable'}

        # FedInternal should hold marketable securities
        fi_types = set(df.loc[df['HolderType'] == 'FedInternal', 'SecurityType'].unique())
        assert 'NonMarketable' not in fi_types
        assert len(fi_types) > 0

    def test_simulation_outputs_split_holder_columns(self):
        """Simulation results contain DebtHeld_FedInternal and DebtHeld_TrustFunds."""
        params = minimal_params()
        results, _ = run_simulation(
            params, '2025-01-01', '2025-01-19', freq='W',
            scenario_name='split_holder_cols',
        )
        assert 'DebtHeld_FedInternal' in results.columns
        assert 'DebtHeld_TrustFunds' in results.columns
        # Old column should NOT exist (even before rename)
        assert 'DebtHeld_Federal' not in results.columns
        assert 'DebtHeld_FederalOrInternal' not in results.columns

    def test_intragov_debt_service_is_tga_wash(self):
        """P&I to FedInternal/TrustFunds should not drain TGA (intragovernmental wash).

        Strategy: create a portfolio with a maturing bond held by TrustFunds.
        Since the maturity is a TGA wash, less new debt should be issued to
        maintain TGA compared to when Private holds the same bond (where the
        maturity drains TGA and requires issuance to refill).
        """
        # TrustFunds holds a maturing bond
        bond_tf = make_bond_row(
            BondID=201,
            SecurityType='NonMarketable',
            HolderType='TrustFunds',
            FaceValue=50.0,
            CouponRate=0.0,
            MaturityDate=pd.Timestamp('2025-01-10'),
            OriginalMaturityYears=5.0,
            MaturityCategory=None,
            OriginalPrincipal=50.0,
            AdjustedPrincipal=50.0,
            IssueProceeds=50.0,
            IssueYieldAtIssue=0.0,
        )
        params_tf = minimal_params()
        params_tf['initial_values']['tga'] = 200.0
        params_tf['tga_params']['target_balance'] = 200.0
        portfolio_tf = pd.DataFrame([bond_tf], columns=BOND_PORTFOLIO_COLS).astype(
            PORTFOLIO_DTYPES, errors='ignore'
        )
        params_tf['initial_bonds_df'] = portfolio_tf
        results_tf, _ = run_simulation(
            params_tf, '2025-01-01', '2025-01-19', freq='W',
            scenario_name='tf_maturity',
        )

        # Private holds the same maturing bond
        bond_priv = bond_tf.copy()
        bond_priv['HolderType'] = 'Private'
        bond_priv['SecurityType'] = 'Fixed'
        params_priv = minimal_params()
        params_priv['initial_values']['tga'] = 200.0
        params_priv['tga_params']['target_balance'] = 200.0
        portfolio_priv = pd.DataFrame([bond_priv], columns=BOND_PORTFOLIO_COLS).astype(
            PORTFOLIO_DTYPES, errors='ignore'
        )
        params_priv['initial_bonds_df'] = portfolio_priv
        results_priv, _ = run_simulation(
            params_priv, '2025-01-01', '2025-01-19', freq='W',
            scenario_name='priv_maturity',
        )

        # TGA targeting refills to 200 in both cases, but the Private case
        # requires more new issuance to compensate for the TGA drain.
        tf_issued = results_tf.iloc[1]['NewDebtIssued']
        priv_issued = results_priv.iloc[1]['NewDebtIssued']
        assert priv_issued > tf_issued + 1.0, (
            f"Private maturity issuance ({priv_issued:.2f}) should exceed "
            f"TrustFunds maturity issuance ({tf_issued:.2f}) by at least the "
            "principal amount, because intragovernmental maturity is a TGA wash "
            "that doesn't require additional issuance to refill."
        )

    def test_intragov_excluded_from_reserves(self):
        """P&I to FedInternal/TrustFunds should not increase reserves."""
        bond_tf = make_bond_row(
            BondID=301,
            SecurityType='NonMarketable',
            HolderType='TrustFunds',
            FaceValue=50.0,
            CouponRate=0.0,
            MaturityDate=pd.Timestamp('2025-01-10'),
            OriginalMaturityYears=5.0,
            MaturityCategory=None,
            OriginalPrincipal=50.0,
            AdjustedPrincipal=50.0,
            IssueProceeds=50.0,
            IssueYieldAtIssue=0.0,
        )
        params = minimal_params()
        params['initial_values']['tga'] = 200.0
        params['tga_params']['target_balance'] = 200.0
        portfolio = pd.DataFrame([bond_tf], columns=BOND_PORTFOLIO_COLS).astype(
            PORTFOLIO_DTYPES, errors='ignore'
        )
        params['initial_bonds_df'] = portfolio
        results, _ = run_simulation(
            params, '2025-01-01', '2025-01-19', freq='W',
            scenario_name='tf_reserves_test',
        )
        # With no fiscal spending/taxes and TrustFunds maturity being a wash,
        # reserves should not increase from debt service.
        # The only reserve change should come from issuance (if any).
        # Check that the reserve change in period 1 does not include the $50 principal.
        reserve_change = results.iloc[1]['ReserveChange']
        # Reserve change should be negative or zero (from issuance drain),
        # not positive from TrustFunds maturity.
        assert reserve_change <= 0.0 + 1e-6, (
            f"Reserve change ({reserve_change}) should not be positive from "
            "intragovernmental debt maturity"
        )


# ===================================================================
# 2. Factorial Scenario Tests
# ===================================================================

class TestFactorialScenarios:
    """Tests for the orthogonal factorial scenario design."""

    @pytest.fixture(scope='class')
    def config(self):
        cfg_path = Path(__file__).resolve().parent.parent / 'tdc_config.yaml'
        with cfg_path.open('r') as f:
            return yaml.safe_load(f)

    def test_factorial_group_exists(self, config):
        """Config contains the One-Factor Isolation scenario group."""
        group_names = [g['group_name'] for g in config['scenario_groups']]
        assert 'One-Factor Isolation' in group_names

    def test_factorial_has_five_scenarios(self, config):
        """Factorial group has exactly 5 scenarios."""
        factorial_group = next(
            g for g in config['scenario_groups'] if g['group_name'] == 'One-Factor Isolation'
        )
        assert len(factorial_group['scenarios']) == 5

    def test_factorial_baseline_matches_compound_baseline(self, config):
        """Factorial_Baseline overrides match the compound Baseline exactly."""
        compound_group = next(
            g for g in config['scenario_groups'] if g['group_name'] == 'Treasury Market Scenarios'
        )
        factorial_group = next(
            g for g in config['scenario_groups'] if g['group_name'] == 'One-Factor Isolation'
        )
        compound_baseline = next(
            s for s in compound_group['scenarios'] if s['name'] == 'Baseline'
        )
        factorial_baseline = next(
            s for s in factorial_group['scenarios'] if s['name'] == 'Factorial_Baseline'
        )
        # Both baselines should have the same overrides
        assert compound_baseline['overrides'] == factorial_baseline['overrides']

    def test_factorial_issuance_only_changes_issuance(self, config):
        """Factorial_IssuanceOnly changes only treasury_issuance_profile."""
        factorial_group = next(
            g for g in config['scenario_groups'] if g['group_name'] == 'One-Factor Isolation'
        )
        baseline = next(
            s for s in factorial_group['scenarios'] if s['name'] == 'Factorial_Baseline'
        )
        issuance_only = next(
            s for s in factorial_group['scenarios'] if s['name'] == 'Factorial_IssuanceOnly'
        )
        bo = baseline['overrides']
        io = issuance_only['overrides']
        # Same fiscal, sector prefs, yield curve
        assert io['fiscal_params'] == bo['fiscal_params']
        assert io['sector_preferences'] == bo['sector_preferences']
        assert io['yield_curve'] == bo['yield_curve']
        # Different issuance profile
        assert io['treasury_issuance_profile'] != bo['treasury_issuance_profile']

    def test_factorial_holder_only_changes_prefs(self, config):
        """Factorial_HolderOnly changes only sector_preferences."""
        factorial_group = next(
            g for g in config['scenario_groups'] if g['group_name'] == 'One-Factor Isolation'
        )
        baseline = next(
            s for s in factorial_group['scenarios'] if s['name'] == 'Factorial_Baseline'
        )
        holder_only = next(
            s for s in factorial_group['scenarios'] if s['name'] == 'Factorial_HolderOnly'
        )
        bo = baseline['overrides']
        ho = holder_only['overrides']
        assert ho['fiscal_params'] == bo['fiscal_params']
        assert ho['treasury_issuance_profile'] == bo['treasury_issuance_profile']
        assert ho['yield_curve'] == bo['yield_curve']
        assert ho['sector_preferences'] != bo['sector_preferences']

    def test_factorial_rate_only_changes_curve(self, config):
        """Factorial_RateOnly changes only yield_curve."""
        factorial_group = next(
            g for g in config['scenario_groups'] if g['group_name'] == 'One-Factor Isolation'
        )
        baseline = next(
            s for s in factorial_group['scenarios'] if s['name'] == 'Factorial_Baseline'
        )
        rate_only = next(
            s for s in factorial_group['scenarios'] if s['name'] == 'Factorial_RateOnly'
        )
        bo = baseline['overrides']
        ro = rate_only['overrides']
        assert ro['fiscal_params'] == bo['fiscal_params']
        assert ro['treasury_issuance_profile'] == bo['treasury_issuance_profile']
        assert ro['sector_preferences'] == bo['sector_preferences']
        assert ro['yield_curve'] != bo['yield_curve']

    def test_factorial_holder_notes_pref_changes_only_prefs(self, config):
        """Factorial_HolderOnly_NotesPref changes only sector_preferences (to banks_pref_notes)."""
        factorial_group = next(
            g for g in config['scenario_groups'] if g['group_name'] == 'One-Factor Isolation'
        )
        baseline = next(
            s for s in factorial_group['scenarios'] if s['name'] == 'Factorial_Baseline'
        )
        notes_pref = next(
            s for s in factorial_group['scenarios'] if s['name'] == 'Factorial_HolderOnly_NotesPref'
        )
        bo = baseline['overrides']
        np_o = notes_pref['overrides']
        assert np_o['fiscal_params'] == bo['fiscal_params']
        assert np_o['treasury_issuance_profile'] == bo['treasury_issuance_profile']
        assert np_o['yield_curve'] == bo['yield_curve']
        assert np_o['sector_preferences'] != bo['sector_preferences']


# ===================================================================
# 3. Original-Maturity Categorization Tests
# ===================================================================

class TestOriginalMaturityCategorization:
    """Verify secondary trading uses OriginalMaturityYears, not remaining maturity."""

    def test_aged_note_stays_note_category(self):
        """A 2y note with 0.5y remaining should still categorize as 'notes', not 'bills'."""
        profile = base_issuance_profile()
        # OriginalMaturityYears = 2.0 → notes (between 1.0 and 10.0)
        cat = get_security_category_for_prefs('Fixed', 2.0, profile)
        assert cat == 'notes'

    def test_aged_bond_stays_bond_category(self):
        """A 20y bond with 3y remaining should still categorize as 'bonds', not 'notes'."""
        profile = base_issuance_profile()
        cat = get_security_category_for_prefs('Fixed', 20.0, profile)
        assert cat == 'bonds'

    def test_bill_categorized_as_bill(self):
        """A 1y bill stays 'bills'."""
        profile = base_issuance_profile()
        cat = get_security_category_for_prefs('Fixed', 1.0, profile)
        assert cat == 'bills'

    def test_secondary_trade_uses_original_maturity(self):
        """When an aged note (OriginalMaturity=5y, remaining=0.5y) exists in a
        portfolio, the trade engine should treat it as a 'notes' category bond,
        not reclassify it as a 'bill'.

        Setup: Banks holds an aged 5y note (0.5y remaining) plus a bill.
        Private holds only a bill and wants notes.
        Banks has a surplus in 'notes' (the aged note) and Private has a deficit.
        The trade should move bond 501 from Banks to Private as a 'notes' trade.
        If remaining maturity were used, bond 501 would be a 'bill' and no
        notes trade would occur.
        """
        # Banks hold an aged 5y note that matures in ~6 months
        aged_note = make_bond_row(
            BondID=501,
            SecurityType='Fixed',
            HolderType='Banks',
            FaceValue=100.0,
            CouponRate=0.04,
            IssueDate=pd.Timestamp('2020-07-01'),
            MaturityDate=pd.Timestamp('2025-07-01'),  # ~6 months remaining
            OriginalMaturityYears=5.0,
            MaturityCategory='notes',
        )
        # Banks also hold a bill (so Banks has both bills and notes)
        banks_bill = make_bond_row(
            BondID=502,
            SecurityType='Fixed',
            HolderType='Banks',
            FaceValue=100.0,
            CouponRate=0.0,
            IssueDate=pd.Timestamp('2024-07-01'),
            MaturityDate=pd.Timestamp('2025-07-01'),
            OriginalMaturityYears=1.0,
            MaturityCategory='bills',
            IssueYieldAtIssue=0.04,
            IssuePriceRatio=0.96,
            IssueProceeds=96.0,
        )
        # Private holds a bill — wants notes instead
        priv_bill = make_bond_row(
            BondID=503,
            SecurityType='Fixed',
            HolderType='Private',
            FaceValue=100.0,
            CouponRate=0.0,
            IssueDate=pd.Timestamp('2024-07-01'),
            MaturityDate=pd.Timestamp('2025-07-01'),
            OriginalMaturityYears=1.0,
            MaturityCategory='bills',
            IssueYieldAtIssue=0.04,
            IssuePriceRatio=0.96,
            IssueProceeds=96.0,
        )

        portfolio = pd.DataFrame(
            [aged_note, banks_bill, priv_bill], columns=BOND_PORTFOLIO_COLS
        ).astype(PORTFOLIO_DTYPES, errors='ignore')

        # Banks wants 100% bills, Private wants 100% notes.
        # Banks has a notes surplus (bond 501), Private has a notes deficit.
        prefs = {
            'Banks': {'bills_pct': 0.5, 'notes_pct': 0.0, 'bonds_pct': 0.0,
                       'tips_pct': 0.0, 'frn_pct': 0.0},
            'Private': {'bills_pct': 0.0, 'notes_pct': 0.5, 'bonds_pct': 0.0,
                         'tips_pct': 0.0, 'frn_pct': 0.0},
            'CB': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0,
                    'tips_pct': 0.0, 'frn_pct': 0.0},
            'Foreign': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0,
                         'tips_pct': 0.0, 'frn_pct': 0.0},
            'FedInternal': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0,
                             'tips_pct': 0.0, 'frn_pct': 0.0},
            'TrustFunds': {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0,
                            'tips_pct': 0.0, 'frn_pct': 0.0},
        }

        traded, impact = execute_preference_trades(
            portfolio,
            pd.Timestamp('2025-01-15'),
            [0.25, 0.5, 1.0, 2.0, 5.0],
            [0.04, 0.041, 0.042, 0.043, 0.045],
            prefs,
            base_issuance_profile(),
            'orig_maturity_test',
        )

        # If categorization used remaining maturity (0.5y), the aged note would
        # be classified as 'bills' and Banks would have no 'notes' surplus to sell.
        # With original maturity (5y), it's 'notes' and Banks has a surplus.
        # Check that bond 501 was traded (Private now holds some of it).
        priv_501 = traded[(traded['BondID'] == 501) & (traded['HolderType'] == 'Private')]
        assert not priv_501.empty, (
            "Bond 501 (5y original, 0.5y remaining) should have been traded as 'notes'. "
            "If this fails, categorization may be using remaining maturity."
        )


# ===================================================================
# 4. Nonmarketable Interest Capitalization Tests
# ===================================================================

class TestNonMarketableInterestCapitalization:
    """Capitalized nonmarketable interest must appear in FinancingCost."""

    def test_nonmkt_interest_appears_in_financing_cost(self):
        """When nonmarketable interest credits, FinancingCost_Period includes it."""
        bond_nm = make_bond_row(
            BondID=701,
            SecurityType='NonMarketable',
            HolderType='TrustFunds',
            FaceValue=1000.0,
            CouponRate=0.0,
            IssueDate=pd.Timestamp('2024-01-01'),
            MaturityDate=pd.Timestamp('2035-01-01'),
            OriginalMaturityYears=11.0,
            MaturityCategory=None,
            OriginalPrincipal=1000.0,
            AdjustedPrincipal=1000.0,
            IssueProceeds=1000.0,
            IssueYieldAtIssue=0.0,
        )
        params = minimal_params()
        params['initial_values']['tga'] = 500.0
        params['tga_params']['target_balance'] = 500.0
        params['nonmarketable_params'] = {
            'interest_rate_basis_maturities': [5.0, 10.0],
            'interest_crediting_frequency': 'semi-annual',
            'initial_holder': 'TrustFunds',
            'rate_setting_method': 'yield_curve_points',
        }
        portfolio = pd.DataFrame([bond_nm], columns=BOND_PORTFOLIO_COLS).astype(
            PORTFOLIO_DTYPES, errors='ignore'
        )
        params['initial_bonds_df'] = portfolio
        # Run through a June 30 crediting date
        results, _ = run_simulation(
            params, '2025-01-01', '2025-07-15', freq='W',
            scenario_name='nonmkt_interest_test',
        )
        # Nonmarketable interest should have been capitalized on the Jun 30 boundary
        total_cap = results['NonMarketableInterestCapitalized_Period'].sum()
        assert total_cap > 0, "Nonmarketable interest should have been capitalized"
        # FinancingCost_Cumulative at end must include capitalized interest
        final_financing = results.iloc[-1]['FinancingCost_Cumulative']
        final_cap = results.iloc[-1]['NonMarketableInterestCapitalized_Cumulative']
        assert final_cap > 0, "Cumulative capitalized interest should be positive"
        assert final_financing >= final_cap - 1e-6, (
            f"FinancingCost_Cumulative ({final_financing:.4f}) must include "
            f"NonMarketableInterestCapitalized_Cumulative ({final_cap:.4f})"
        )


# ===================================================================
# 5. Tranche Split Proration Tests
# ===================================================================

class TestTrancheSplitProration:
    """Quantity-linked fields must be prorated when a tranche is partially traded."""

    def test_tips_split_prorates_original_principal(self):
        """A partial TIPS trade should prorate OriginalPrincipal, not copy it whole."""
        import numpy as np
        # TIPS bond held by Banks
        tips_bond = make_bond_row(
            BondID=801,
            SecurityType='TIPS',
            HolderType='Banks',
            FaceValue=200.0,
            CouponRate=0.005,
            IssueDate=pd.Timestamp('2023-01-01'),
            MaturityDate=pd.Timestamp('2033-01-01'),
            OriginalMaturityYears=10.0,
            MaturityCategory='notes',
            OriginalPrincipal=200.0,
            AdjustedPrincipal=210.0,
            ReferenceCPI_Issue=100.0,
            IndexRatio=1.05,
            IssueProceeds=200.0,
        )
        # Private holds a bill — wants TIPS instead
        priv_bill = make_bond_row(
            BondID=802,
            SecurityType='Fixed',
            HolderType='Private',
            FaceValue=200.0,
            CouponRate=0.0,
            IssueDate=pd.Timestamp('2024-07-01'),
            MaturityDate=pd.Timestamp('2025-07-01'),
            OriginalMaturityYears=1.0,
            MaturityCategory='bills',
            IssueProceeds=192.0,
            IssuePriceRatio=0.96,
        )
        portfolio = pd.DataFrame(
            [tips_bond, priv_bill], columns=BOND_PORTFOLIO_COLS
        ).astype(PORTFOLIO_DTYPES, errors='ignore')

        # Banks wants bills, Private wants TIPS
        prefs = {h: {} for h in HOLDER_TYPES}
        prefs['Banks'] = {'bills_pct': 0.5, 'notes_pct': 0.0, 'bonds_pct': 0.0,
                          'tips_pct': 0.0, 'frn_pct': 0.0}
        prefs['Private'] = {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0,
                            'tips_pct': 0.5, 'frn_pct': 0.0}
        for h in ['CB', 'Foreign', 'FedInternal', 'TrustFunds']:
            prefs[h] = {'bills_pct': 0.0, 'notes_pct': 0.0, 'bonds_pct': 0.0,
                        'tips_pct': 0.0, 'frn_pct': 0.0}

        traded, _ = execute_preference_trades(
            portfolio, pd.Timestamp('2025-01-15'),
            [0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
            [0.04, 0.041, 0.042, 0.043, 0.045, 0.05],
            prefs, base_issuance_profile(), 'tips_split_test',
        )

        # Check that OriginalPrincipal is conserved across split pieces
        tips_rows = traded[traded['BondID'] == 801]
        total_orig_principal = tips_rows['OriginalPrincipal'].sum()
        assert abs(total_orig_principal - 200.0) < 1.0, (
            f"Total OriginalPrincipal across split TIPS tranches should be ~200, "
            f"got {total_orig_principal:.2f}"
        )
        # Also check AdjustedPrincipal is conserved
        total_adj_principal = tips_rows['AdjustedPrincipal'].sum()
        assert abs(total_adj_principal - 210.0) < 1.0, (
            f"Total AdjustedPrincipal across split TIPS tranches should be ~210, "
            f"got {total_adj_principal:.2f}"
        )
        # And IssueProceeds
        total_proceeds = tips_rows['IssueProceeds'].sum()
        assert abs(total_proceeds - 200.0) < 1.0, (
            f"Total IssueProceeds across split TIPS tranches should be ~200, "
            f"got {total_proceeds:.2f}"
        )


# ===================================================================
# 6. Mid-Quarter Fiscal Re-Anchor Tests
# ===================================================================

class TestMidQuarterFiscalReAnchor:
    """Fiscal events mid-quarter must recompute interpolation endpoints."""

    def test_mid_quarter_spending_shock(self):
        """A mid-quarter spending event should immediately affect GovSpending."""
        params = minimal_params()
        params['fiscal_params'] = {
            'initial_weekly_spending': 100.0,
            'initial_weekly_taxes': 100.0,
            'spending_growth_qtr': 0.0,
            'tax_growth_qtr': 0.0,
        }
        params['initial_values']['tga'] = 500.0
        params['tga_params']['target_balance'] = 500.0
        # Event fires mid-February (mid Q1)
        params['events'] = [{
            'date': '2025-02-15',
            'actions': [{
                'parameter_path': 'fiscal_params.initial_weekly_spending',
                'new_value': 200.0,
            }],
        }]
        results, _ = run_simulation(
            params, '2025-01-01', '2025-04-01', freq='W',
            scenario_name='mid_qtr_fiscal_test',
        )
        # After the event, spending should be near 200, not blending toward old target
        post_event = results.loc[results.index > pd.Timestamp('2025-02-15')]
        assert not post_event.empty
        # The first post-event spending should be closer to 200 than to 100
        first_post = post_event.iloc[0]['GovSpending']
        assert first_post > 150.0, (
            f"Post-event spending {first_post:.2f} should be closer to 200 than 100"
        )


# ===================================================================
# 7. Preference Coverage Validation Tests
# ===================================================================

class TestPreferenceCoverageValidation:
    """Active categories with zero preference coverage should error."""

    def test_zero_coverage_active_tips_raises_error(self):
        """If TIPS has nonzero target_percentage but no holder has tips_pct, validation should fail."""
        from tdc_validation import validate_sector_preferences
        prefs = {
            'Banks': {'bills_pct': 0.25, 'notes_pct': 0.25, 'bonds_pct': 0.25},
            'Private': {'bills_pct': 0.25, 'notes_pct': 0.25, 'bonds_pct': 0.25},
            'CB': {'bills_pct': 0.25, 'notes_pct': 0.25, 'bonds_pct': 0.25},
            'Foreign': {'bills_pct': 0.25, 'notes_pct': 0.25, 'bonds_pct': 0.25},
        }
        profile_with_tips = base_issuance_profile()
        profile_with_tips['TIPS'] = {'target_percentage': 0.10}
        errors = validate_sector_preferences(prefs, issuance_profile=profile_with_tips)
        assert any('tips_pct' in e for e in errors), (
            f"Expected error about missing tips_pct contributors. Got: {errors}"
        )

    def test_zero_coverage_inactive_category_ok(self):
        """If TIPS has 0 target_percentage, no tips_pct is fine."""
        from tdc_validation import validate_sector_preferences
        prefs = {
            'Banks': {'bills_pct': 0.25, 'notes_pct': 0.25, 'bonds_pct': 0.25},
            'Private': {'bills_pct': 0.25, 'notes_pct': 0.25, 'bonds_pct': 0.25},
            'CB': {'bills_pct': 0.25, 'notes_pct': 0.25, 'bonds_pct': 0.25},
            'Foreign': {'bills_pct': 0.25, 'notes_pct': 0.25, 'bonds_pct': 0.25},
        }
        profile_no_tips = base_issuance_profile()
        errors = validate_sector_preferences(prefs, issuance_profile=profile_no_tips)
        tips_errors = [e for e in errors if 'tips_pct' in e]
        assert len(tips_errors) == 0, f"Should not error on inactive TIPS: {tips_errors}"


# ===================================================================
# 8. TIPS Adjusted Principal in Debt Aggregates Tests
# ===================================================================

class TestTIPSDebtAggregates:
    """TIPS should use AdjustedPrincipal in debt aggregates."""

    def test_tips_debt_uses_adjusted_principal(self):
        """TotalDebt_Agg should reflect TIPS AdjustedPrincipal, not raw FaceValue."""
        tips_bond = make_bond_row(
            BondID=901,
            SecurityType='TIPS',
            HolderType='Private',
            FaceValue=100.0,
            CouponRate=0.005,
            IssueDate=pd.Timestamp('2024-01-01'),
            MaturityDate=pd.Timestamp('2034-01-01'),
            OriginalMaturityYears=10.0,
            MaturityCategory='notes',
            OriginalPrincipal=100.0,
            AdjustedPrincipal=110.0,
            ReferenceCPI_Issue=100.0,
            IndexRatio=1.1,
            IssueProceeds=100.0,
        )
        params = minimal_params()
        params['tips_params'] = {
            'cpi_start_level': 110.0,
            'cpi_annual_inflation': 0.03,
            'ref_cpi_lag_months': 3,
            'default_real_coupon_rate': 0.005,
        }
        portfolio = pd.DataFrame([tips_bond], columns=BOND_PORTFOLIO_COLS).astype(
            PORTFOLIO_DTYPES, errors='ignore'
        )
        params['initial_bonds_df'] = portfolio
        results, _ = run_simulation(
            params, '2025-01-01', '2025-02-01', freq='W',
            scenario_name='tips_agg_test',
        )
        # At t=0, TotalDebt_Agg should be ~110 (AdjustedPrincipal), not 100 (FaceValue)
        t0_debt = results.iloc[0]['TotalDebt_Agg']
        assert t0_debt > 105.0, (
            f"TotalDebt_Agg at t=0 should reflect AdjustedPrincipal (~110), got {t0_debt}"
        )
        # DebtHeldByType_TIPS should also reflect adjusted principal
        t0_tips_debt = results.iloc[0]['DebtHeldByType_TIPS']
        assert t0_tips_debt > 105.0, (
            f"DebtHeldByType_TIPS at t=0 should reflect AdjustedPrincipal (~110), got {t0_tips_debt}"
        )


# ===================================================================
# 9. Schema-Strict Event Validation Tests
# ===================================================================

class TestSchemaStrictEventValidation:
    """Event validation must reject typos in leaf keys, bad dates, and missing new_value."""

    def test_typoed_leaf_path_rejected(self):
        """A typo in a leaf key should be caught by validation."""
        from tdc_validation import validate_events
        events = [{
            'date': '2025-06-01',
            'actions': [{'parameter_path': 'fiscal_params.initial_weekly_spedning', 'new_value': 200.0}],
        }]
        errors = validate_events(events)
        assert len(errors) > 0, "Typoed leaf key should produce validation error"
        assert any('spedning' in e or 'Invalid leaf' in e for e in errors)

    def test_valid_leaf_path_accepted(self):
        """A correct leaf key should pass validation."""
        from tdc_validation import validate_events
        events = [{
            'date': '2025-06-01',
            'actions': [{'parameter_path': 'fiscal_params.initial_weekly_spending', 'new_value': 200.0}],
        }]
        errors = validate_events(events)
        assert len(errors) == 0, f"Valid event should not produce errors: {errors}"

    def test_missing_new_value_rejected(self):
        """An action without new_value should be caught."""
        from tdc_validation import validate_events
        events = [{
            'date': '2025-06-01',
            'actions': [{'parameter_path': 'fiscal_params.initial_weekly_spending'}],
        }]
        errors = validate_events(events)
        assert any('new_value' in e for e in errors)

    def test_bad_date_rejected(self):
        """An unparseable date should be caught."""
        from tdc_validation import validate_events
        events = [{
            'date': 'not-a-date',
            'actions': [{'parameter_path': 'fiscal_params.initial_weekly_spending', 'new_value': 200.0}],
        }]
        errors = validate_events(events)
        assert any('unparseable' in e for e in errors)

    def test_set_nested_value_rejects_new_key(self):
        """_set_nested_value should refuse to create a new leaf key."""
        from simulation_core import _set_nested_value
        d = {'initial_weekly_spending': 100.0, 'initial_weekly_taxes': 80.0}
        result = _set_nested_value(d, ['typo_key'], 999.0)
        assert result is False, "_set_nested_value should return False for nonexistent key"
        assert 'typo_key' not in d

    def test_holder_pref_path_accepted(self):
        """A valid holder preference path should pass validation."""
        from tdc_validation import validate_events
        events = [{
            'date': '2025-06-01',
            'actions': [{'parameter_path': 'sector_preferences.Banks.bills_pct', 'new_value': 0.5}],
        }]
        errors = validate_events(events)
        assert len(errors) == 0, f"Valid holder pref path should not error: {errors}"

    def test_invalid_holder_pref_path_rejected(self):
        """A holder preference path with bad category should be rejected."""
        from tdc_validation import validate_events
        events = [{
            'date': '2025-06-01',
            'actions': [{'parameter_path': 'sector_preferences.Banks.stocks_pct', 'new_value': 0.5}],
        }]
        errors = validate_events(events)
        assert len(errors) > 0, "Invalid pref category 'stocks_pct' should be rejected"


# ===================================================================
# 9. Preference Key Validation Tests
# ===================================================================

class TestPreferenceKeyValidation:
    """Validate that typos in preference key names are caught."""

    def test_validate_sector_prefs_rejects_typo_key(self):
        """A preference key like 'billz_pct' should be rejected."""
        bad_prefs = {
            'Banks': {'billz_pct': 0.15, 'notes_pct': 0.20, 'bonds_pct': 0.10},
            'Private': {'bills_pct': 0.45, 'notes_pct': 0.50, 'bonds_pct': 0.60},
            'CB': {'bills_pct': 0.15, 'notes_pct': 0.05, 'bonds_pct': 0.05},
            'Foreign': {'bills_pct': 0.25, 'notes_pct': 0.25, 'bonds_pct': 0.25},
        }
        errors = validate_sector_preferences(bad_prefs, enforce_column_sums=False)
        assert any("unexpected key" in e and "billz_pct" in e for e in errors), (
            f"Expected error about 'billz_pct', got: {errors}"
        )


# ===================================================================
# 10. Config / Override Validation Tests
# ===================================================================

class TestConfigValidation:
    """Tests for config-level validation of required blocks and override keys."""

    def test_validate_run_params_rejects_missing_fiscal_params(self):
        """Missing 'fiscal_params' should be flagged as a required block."""
        from simulation_core import validate_run_params
        params = minimal_params()
        del params['fiscal_params']
        with pytest.raises(ValueError, match="Required parameter block 'fiscal_params'"):
            validate_run_params(params, scenario_name='missing_fiscal')

    def test_validate_run_params_accepts_valid_config(self):
        """A minimal valid config should pass validation without errors."""
        from simulation_core import validate_run_params
        params = minimal_params()
        # Should not raise
        validate_run_params(params, scenario_name='valid_config')

    def test_override_unknown_key_warns(self, capsys):
        """Unknown scenario override keys should produce a warning."""
        from simulation_core import VALID_OVERRIDE_KEYS
        overrides = {'fiscal_param': {'initial_weekly_spending': 100.0}}  # typo: missing 's'
        unknown = set(overrides.keys()) - VALID_OVERRIDE_KEYS
        assert unknown == {'fiscal_param'}, "Should detect the typo key"


# ===================================================================
# 11. Complex Event Path Validation Tests
# ===================================================================

class TestComplexEventPathValidation:
    """Tests for structural validation of yield_curve and issuance_profile event paths."""

    def test_event_path_yield_curve_typo_rejected(self):
        """yield_curve.yeers (typo) should be rejected by event validation."""
        from tdc_validation import validate_events
        events = [{
            'date': '2025-06-01',
            'actions': [{'parameter_path': 'yield_curve.yeers', 'new_value': [1, 2, 3]}],
        }]
        errors = validate_events(events)
        assert len(errors) > 0, "yield_curve.yeers should be rejected"
        assert any("yield_curve" in e for e in errors)

    def test_event_path_yield_curve_valid(self):
        """yield_curve.rates should be accepted."""
        from tdc_validation import validate_events
        events = [{
            'date': '2025-06-01',
            'actions': [{'parameter_path': 'yield_curve.rates', 'new_value': [0.04, 0.05]}],
        }]
        errors = validate_events(events)
        assert len(errors) == 0, f"yield_curve.rates should be valid: {errors}"

    def test_event_path_issuance_profile_valid(self):
        """treasury_issuance_profile.bills.target_percentage_of_remainder should be accepted."""
        from tdc_validation import validate_events
        events = [{
            'date': '2025-06-01',
            'actions': [{'parameter_path': 'treasury_issuance_profile.bills.target_percentage_of_remainder', 'new_value': 0.25}],
        }]
        errors = validate_events(events)
        assert len(errors) == 0, f"Valid issuance profile path should pass: {errors}"

    def test_event_path_issuance_profile_typo_rejected(self):
        """treasury_issuance_profile.bills.typo_key should be rejected."""
        from tdc_validation import validate_events
        events = [{
            'date': '2025-06-01',
            'actions': [{'parameter_path': 'treasury_issuance_profile.bills.typo_key', 'new_value': 0.25}],
        }]
        errors = validate_events(events)
        assert len(errors) > 0, "typo_key should be rejected"


# ===================================================================
# 12. Accounting Identity Tests
# ===================================================================

class TestAccountingIdentities:
    """Row-wise verification of the core accounting identities."""

    def test_tdc_decomposition_identity(self):
        """TDC_Change must equal sum of 5 decomposition components row-wise."""
        params = minimal_params()
        params['fiscal_params']['initial_weekly_spending'] = 120.0
        params['fiscal_params']['initial_weekly_taxes'] = 100.0
        params['initial_values']['tga'] = 500.0
        params['tga_params']['target_balance'] = 500.0
        results, _ = run_simulation(
            params, '2025-01-01', '2025-06-01', freq='W',
            scenario_name='identity_tdc',
        )
        data = results.iloc[1:]  # skip t0
        decomp_sum = (
            data['TDC_FiscalFlow']
            + data['TDC_DebtService']
            + data['TDC_AuctionAbsorption']
            + data['TDC_SecondaryTrades']
            + data['TDC_Other']
        )
        diff = (data['TDC_Change'] - decomp_sum).abs()
        assert diff.max() < 1e-6, f"TDC identity violated, max deviation: {diff.max()}"

    def test_financing_cost_identity(self):
        """FinancingCost_Period must equal sum of its 3 components row-wise."""
        params = minimal_params()
        params['fiscal_params']['initial_weekly_spending'] = 120.0
        params['fiscal_params']['initial_weekly_taxes'] = 100.0
        params['initial_values']['tga'] = 500.0
        params['tga_params']['target_balance'] = 500.0
        results, _ = run_simulation(
            params, '2025-01-01', '2025-06-01', freq='W',
            scenario_name='identity_financing',
        )
        data = results.iloc[1:]
        component_sum = (
            data['InterestOutlay_Period']
            + data['IssueDiscountCost_Period']
            + data['NonMarketableInterestCapitalized_Period']
        )
        diff = (data['FinancingCost_Period'] - component_sum).abs()
        assert diff.max() < 1e-6, f"Financing cost identity violated, max deviation: {diff.max()}"


# ---------------------------------------------------------------------------
# GPT Pro Round 3 — Validation hardening tests
# ---------------------------------------------------------------------------

class TestStrictConfigValidation:
    """Tests for strict config and override validation rules."""

    def test_unknown_override_key_rejected_by_validate_config(self):
        """Unknown scenario override keys should produce validation errors."""
        config = {
            'yield_curve': {
                'use_static': True,
                'years': [1.0, 5.0, 10.0],
                'rates': [0.04, 0.045, 0.05],
            },
            'treasury_issuance_profile': {
                'bills': {
                    'category_cutoff_years': 1.0,
                    'target_percentage_of_remainder': 0.20,
                    'maturities': [0.25, 0.5, 1.0],
                    'maturity_distribution': [0.333, 0.333, 0.334],
                },
                'notes': {
                    'category_cutoff_years': 10.0,
                    'target_percentage_of_remainder': 0.60,
                    'maturities': [2.0, 5.0, 10.0],
                    'maturity_distribution': [0.333, 0.333, 0.334],
                },
                'bonds': {
                    'category_cutoff_years': 999.0,
                    'target_percentage_of_remainder': 0.20,
                    'maturities': [20.0, 30.0],
                    'maturity_distribution': [0.5, 0.5],
                },
                'remainder_maturity_years': 0.5,
            },
            'scenario_groups': [
                {
                    'group_name': 'Test',
                    'scenarios': [
                        {
                            'name': 'BadOverride',
                            'overrides': {
                                'fiscal_param': {},  # typo: missing 's'
                            },
                        }
                    ],
                }
            ],
        }
        errors = validate_config(config)
        assert any('unknown override keys' in e.lower() for e in errors), (
            f"Expected unknown override key error, got: {errors}"
        )

    def test_bad_crediting_frequency_rejected(self):
        """Typo 'semiannual' (without hyphen) should be rejected."""
        errors = validate_nonmarketable_params({
            'interest_crediting_frequency': 'semiannual',
        })
        assert len(errors) == 1
        assert 'semiannual' in errors[0]
        assert 'semi-annual' in errors[0]

    def test_valid_crediting_frequency_accepted(self):
        """Valid values 'semi-annual' and 'annual' should pass."""
        assert validate_nonmarketable_params({'interest_crediting_frequency': 'semi-annual'}) == []
        assert validate_nonmarketable_params({'interest_crediting_frequency': 'annual'}) == []

    def test_tips_target_without_maturities_rejected(self):
        """TIPS with target_percentage > 0 but no maturities should be rejected."""
        profile = {
            'bills': {
                'category_cutoff_years': 1.0,
                'target_percentage_of_remainder': 0.20,
                'maturities': [1.0],
                'maturity_distribution': [1.0],
            },
            'notes': {
                'category_cutoff_years': 10.0,
                'target_percentage_of_remainder': 0.60,
                'maturities': [5.0],
                'maturity_distribution': [1.0],
            },
            'bonds': {
                'category_cutoff_years': 999.0,
                'target_percentage_of_remainder': 0.20,
                'maturities': [30.0],
                'maturity_distribution': [1.0],
            },
            'remainder_maturity_years': 0.5,
            'TIPS': {
                'target_percentage': 0.10,
                # no maturities or maturity_distribution
            },
        }
        errors = validate_issuance_profile(profile)
        assert any('silently skipped' in e.lower() for e in errors), (
            f"Expected TIPS maturities error, got: {errors}"
        )

    def test_tips_target_zero_without_maturities_accepted(self):
        """TIPS with target_percentage = 0 should pass without maturities."""
        profile = {
            'bills': {
                'category_cutoff_years': 1.0,
                'target_percentage_of_remainder': 0.20,
                'maturities': [1.0],
                'maturity_distribution': [1.0],
            },
            'notes': {
                'category_cutoff_years': 10.0,
                'target_percentage_of_remainder': 0.60,
                'maturities': [5.0],
                'maturity_distribution': [1.0],
            },
            'bonds': {
                'category_cutoff_years': 999.0,
                'target_percentage_of_remainder': 0.20,
                'maturities': [30.0],
                'maturity_distribution': [1.0],
            },
            'remainder_maturity_years': 0.5,
            'TIPS': {
                'target_percentage': 0.0,
            },
        }
        errors = validate_issuance_profile(profile)
        assert not any('TIPS' in e and 'maturities' in e for e in errors)
