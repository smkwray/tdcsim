"""
Initial bond portfolio generator for the TDC Simulator.

Can be used as:
  - Importable function: generate_initial_portfolio(gen_config, sim_start_date) → DataFrame
  - Standalone script: python csv_gen.py (loads tdc_config.yaml, generates and saves CSV)
"""
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import random
import numpy as np
import os
import yaml
from tdc_shared import BOND_PORTFOLIO_COLS, ISSUANCE_PROFILE_CUTOFFS

# --- Constants ---
SECTOR_PREFERENCES = {
  'Banks':       { 'bills_pct': 0.300, 'notes_pct': 0.400, 'bonds_pct': 0.150, 'tips_pct': 0.100, 'frn_pct': 0.050 },
  'Private':     { 'bills_pct': 0.200, 'notes_pct': 0.350, 'bonds_pct': 0.250, 'tips_pct': 0.150, 'frn_pct': 0.050 },
  'CB':          { 'bills_pct': 0.060, 'notes_pct': 0.400, 'bonds_pct': 0.305, 'tips_pct': 0.185, 'frn_pct': 0.050 },
  'Foreign':     { 'bills_pct': 0.250, 'notes_pct': 0.400, 'bonds_pct': 0.250, 'tips_pct': 0.050, 'frn_pct': 0.050 },
  'FedInternal': { 'bills_pct': 0.400, 'notes_pct': 0.400, 'bonds_pct': 0.100, 'tips_pct': 0.050, 'frn_pct': 0.050 },
  'TrustFunds':  { 'bills_pct': 0.000, 'notes_pct': 0.000, 'bonds_pct': 0.000, 'tips_pct': 0.000, 'frn_pct': 0.000 },
}

DEFAULT_TARGET_FACE_VALUES = {
    'Banks': 4400.0,
    'Private_Marketable': 10200.0,
    'Private_NonMarketable': 700.0,
    'CB': 4600.0,
    'Foreign': 8100.0,
    'FedInternal': 200.0,
    'TrustFunds_NonMarketable': 6800.0,
}

MIN_FACE_VALUE_BILLIONS_MARKETABLE = 0.001
MIN_FACE_VALUE_BILLIONS_NONMARKETABLE_PRIVATE = 0.0001
MIN_FACE_VALUE_BILLIONS_NONMARKETABLE_FEDERAL = 0.01

# ISSUANCE_PROFILE_CUTOFFS imported from tdc_shared

MATURITY_YEAR_OPTIONS = {
    'bills': [0.25, 0.5, 1.0],
    'notes': [2.0, 3.0, 5.0, 7.0, 10.0],
    'bonds': [20.0, 30.0],
    'tips': [5.0, 10.0, 30.0],
    'frn': [2.0],
    'nonmarketable_private': [5.0, 10.0, 20.0, 30.0],
    'nonmarketable_federal': [10.0, 15.0, 20.0, 25.0, 30.0]
}

MATURITY_YEAR_WEIGHTS = {
    'bills': [0.40, 0.40, 0.20],
    'notes': [0.25, 0.25, 0.30, 0.15, 0.05],
    'bonds': [0.70, 0.30],
    'tips':  [0.50, 0.30, 0.20],
    'frn':   [1.0],
    'nonmarketable_private': [0.3, 0.3, 0.2, 0.2],
    'nonmarketable_federal': [0.2, 0.2, 0.3, 0.2, 0.1]
}

NUM_TRANCHES_PER_TYPE = {
    'bills': 1000,
    'notes': 750,
    'bonds': 250,
    'tips': 200,
    'frn': 100,
    'nonmarketable_private': 50,
    'nonmarketable_federal': 50
}


def _get_maturity_category(maturity_years):
    if maturity_years <= ISSUANCE_PROFILE_CUTOFFS['bills_cutoff_years']: return 'bills'
    elif maturity_years <= ISSUANCE_PROFILE_CUTOFFS['notes_cutoff_years']: return 'notes'
    else: return 'bonds'


def _get_fixed_coupon_at_issue(maturity_years):
    if maturity_years <= 1.0: return 0.00
    elif maturity_years <= 3.0: return random.uniform(0.005, 0.015)
    elif maturity_years <= 7.0: return random.uniform(0.010, 0.020)
    elif maturity_years <= 10.0: return random.uniform(0.015, 0.025)
    else: return random.uniform(0.020, 0.035)


def _issue_yield_at_issue(maturity_years, security_type, coupon_rate=0.0):
    if security_type == 'Fixed' and maturity_years <= 1.0:
        # Bills should be issued at a discount. Use a simple upward-sloping range.
        return random.uniform(0.035, 0.055)
    if security_type == 'Fixed':
        return float(coupon_rate)
    if security_type == 'TIPS':
        return float(coupon_rate)
    if security_type == 'FRN':
        return 0.0
    return float(coupon_rate) if coupon_rate is not None else 0.0


def _issue_price_ratio(security_type, maturity_years, coupon_rate, issue_yield):
    if security_type == 'Fixed' and maturity_years <= 1.0 and float(coupon_rate) <= 1.0e-12:
        return 1.0 / ((1.0 + max(0.0, float(issue_yield))) ** float(maturity_years))
    return 1.0


def _get_random_issue_date(sim_start_date):
    start_issue_date = datetime(2020, 1, 1)
    end_issue_date = sim_start_date - timedelta(days=1)
    days_between = (end_issue_date - start_issue_date).days
    if days_between <= 0: return start_issue_date
    return start_issue_date + timedelta(days=random.randrange(days_between))


def _generate_one_portfolio(target_face_values, sim_start_date):
    """Generate a single portfolio attempt. Returns (DataFrame, bond_id_counter)."""
    bond_id_counter = 100
    all_bonds_data = []

    for holder_key_target in target_face_values.keys():
        if holder_key_target.startswith('Private_'): holder_type_csv = 'Private'
        elif holder_key_target.startswith('TrustFunds_'): holder_type_csv = 'TrustFunds'
        else: holder_type_csv = holder_key_target

        if holder_key_target == 'TrustFunds_NonMarketable' or holder_key_target == 'Private_NonMarketable':
            target_nonmarketable_value = target_face_values[holder_key_target]
            nm_type_key = 'nonmarketable_federal' if holder_key_target == 'TrustFunds_NonMarketable' else 'nonmarketable_private'
            min_face_value_nm = MIN_FACE_VALUE_BILLIONS_NONMARKETABLE_FEDERAL if nm_type_key == 'nonmarketable_federal' else MIN_FACE_VALUE_BILLIONS_NONMARKETABLE_PRIVATE
            num_tranches_nm = NUM_TRANCHES_PER_TYPE[nm_type_key]
            maturity_options_nm = MATURITY_YEAR_OPTIONS[nm_type_key]
            weights_nm = MATURITY_YEAR_WEIGHTS.get(nm_type_key)
            if target_nonmarketable_value <= 0 or num_tranches_nm == 0: continue
            face_per_nm_tranche = target_nonmarketable_value / num_tranches_nm
            for _ in range(num_tranches_nm):
                bond_id_counter += 1
                bond = {col: None for col in BOND_PORTFOLIO_COLS}
                bond['BondID'] = bond_id_counter; bond['SecurityType'] = 'NonMarketable'; bond['HolderType'] = holder_type_csv; bond['Status'] = 'Active'
                maturity_date_dt = datetime(1900,1,1)
                while maturity_date_dt <= sim_start_date:
                    issue_dt = _get_random_issue_date(sim_start_date)
                    if weights_nm and len(weights_nm) == len(maturity_options_nm): orig_mat_yrs = random.choices(maturity_options_nm, weights=weights_nm, k=1)[0]
                    else: orig_mat_yrs = random.choice(maturity_options_nm)
                    years_int = int(orig_mat_yrs); months_int = int(round((orig_mat_yrs - years_int) * 12))
                    maturity_date_dt = issue_dt + relativedelta(years=years_int, months=months_int)
                bond['IssueDate'] = issue_dt; bond['OriginalMaturityYears'] = orig_mat_yrs; bond['MaturityDate'] = maturity_date_dt
                rounding_decimals = 4 if nm_type_key == 'nonmarketable_private' else 2
                bond['FaceValue'] = round(face_per_nm_tranche, rounding_decimals)
                if bond['FaceValue'] < min_face_value_nm: bond['FaceValue'] = min_face_value_nm
                bond['CouponRate'] = 0.0; bond['OriginalPrincipal'] = bond['FaceValue']; bond['AdjustedPrincipal'] = bond['FaceValue']; bond['IndexRatio'] = 1.0
                bond['IssueYieldAtIssue'] = 0.0
                bond['IssuePriceRatio'] = 1.0
                bond['IssueProceeds'] = bond['FaceValue']
                all_bonds_data.append(bond)
            continue

        holder_total_target = target_face_values.get(holder_key_target, 0.0)
        if holder_total_target <= 0: continue
        prefs = SECTOR_PREFERENCES.get(holder_type_csv, {})
        for sec_cat_key_with_pct, cat_prefs_pct in prefs.items():
            target_fv_cat = holder_total_target * cat_prefs_pct
            if target_fv_cat < MIN_FACE_VALUE_BILLIONS_MARKETABLE: continue
            cat_key = sec_cat_key_with_pct.replace('_pct', '')
            if cat_key == 'bills': actual_sec_type = 'Fixed'; mat_opts = MATURITY_YEAR_OPTIONS['bills']; num_tranches = NUM_TRANCHES_PER_TYPE['bills']; weights = MATURITY_YEAR_WEIGHTS['bills']
            elif cat_key == 'notes': actual_sec_type = 'Fixed'; mat_opts = MATURITY_YEAR_OPTIONS['notes']; num_tranches = NUM_TRANCHES_PER_TYPE['notes']; weights = MATURITY_YEAR_WEIGHTS['notes']
            elif cat_key == 'bonds': actual_sec_type = 'Fixed'; mat_opts = MATURITY_YEAR_OPTIONS['bonds']; num_tranches = NUM_TRANCHES_PER_TYPE['bonds']; weights = MATURITY_YEAR_WEIGHTS['bonds']
            elif cat_key == 'tips': actual_sec_type = 'TIPS'; mat_opts = MATURITY_YEAR_OPTIONS['tips']; num_tranches = NUM_TRANCHES_PER_TYPE['tips']; weights = MATURITY_YEAR_WEIGHTS['tips']
            elif cat_key == 'frn': actual_sec_type = 'FRN'; mat_opts = MATURITY_YEAR_OPTIONS['frn']; num_tranches = NUM_TRANCHES_PER_TYPE['frn']; weights = MATURITY_YEAR_WEIGHTS['frn']
            else: continue
            if not mat_opts or num_tranches == 0: continue
            num_tranches = max(1, num_tranches); face_per_tranche = target_fv_cat / num_tranches
            if face_per_tranche < MIN_FACE_VALUE_BILLIONS_MARKETABLE and target_fv_cat >= MIN_FACE_VALUE_BILLIONS_MARKETABLE:
                num_tranches = int(target_fv_cat / MIN_FACE_VALUE_BILLIONS_MARKETABLE); num_tranches = max(1, num_tranches)
                face_per_tranche = target_fv_cat / num_tranches
            if face_per_tranche < MIN_FACE_VALUE_BILLIONS_MARKETABLE: continue
            for _ in range(num_tranches):
                bond_id_counter += 1
                bond = {col: None for col in BOND_PORTFOLIO_COLS}
                bond['BondID'] = bond_id_counter; bond['SecurityType'] = actual_sec_type; bond['HolderType'] = holder_type_csv; bond['Status'] = 'Active'
                maturity_date_dt = datetime(1900,1,1)
                while maturity_date_dt <= sim_start_date:
                    issue_dt = _get_random_issue_date(sim_start_date)
                    if weights and len(weights) == len(mat_opts): orig_mat_yrs = random.choices(mat_opts, weights=weights, k=1)[0]
                    else: orig_mat_yrs = random.choice(mat_opts)
                    years_int = int(orig_mat_yrs); months_int = int(round((orig_mat_yrs - years_int) * 12))
                    maturity_date_dt = issue_dt + relativedelta(years=years_int, months=months_int)
                bond['IssueDate'] = issue_dt; bond['OriginalMaturityYears'] = orig_mat_yrs; bond['MaturityDate'] = maturity_date_dt
                bond['FaceValue'] = round(face_per_tranche, 3)
                if bond['FaceValue'] < MIN_FACE_VALUE_BILLIONS_MARKETABLE: bond['FaceValue'] = MIN_FACE_VALUE_BILLIONS_MARKETABLE
                bond['OriginalPrincipal'] = bond['FaceValue']; bond['AdjustedPrincipal'] = bond['FaceValue']
                if actual_sec_type == 'Fixed':
                    bond['CouponRate'] = _get_fixed_coupon_at_issue(bond['OriginalMaturityYears'])
                    bond['MaturityCategory'] = _get_maturity_category(bond['OriginalMaturityYears'])
                    bond['IndexRatio'] = 1.0
                elif actual_sec_type == 'TIPS':
                    bond['CouponRate'] = random.uniform(0.001, 0.0075)
                    bond['ReferenceCPI_Issue'] = round(random.uniform(280.0, 310.0), 1)
                    bond['IndexRatio'] = 1.0
                elif actual_sec_type == 'FRN':
                    bond['CouponRate'] = 0.0
                    bond['FixedSpread'] = random.uniform(0.0005, 0.0030)
                    bond['AccruedInterest_FRN'] = 0.0
                    bond['BenchmarkRate_FRN'] = 0.0
                    bond['LastAccrualDate'] = issue_dt
                    bond['IndexRatio'] = 1.0
                issue_yield = _issue_yield_at_issue(bond['OriginalMaturityYears'], actual_sec_type, bond.get('CouponRate', 0.0))
                issue_price_ratio = _issue_price_ratio(actual_sec_type, bond['OriginalMaturityYears'], bond.get('CouponRate', 0.0), issue_yield)
                bond['IssueYieldAtIssue'] = issue_yield
                bond['IssuePriceRatio'] = issue_price_ratio
                bond['IssueProceeds'] = bond['FaceValue'] * issue_price_ratio
                all_bonds_data.append(bond)

    df = pd.DataFrame(all_bonds_data, columns=BOND_PORTFOLIO_COLS)
    df['FaceValue'] = pd.to_numeric(df['FaceValue'], errors='coerce').fillna(0.0)
    df['MaturityDate'] = pd.to_datetime(df['MaturityDate'], errors='coerce')
    return df


def generate_initial_portfolio(gen_config, sim_start_date):
    """
    Generate an initial bond portfolio using WAM targeting.

    Parameters:
        gen_config: dict from config's initial_portfolio.generation (or initial_portfolio_generation)
        sim_start_date: datetime — simulation start date

    Returns:
        pd.DataFrame with columns matching BOND_PORTFOLIO_COLS
    """
    if isinstance(sim_start_date, str):
        sim_start_date = datetime.fromisoformat(sim_start_date)

    random_seed = gen_config.get('random_seed')
    if random_seed is not None:
        random.seed(int(random_seed))
        np.random.seed(int(random_seed))

    target_wam = gen_config.get('target_public_marketable_wam', 6.0)
    iterations = gen_config.get('wam_targeting_iterations', 300)
    target_face_values = gen_config.get('target_face_values_billions', DEFAULT_TARGET_FACE_VALUES)

    best_portfolio_df = None
    closest_wam_diff = float('inf')
    achieved_wam = 0.0

    print(f"--- Starting WAM Targeting ({iterations} iterations, Target WAM: {target_wam:.2f} years) ---")

    for i in range(iterations):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Generating portfolio attempt {i+1}/{iterations}...")

        df = _generate_one_portfolio(target_face_values, sim_start_date)

        # Calculate public marketable WAM
        public_holders = ['Banks', 'Private', 'Foreign']
        marketable_types = ['Fixed', 'TIPS', 'FRN']
        wam_df = df[
            (df['SecurityType'].isin(marketable_types)) &
            (df['HolderType'].isin(public_holders))
        ].copy()

        current_wam = 0.0
        if not wam_df.empty:
            wam_df['TTM'] = (wam_df['MaturityDate'] - sim_start_date).dt.total_seconds() / (365.25 * 24 * 60 * 60)
            active = wam_df[wam_df['TTM'] > 1e-9]
            if not active.empty:
                total_fv = active['FaceValue'].sum()
                if total_fv > 0:
                    current_wam = (active['FaceValue'] * active['TTM']).sum() / total_fv

        diff = abs(current_wam - target_wam)
        if diff < closest_wam_diff:
            closest_wam_diff = diff
            best_portfolio_df = df.copy()
            achieved_wam = current_wam

    if best_portfolio_df is None:
        print("WARNING: No portfolio generated. Returning empty DataFrame.")
        return pd.DataFrame(columns=BOND_PORTFOLIO_COLS)

    print(f"--- Selected portfolio: WAM={achieved_wam:.2f} (target={target_wam:.2f}, diff={closest_wam_diff:.2f}) ---")
    return best_portfolio_df


def save_portfolio_csv(df, filepath):
    """Save a portfolio DataFrame to CSV with proper formatting."""
    df_out = df.copy()
    for col in ['IssueDate', 'MaturityDate', 'LastAccrualDate']:
        df_out[col] = pd.to_datetime(df_out[col], errors='coerce').dt.strftime('%Y-%m-%d')
        df_out[col] = df_out[col].replace('NaT', '')

    for col in df_out.columns:
        if col in ['FaceValue', 'CouponRate', 'OriginalMaturityYears',
                   'OriginalPrincipal', 'AdjustedPrincipal', 'ReferenceCPI_Issue',
                   'IndexRatio', 'FixedSpread', 'AccruedInterest_FRN', 'BenchmarkRate_FRN',
                   'IssuePriceRatio', 'IssueProceeds', 'IssueYieldAtIssue']:
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce').fillna(0.0)
            if col in ['FaceValue', 'OriginalPrincipal', 'AdjustedPrincipal', 'AccruedInterest_FRN']:
                 df_out[col] = df_out[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else '')
            elif col in ['CouponRate', 'FixedSpread', 'BenchmarkRate_FRN', 'IndexRatio', 'IssuePriceRatio', 'IssueYieldAtIssue']:
                 df_out[col] = df_out[col].apply(lambda x: f"{x:.6f}" if pd.notnull(x) else '')
            elif col == 'IssueProceeds':
                 df_out[col] = df_out[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else '')
            elif col == 'ReferenceCPI_Issue':
                 df_out[col] = df_out[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) and x != 0.0 else ('0.0' if x == 0.0 else ''))
            elif col == 'OriginalMaturityYears':
                 df_out[col] = df_out[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else '')
        elif col == 'BondID':
            pass
        elif col not in ['IssueDate', 'MaturityDate', 'LastAccrualDate']:
            df_out[col] = df_out[col].fillna('')

    for calc_col in ['TimeToMaturity', 'DiscountYield', 'CleanPrice', 'AccruedInterest', 'DirtyValue', 'DirtyPriceRatio']:
        if calc_col not in df_out.columns: df_out[calc_col] = ''
        else: df_out[calc_col] = df_out[calc_col].fillna('').astype(str)

    df_out.to_csv(filepath, index=False, lineterminator='\n')
    print(f"CSV saved to: {os.path.abspath(filepath)} ({len(df_out)} bonds)")


def print_portfolio_summary(df, target_face_values=None):
    """Print summary statistics for a generated portfolio."""
    print("\n--- Face Value by Holder (Billions USD) ---")
    for holder, total_fv in df.groupby('HolderType')['FaceValue'].sum().items():
        print(f"  {holder}: {total_fv:.4f}")

    print("\n--- Face Value by Security Type (Billions USD) ---")
    for sec_type, total_fv in df.groupby('SecurityType')['FaceValue'].sum().items():
        print(f"  {sec_type}: {total_fv:.4f}")

    total = df['FaceValue'].sum()
    print(f"\n  Total: {total:.4f}")
    if target_face_values:
        expected = sum(target_face_values.values())
        print(f"  Expected: {expected:.4f}")
        print(f"  Difference: {total - expected:.4f}")


# --- Standalone execution ---
if __name__ == "__main__":
    try:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        config_path = os.path.join(project_root, 'tdc_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
    except Exception as e:
        print(f"Warning: Could not load tdc_config.yaml ({e}). Using defaults.")
        config = {}

    # Support both old and new config structure
    portfolio_config = config.get('initial_portfolio', {})
    gen_config = portfolio_config.get('generation', config.get('initial_portfolio_generation', {}))

    sim_period = config.get('simulation_period', {})
    sim_start = sim_period.get('start_date', '2025-01-01')

    target_fv = gen_config.get('target_face_values_billions', DEFAULT_TARGET_FACE_VALUES)
    output_filename = gen_config.get('output_filename', 'initial_bond_portfolio.csv')

    df = generate_initial_portfolio(gen_config, sim_start)

    if not df.empty:
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            abs_output_dir = os.path.join(project_root, output_dir)
            os.makedirs(abs_output_dir, exist_ok=True)
        filepath = os.path.join(project_root, output_filename)
        save_portfolio_csv(df, filepath)
        print_portfolio_summary(df, target_fv)
    else:
        print("No portfolio generated.")
