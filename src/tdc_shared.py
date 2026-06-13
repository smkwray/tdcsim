# tdc_shared.py
# Shared definitions for the TDC simulator to ensure consistency between generation and simulation.

# --- Core Columns ---
BOND_PORTFOLIO_COLS = [
    'BondID', 'SecurityType', 'IssueDate', 'MaturityDate',
    'OriginalMaturityYears', 'FaceValue', 'CouponRate', 'HolderType', 'HolderSubBucket', 'Status',
    'MaturityCategory',
    'OriginalPrincipal', 'AdjustedPrincipal', 'ReferenceCPI_Issue', 'IndexRatio',
    'FixedSpread', 'AccruedInterest_FRN', 'BenchmarkRate_FRN', 'LastAccrualDate',
    'IssuePriceRatio', 'IssueProceeds', 'IssueYieldAtIssue',
    'TimeToMaturity', 'DiscountYield', 'CleanPrice', 'AccruedInterest', 'DirtyValue', 'DirtyPriceRatio'
]

# --- Categories ---
HOLDER_TYPES = ['Banks', 'CB', 'Foreign', 'FedInternal', 'TrustFunds', 'Private']
INTRAGOV_HOLDERS = frozenset({'FedInternal', 'TrustFunds'})
SECURITY_TYPES = ['Fixed', 'TIPS', 'FRN', 'NonMarketable']
MATURITY_CATEGORIES = ['bills', 'notes', 'bonds']
PREFERENCE_CATEGORIES = ['bills', 'notes', 'bonds', 'tips', 'frn', 'nonmarketable']
PRIVATE_SUBBUCKET_DOMESTIC_NONBANK = 'domestic_nonbank_deposit_funded'
PRIVATE_SUBBUCKET_MMF = 'mmf_cash_fund_route'
PRIVATE_SUBBUCKETS = [
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
]
MMF_DEPOSIT_PASS_THROUGH_DEFAULT = 0.15
MMF_DEPOSIT_PASS_THROUGH_STATUS = 'source_grounded_model_default_not_measured'
MMF_DEPOSIT_PASS_THROUGH_SENSITIVITY_GRID = [0.00, 0.15, 0.25, 0.50, 1.00]

# --- Global Constants ---
TGA_FLOOR_TOLERANCE = 1.0e-9
DAYS_PER_YEAR_ACTUAL = 365.25
FRN_DAY_COUNT_BASIS = 360.0

# --- Data Types ---
PORTFOLIO_DTYPES = {
    'BondID': 'Int64',
    'SecurityType': 'string',
    'IssueDate': 'datetime64[ns]',
    'MaturityDate': 'datetime64[ns]',
    'OriginalMaturityYears': 'float64',
    'FaceValue': 'float64',
    'CouponRate': 'float64',
    'HolderType': 'string',
    'HolderSubBucket': 'string',
    'Status': 'string',
    'MaturityCategory': 'string',
    'OriginalPrincipal': 'float64',
    'AdjustedPrincipal': 'float64',
    'ReferenceCPI_Issue': 'float64',
    'IndexRatio': 'float64',
    'FixedSpread': 'float64',
    'AccruedInterest_FRN': 'float64',
    'BenchmarkRate_FRN': 'float64',
    'LastAccrualDate': 'datetime64[ns]',
    'IssuePriceRatio': 'float64',
    'IssueProceeds': 'float64',
    'IssueYieldAtIssue': 'float64',
    'TimeToMaturity': 'float64',
    'DiscountYield': 'float64',
    'CleanPrice': 'float64',
    'AccruedInterest': 'float64',
    'DirtyValue': 'float64',
    'DirtyPriceRatio': 'float64'
}

# --- Issuance Defaults ---
ISSUANCE_PROFILE_CUTOFFS = {
    'bills_cutoff_years': 1.0,
    'notes_cutoff_years': 10.0
}
