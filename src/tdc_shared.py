# tdc_shared.py
# Shared definitions for the TDC simulator to ensure consistency between generation and simulation.

# --- Core Columns ---
BOND_PORTFOLIO_COLS = [
    'BondID', 'IssueID', 'SecurityType', 'IssueDate', 'MaturityDate',
    'DatedDate', 'OriginalDatedDate', 'FirstInterestPaymentDate', 'InterestPaymentFrequency',
    'OriginalMaturityYears', 'FaceValue', 'CouponRate', 'HolderType', 'HolderSubBucket',
    'TDCPrincipalHolderType', 'TDCPrincipalHolderSubBucket', 'Status',
    'MaturityCategory',
    'OriginalPrincipal', 'AdjustedPrincipal', 'ReferenceCPI_Issue', 'IndexRatio',
    'FixedSpread', 'AccruedInterest_FRN', 'BenchmarkRate_FRN', 'LastAccrualDate',
    'IssuePriceRatio', 'IssueProceeds', 'IssueYieldAtIssue',
    'TimeToMaturity', 'DiscountYield', 'CleanPrice', 'AccruedInterest', 'DirtyValue', 'DirtyPriceRatio'
]

# --- Categories ---
HOLDER_TYPES = ['Banks', 'CB', 'Foreign', 'FedInternal', 'TrustFunds', 'Private']
# Stable event serialization order. This is presentation policy, not a second
# taxonomy: it must contain each canonical holder exactly once.
HOLDER_EVENT_ORDER = ('Banks', 'CB', 'Foreign', 'Private', 'TrustFunds', 'FedInternal')
if set(HOLDER_EVENT_ORDER) != set(HOLDER_TYPES) or len(HOLDER_EVENT_ORDER) != len(HOLDER_TYPES):
    raise RuntimeError('HOLDER_EVENT_ORDER must be a permutation of HOLDER_TYPES')
INTRAGOV_HOLDERS = frozenset({'FedInternal', 'TrustFunds'})
SECURITY_TYPES = ['Fixed', 'TIPS', 'FRN', 'NonMarketable']
# The publicly marketable subset of SECURITY_TYPES. Derived, not restated, so adding a
# security type is one edit here rather than an edit plus a hunt for parallel copies.
PUBLIC_MARKETABLE_SECURITY_TYPES = frozenset(SECURITY_TYPES) - {'NonMarketable'}
MATURITY_CATEGORIES = ['bills', 'notes', 'bonds']
# Quantities that scale with the share of a lot moved when a secondary transfer splits it.
TRANSFER_SCALABLE_COLUMNS = (
    'FaceValue',
    'OriginalPrincipal',
    'AdjustedPrincipal',
    'AccruedInterest_FRN',
    'IssueProceeds',
    'CleanPrice',
    'AccruedInterest',
    'DirtyValue',
)
MARKETABLE_PREFERENCE_CATEGORIES = ('bills', 'notes', 'bonds', 'tips', 'frn')
PREFERENCE_CATEGORIES = [*MARKETABLE_PREFERENCE_CATEGORIES, 'nonmarketable']
PRIVATE_SUBBUCKET_DOMESTIC_NONBANK = 'domestic_nonbank_deposit_funded'
PRIVATE_SUBBUCKET_MMF = 'mmf_cash_fund_route'
PRIVATE_SUBBUCKET_MMF_ADDED = 'mmf_added_position'
# Key under which holder-preference blocks carry the private route split of each category.
PRIVATE_SUBBUCKET_SHARES_KEY = '__private_subbucket_shares__'
PRIVATE_AUCTION_SUBBUCKETS = (
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
)
PRIVATE_SUBBUCKETS = [*PRIVATE_AUCTION_SUBBUCKETS, PRIVATE_SUBBUCKET_MMF_ADDED]
# Every money-fund route, ordinary holdings and the added position; each may acquire only
# securities a money fund may hold (mmf_eligibility).
MMF_SUBBUCKETS = (PRIVATE_SUBBUCKET_MMF, PRIVATE_SUBBUCKET_MMF_ADDED)
MMF_ADDED_TDC_COLUMNS = {
    name: f"{name}_MMFAdded"
    for name in ("TDC_PrincipalToDU", "TDC_BillDiscountInterestToDU", "TDC_SecondaryTrades")
}
MMF_DEPOSIT_PASS_THROUGH_DEFAULT = 1.0
MMF_DEPOSIT_PASS_THROUGH_STATUS = 'ordinary_money_fund_cash_fully_settled_through_covered_deposits'
MMF_DEPOSIT_PASS_THROUGH_SENSITIVITY_GRID = [0.00, 0.50, 0.90, 0.95, 0.97, 1.00]

# --- Global Constants ---
TGA_FLOOR_TOLERANCE = 1.0e-9
DAYS_PER_YEAR_ACTUAL = 365.25
FRN_DAY_COUNT_BASIS = 360.0

# --- Data Types ---
PORTFOLIO_DTYPES = {
    'BondID': 'Int64',
    'IssueID': 'string',
    'SecurityType': 'string',
    'IssueDate': 'datetime64[ns]',
    'MaturityDate': 'datetime64[ns]',
    'DatedDate': 'datetime64[ns]',
    'OriginalDatedDate': 'datetime64[ns]',
    'FirstInterestPaymentDate': 'datetime64[ns]',
    'InterestPaymentFrequency': 'float64',
    'OriginalMaturityYears': 'float64',
    'FaceValue': 'float64',
    'CouponRate': 'float64',
    'HolderType': 'string',
    'HolderSubBucket': 'string',
    'TDCPrincipalHolderType': 'string',
    'TDCPrincipalHolderSubBucket': 'string',
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
