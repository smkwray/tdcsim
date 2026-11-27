"""Build source-backed input contracts for RateWall-oriented tdcsim runs.

The builder deliberately separates exact public source objects from constrained
allocations. MSPD supplies security-level marketable Treasury cohorts and NY Fed
SOMA supplies Federal Reserve holdings by CUSIP. No public source gives other
holders by CUSIP, so the non-Fed remainder of each security is split by WAMEST
sector bill and nonbill holdings: banks, foreign holders and money funds keep their
measured holdings converted to par, and domestic nonbanks take the rest. Money funds'
floating-rate notes are the exception: SEC Form N-MFP reports them by CUSIP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml

from bill_quote_basis import CBO_3M_BILL_DAYS, discount_rate_to_investment_rate
from cbo_yield_curve_surface import INTERPOLATION_METHOD
from mmf_eligibility import MMF_MAX_REMAINING_DAYS, mmf_eligible_mask
from sim_pricing import evaluate_nominal_yield, value_treasury_security
from tdc_shared import (
    DAYS_PER_YEAR_ACTUAL,
    FRN_MMF_AUCTION_SHARE_KEY,
    MARKETABLE_PREFERENCE_CATEGORIES,
    MMF_DEPOSIT_PASS_THROUGH_DEFAULT,
    MMF_DEPOSIT_PASS_THROUGH_SENSITIVITY_GRID,
    MMF_DEPOSIT_PASS_THROUGH_STATUS,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
)
from yield_curve_path import curve_for_date, load_yield_curve_surface


FISCALDATA_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
SOMA_ASOF_DATES_URL = "https://markets.newyorkfed.org/api/soma/asofdates/list.json"
SOMA_TREASURY_ASOF_URL = "https://markets.newyorkfed.org/api/soma/tsy/get/asof/{as_of}.csv"
SOMA_HOLDINGS_FILENAME = "nyfed_soma_treasury_holdings.csv"
YIELD_CURVE_SURFACE_FILENAME = "tdcsim_yield_curve_surface.csv"
SOMA_TYPE_CLASSES = {
    "Bills": ("Bills",),
    "NotesBonds": ("Notes", "Bonds"),
    "TIPS": ("Treasury Inflation-Protected Securities",),
    "FRNs": ("Floating Rate Notes",),
}
MATURED_CLASS_CATEGORIES = {"Bills": "bills", "Notes": "notes", "Bonds": "bonds"}
MSPD_TOTAL_TOLERANCE_MIL = 1.0
BOND_HISTORY_SOURCE = (
    "FiscalData MSPD security cohorts scaled to MSPD Table 1 class totals; "
    "CB from NY Fed SOMA by CUSIP plus SOMA auction add-ons settling by the record date; "
    "banks, foreign and money funds at WAMEST bill and nonbill holdings converted to par, "
    "domestic nonbanks take the rest of the non-Fed remainder; "
    "money funds exchange ineligible holdings with domestic nonbanks at equal par; "
    "money funds hold their SEC N-MFP floating-rate notes by CUSIP, exchanged for their bills at equal par"
)
WAMEST_CANONICAL_RELATIVE_PATH = "wamest/outputs/full_coverage_release/canonical_sector_maturity.csv"
# Opening-book holder routes for non-Fed holders: (HolderType, HolderSubBucket).
NONCB_OPENING_ROUTES = (
    ("Banks", ""),
    ("Foreign", ""),
    ("Private", PRIVATE_SUBBUCKET_DOMESTIC_NONBANK),
    ("Private", PRIVATE_SUBBUCKET_MMF),
)
# Domestic nonbanks are the Financial Accounts' residual sector, so in the opening book
# they take whatever non-Fed par the measured routes do not hold.
NONCB_RESIDUAL_ROUTE = ("Private", PRIVATE_SUBBUCKET_DOMESTIC_NONBANK)
MMF_OPENING_ROUTE = ("Private", PRIVATE_SUBBUCKET_MMF)
# SEC Form N-MFP money-fund Treasury holdings by report month-end and CUSIP, written by
# scripts/extract_sec_nmfp_mmf_holdings.py beside a provenance file naming its sources.
NMFP_HOLDINGS_FILENAME = "sec_nmfp_mmf_treasury_holdings.csv"
NMFP_HOLDINGS_COLUMNS = ("report_date", "cusip", "security_class", "maturity_date", "value_bil", "class_source", "series_count")
NMFP_SECURITY_CLASSES = frozenset({"bill", "frn", "fixed_coupon", "tips", "unresolved"})
# Money funds hold their filed floating-rate notes by CUSIP. N-MFP reports value; an FRN
# resets weekly and prices within a few basis points of par, so its value is read as par.
# Domestic nonbanks give the notes first, foreign holders and banks only the shortfall,
# each pro rata to its holding of the CUSIP; every giver takes money-fund bills at equal par.
FRN_OVERRIDE_GIVER_TIERS = ((NONCB_RESIDUAL_ROUTE,), (("Foreign", ""), ("Banks", "")))
FRN_OVERRIDE_CLAIM = (
    "observed-holdings override of the WAMEST maturity split for FRNs only, by owner decision: "
    "SEC N-MFP filings of money-fund FRN holdings outrank the model's estimated split for this instrument"
)
# The opening money-fund book must match the filings' total, bills, FRNs and coupon
# securities within 397 days, each within this relative gap.
NMFP_MIX_TOLERANCE = 0.05
NMFP_MIX_BUCKETS = ("total", "bills", "frn", "coupon_le397")
# FRN auction tranches, by settlement date, that calibrate the money-fund share of new FRN
# supply: four two-year FRNs, each a new issue and two reopenings, measured on N-MFP
# month-ends through April 2026.
FRN_AUCTION_SHARE_WINDOW = ("2025-04-01", "2026-03-31")
OPENING_ALLOCATION_RULE = "measured_holdings_at_par_domestic_nonbank_residual"
# WAMEST records each sector's Financial Accounts level basis. A market value (an LM series,
# as for foreign holders) converts to par with its group's factor. Any other level (the FL
# series of depositories and money funds: book value or amortized cost) is a book value:
# bills at book still carry their discount and convert with the bill factor, while coupon
# securities at book are read as par. That is a declared approximation: book value mixes
# amortized cost, near par, with fair value for banks' securities available for sale.
WAMEST_LEVEL_BASES = {
    "z1_market_value_level": "market_value",
    "z1_level_basis_not_confirmed_market_value": "book_value",
    "computed_or_unknown": "book_value",
}
# The book is at par. Each group's par-to-market
# factor prices its non-Fed securities on the governed baseline curve, evaluated as the
# CBO runtime evaluates it.
OPENING_PRICE_CURVE = {
    "scenario_id": "cbo_shape_preserving_baseline",
    "interpolation_method": INTERPOLATION_METHOD,
    "floor_zero": False,
}
# Every WAMEST canonical sector maps to one opening route. The Fed maps to None
# because CB holdings come from SOMA by CUSIP. Depository institutions, including
# credit unions, settle through reserve or correspondent balances and route to Banks.
WAMEST_SECTOR_ROUTES: dict[str, tuple[str, str] | None] = {
    "fed": None,
    "bank_us_chartered": ("Banks", ""),
    "bank_foreign_banking_offices_us": ("Banks", ""),
    "bank_us_affiliated_areas": ("Banks", ""),
    "credit_unions_marketable_proxy": ("Banks", ""),
    "foreigners_total": ("Foreign", ""),
    "money_market_funds": ("Private", PRIVATE_SUBBUCKET_MMF),
    **{
        sector: ("Private", PRIVATE_SUBBUCKET_DOMESTIC_NONBANK)
        for sector in (
            "asset_backed_securities_issuers",
            "closed_end_funds",
            "exchange_traded_funds",
            "federal_defined_benefit_pensions",
            "federal_defined_contribution_pensions",
            "government_sponsored_enterprises",
            "holding_companies",
            "households_nonprofits",
            "life_insurers",
            "mutual_funds",
            "nonfinancial_corporates",
            "nonfinancial_noncorporate_business",
            "other_financial_business",
            "private_defined_benefit_pensions",
            "private_defined_contribution_pensions",
            "property_casualty_insurers",
            "security_brokers_and_dealers",
            "state_local_employee_defined_benefit_pensions",
            "state_local_governments",
        )
    },
}
TREASURY_CURVE_XML = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve"
RATEWALL_CLAIM_BOUNDARY = "tdcsim_ratewall_source_backed_input_contract_not_evidence_mode"
TENORS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
TDCSIM_HOLDERS = ["Banks", "CB", "Foreign", "Private"]
PREFERENCE_CATEGORIES = list(MARKETABLE_PREFERENCE_CATEGORIES)
PRIVATE_SUBBUCKET_SOURCE_BUCKETS = {
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: "domestic_nonbank",
    PRIVATE_SUBBUCKET_MMF: "mmf",
}
Z1_LOOKBACK_QUARTERS = 8
Z1_SHRINKAGE_K_QUARTERS = 8.0
Z1_POSITIVE_ABSORPTION_FLOOR_BN_SAAR = 25.0
Z1_FLOOR_MEDIAN_MULTIPLIER = 0.10
PRIMARY_MARKET_LOOKBACK_QUARTERS = 8
PRIMARY_MARKET_OVERLAY_WEIGHT = 0.20
SCENARIO_SHIFT_LAMBDA_GRID = [0.25, 0.50, 0.75, 1.00]
SCENARIO_SHIFT_LAMBDA_DEFAULT = 0.50
RATE_SCENARIOS = {
    "cbo_shape_preserving_baseline": {
        "short_shift": 0.0,
        "long_shift": 0.0,
        "central_default_eligible": "true",
        "sensitivity_only": "false",
        "source_status": "source_backed_dynamic_curve_surface_not_flat_rate",
    },
    "higher_for_longer_sensitivity": {
        "short_shift": 0.0075,
        "long_shift": 0.0075,
        "central_default_eligible": "false",
        "sensitivity_only": "true",
        "source_status": "transparent_parallel_stress_around_source_backed_cbo_curve",
    },
    "rapid_easing_sensitivity": {
        "short_shift": -0.0075,
        "long_shift": -0.0075,
        "central_default_eligible": "false",
        "sensitivity_only": "true",
        "source_status": "transparent_parallel_stress_around_source_backed_cbo_curve",
    },
    "bear_steepener_sensitivity": {
        "short_shift": 0.0025,
        "long_shift": 0.0100,
        "central_default_eligible": "false",
        "sensitivity_only": "true",
        "source_status": "transparent_curve_shape_stress_around_source_backed_cbo_curve",
    },
    "bull_flattener_sensitivity": {
        "short_shift": -0.0100,
        "long_shift": -0.0025,
        "central_default_eligible": "false",
        "sensitivity_only": "true",
        "source_status": "transparent_curve_shape_stress_around_source_backed_cbo_curve",
    },
}


@dataclass(frozen=True)
class RateWallScenario:
    scenario_id: str
    curve_id: str
    holder_target: str


RATEWALL_SCENARIO_REGISTRY = (
    RateWallScenario("current_mix_baseline", "cbo_shape_preserving_baseline", "current_mix_baseline"),
    RateWallScenario(
        "domestic_nonbank_absorption_shift",
        "cbo_shape_preserving_baseline",
        "domestic_nonbank_absorption_shift",
    ),
    RateWallScenario(
        "reserve_user_absorption_shift",
        "cbo_shape_preserving_baseline",
        "reserve_user_absorption_shift",
    ),
    RateWallScenario(
        "current_mix_higher_for_longer",
        "higher_for_longer_sensitivity",
        "current_mix_baseline",
    ),
    RateWallScenario(
        "current_mix_rapid_easing",
        "rapid_easing_sensitivity",
        "current_mix_baseline",
    ),
    RateWallScenario(
        "domestic_nonbank_absorption_shift_higher_for_longer",
        "higher_for_longer_sensitivity",
        "domestic_nonbank_absorption_shift",
    ),
    RateWallScenario(
        "domestic_nonbank_absorption_shift_rapid_easing",
        "rapid_easing_sensitivity",
        "domestic_nonbank_absorption_shift",
    ),
    RateWallScenario(
        "reserve_user_absorption_shift_higher_for_longer",
        "higher_for_longer_sensitivity",
        "reserve_user_absorption_shift",
    ),
    RateWallScenario(
        "reserve_user_absorption_shift_rapid_easing",
        "rapid_easing_sensitivity",
        "reserve_user_absorption_shift",
    ),
    RateWallScenario(
        "current_mix_bear_steepener",
        "bear_steepener_sensitivity",
        "current_mix_baseline",
    ),
    RateWallScenario(
        "current_mix_bull_flattener",
        "bull_flattener_sensitivity",
        "current_mix_baseline",
    ),
)


def _ratewall_scenarios() -> tuple[RateWallScenario, ...]:
    scenario_ids = [row.scenario_id for row in RATEWALL_SCENARIO_REGISTRY]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("RateWall scenario registry contains duplicate scenario IDs.")
    unknown_curves = sorted(
        {row.curve_id for row in RATEWALL_SCENARIO_REGISTRY} - set(RATE_SCENARIOS)
    )
    if unknown_curves:
        raise ValueError(f"RateWall scenario registry references unknown curves: {unknown_curves}")
    return RATEWALL_SCENARIO_REGISTRY


SCENARIO_CURVE_MAP = {
    row.scenario_id: row.curve_id
    for row in _ratewall_scenarios()
}


def _request_json(url: str) -> dict:
    with urlopen(Request(url, headers={"User-Agent": "tdcsim-ratewall-source-builder"}), timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _fiscaldata_url(path: str, **params: str | int) -> str:
    return f"{FISCALDATA_BASE}/{path}?{urlencode(params)}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_float(value, default: float = 0.0) -> float:
    if value in {None, "", "null"}:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return default


def _quarter_label(ts) -> str:
    stamp = pd.to_datetime(ts)
    return f"{stamp.year}Q{stamp.quarter}"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _latest_record_date(endpoint: str) -> str:
    payload = _request_json(
        _fiscaldata_url(
            endpoint,
            sort="-record_date",
            **{"page[size]": 1, "format": "json"},
        )
    )
    return payload["data"][0]["record_date"]


def _fetch_fiscaldata_rows(endpoint: str, *, record_date: str, page_size: int = 10000) -> list[dict]:
    payload = _request_json(
        _fiscaldata_url(
            endpoint,
            filter=f"record_date:eq:{record_date}",
            **{"page[size]": page_size, "format": "json"},
        )
    )
    return list(payload.get("data", []))


def _security_type(class_desc: str) -> str | None:
    label = class_desc.lower()
    if "bill" in label:
        return "Fixed"
    if "floating" in label or "frn" in label:
        return "FRN"
    if "inflation-protected" in label or "tips" in label:
        return "TIPS"
    if "note" in label or "bond" in label:
        return "Fixed"
    return None


def _maturity_category(class_desc: str, original_maturity: float) -> str | None:
    label = class_desc.lower()
    if "bill" in label or original_maturity <= 1.0 + 1e-9:
        return "bills"
    if "note" in label or original_maturity <= 10.0 + 1e-9:
        return "notes"
    if "bond" in label:
        return "bonds"
    return None


def _class_key(class_desc: str) -> str | None:
    label = class_desc.lower()
    if "bill" in label:
        return "Bills"
    if "floating" in label or "frn" in label:
        return "Floating Rate Notes"
    if "inflation-protected" in label or "tips" in label:
        return "Treasury Inflation-Protected Securities"
    if "note" in label:
        return "Notes"
    if "bond" in label:
        return "Bonds"
    return None


def fetch_soma_treasury_holdings(record_date: str, output_path: Path) -> str:
    """Save NY Fed SOMA Treasury holdings for the last weekly as-of date on or before ``record_date``."""

    as_of_dates = _request_json(SOMA_ASOF_DATES_URL)["soma"]["asOfDates"]
    eligible = sorted(str(value) for value in as_of_dates if str(value) <= record_date)
    if not eligible:
        raise ValueError(f"no SOMA as-of date on or before {record_date}")
    url = SOMA_TREASURY_ASOF_URL.format(as_of=eligible[-1])
    with urlopen(Request(url, headers={"User-Agent": "tdcsim-ratewall-source-builder"}), timeout=60) as response:
        output_path.write_bytes(response.read())
    return url


def load_soma_cusip_holdings(path: Path) -> tuple[str, pd.DataFrame]:
    """Return the SOMA as-of date and CUSIP amounts in USD millions on the MSPD outstanding basis.

    MSPD Table 3 reports TIPS outstanding including the inflation adjustment, so
    TIPS amounts add SOMA inflation compensation to par.
    """

    frame = pd.read_csv(path, dtype=str)
    required = ["As Of Date", "CUSIP", "Security Type", "Maturity Date", "Par Value", "Inflation Compensation"]
    missing = [column for column in required if column not in frame.columns]
    if missing or frame.empty:
        raise ValueError(f"SOMA holdings file {path.name} is empty or lacks columns: {missing}")
    as_of_dates = sorted(set(frame["As Of Date"].astype(str).str.strip()))
    if len(as_of_dates) != 1:
        raise ValueError(f"SOMA holdings file {path.name} must hold one as-of date, found {as_of_dates}")
    cusips = frame["CUSIP"].astype(str).str.strip().str.strip("'")
    par = pd.to_numeric(frame["Par Value"], errors="coerce")
    is_tips = frame["Security Type"].astype(str).str.strip().eq("TIPS")
    inflation = pd.to_numeric(frame["Inflation Compensation"], errors="coerce")
    maturity = pd.to_datetime(frame["Maturity Date"], errors="coerce")
    bad = cusips.eq("") | par.isna() | (par < 0) | maturity.isna() | (is_tips & inflation.isna())
    if bad.any():
        raise ValueError(f"SOMA holdings file {path.name} has {int(bad.sum())} unparseable rows")
    if cusips.duplicated().any():
        raise ValueError(f"SOMA holdings file {path.name} repeats CUSIPs: {sorted(set(cusips[cusips.duplicated()]))}")
    amount_mil = (par + inflation.where(is_tips, 0.0)) / 1.0e6
    return as_of_dates[0], pd.DataFrame(
        {
            "cusip": cusips,
            "security_type": frame["Security Type"].astype(str).str.strip(),
            "amount_mil": amount_mil,
            "par_mil": par / 1.0e6,
            "maturity_date": maturity.dt.date.astype(str),
        }
    )


def load_nmfp_mmf_holdings(path: Path) -> tuple[pd.DataFrame, dict]:
    """Return the N-MFP money-fund Treasury holdings and the provenance its producer wrote.

    The provenance file must describe these exact bytes, so the book never binds to an
    extract whose sources are unrecorded, and must declare the information date that
    bounds every filing it selected.
    """

    provenance_path = path.with_suffix(".provenance.json")
    if not path.exists() or not provenance_path.exists():
        raise ValueError(f"N-MFP money-fund holdings {path.name} or its provenance file is missing")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("output", {}).get("sha256") != _sha256(path) or not provenance.get("zips"):
        raise ValueError(f"{provenance_path.name} does not describe {path.name} and its source zips")
    filed_through = provenance.get("filed_through")
    selected = provenance.get("selected_filings")
    if (
        not isinstance(filed_through, str)
        or not isinstance(selected, dict)
        or not selected
        or any(not rows or any(str(row[2]) > filed_through for row in rows) for rows in selected.values())
    ):
        raise ValueError(f"{provenance_path.name} does not list selected filings all made by a declared information date")
    frame = pd.read_csv(path, dtype={"report_date": str, "cusip": str, "security_class": str, "maturity_date": str})
    missing = [column for column in NMFP_HOLDINGS_COLUMNS if column not in frame.columns]
    if missing or frame.empty:
        raise ValueError(f"N-MFP money-fund holdings {path.name} is empty or lacks columns: {missing}")
    value = pd.to_numeric(frame["value_bil"], errors="coerce")
    unknown = sorted(set(frame["security_class"]) - NMFP_SECURITY_CLASSES)
    coupon = frame["security_class"].isin({"fixed_coupon", "tips"})
    bad = value.isna() | (value < 0) | frame["report_date"].isna() | frame["cusip"].isna()
    bad |= coupon & pd.to_datetime(frame["maturity_date"], errors="coerce").isna()
    if unknown or bad.any() or frame.duplicated(["report_date", "cusip"]).any():
        raise ValueError(
            f"N-MFP money-fund holdings {path.name} has unknown classes {unknown}, {int(bad.sum())} unusable rows "
            "or a CUSIP twice in one month"
        )
    return frame.assign(value_bil=value), provenance


def _nmfp_bucket(security_class: str, maturity_date, record_date: str) -> str:
    if security_class == "bill":
        return "bills"
    if security_class in {"frn", "unresolved"}:
        return security_class
    days = (pd.Timestamp(maturity_date) - pd.Timestamp(record_date)).days
    return "coupon_le397" if days <= MMF_MAX_REMAINING_DAYS else "coupon_gt397"


def calibrate_mmf_frn_auction_share(holdings: pd.DataFrame, auctions_path: Path) -> dict:
    """Money funds' share of new FRN supply, from N-MFP holdings and auction results.

    Treasury sells each two-year FRN as a new issue and two monthly reopenings, and N-MFP
    reports month-end holdings, so a month after any tranche but the last the next one has
    already settled. Each FRN whose tranches all settle in ``FRN_AUCTION_SHARE_WINDOW`` is
    therefore one observation: money-fund holdings at the month-end one month after its
    last tranche, over the public amount accepted (total less SOMA) across its tranches.
    The share is the amount-weighted mean of the observations.
    """

    auctions = pd.read_csv(auctions_path, dtype=str)
    missing = [column for column in ("cusip", "issue_date", "floating_rate", "total_accepted", "soma_accepted") if column not in auctions]
    if missing:
        raise ValueError(f"{auctions_path.name} lacks FRN auction columns {missing}")
    frns = auctions.loc[auctions["floating_rate"].eq("Yes")].assign(issue=pd.to_datetime(auctions["issue_date"], errors="coerce"))
    start, end = (pd.Timestamp(day) for day in FRN_AUCTION_SHARE_WINDOW)
    in_window = sorted(set(frns.loc[frns["issue"].between(start, end), "cusip"]))
    if not in_window:
        raise ValueError(f"{auctions_path.name} has no FRN auction settling in {FRN_AUCTION_SHARE_WINDOW}")
    filed = holdings.loc[holdings["security_class"].eq("frn")].set_index(["report_date", "cusip"])["value_bil"]
    report_dates = set(holdings["report_date"])
    observations = []
    for cusip in in_window:
        tranches = frns.loc[frns["cusip"].eq(cusip)].sort_values("issue")
        if not tranches["issue"].between(start, end).all():
            raise ValueError(f"FRN {cusip} has tranches outside the calibration window {FRN_AUCTION_SHARE_WINDOW}")
        accepted = pd.to_numeric(tranches["total_accepted"], errors="coerce")
        soma = pd.to_numeric(tranches["soma_accepted"], errors="coerce")
        if accepted.isna().any() or soma.isna().any():
            raise ValueError(f"{auctions_path.name} lacks accepted amounts for a tranche of FRN {cusip}")
        public = float((accepted - soma).sum()) / 1e9
        measured = (tranches["issue"].iloc[-1] + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)).date().isoformat()
        if measured not in report_dates:
            raise ValueError(f"N-MFP holdings have no report at {measured}, a month after FRN {cusip}'s last tranche")
        held = float(filed.get((measured, cusip), 0.0))
        observations.append(
            {
                "cusip": cusip,
                "tranche_issue_dates": [day.date().isoformat() for day in tranches["issue"]],
                "public_accepted_bil": public,
                "measured_at": measured,
                "mmf_holdings_bil": held,
                "share": held / public,
            }
        )
    shares = [row["share"] for row in observations]
    return {
        "share": sum(row["mmf_holdings_bil"] for row in observations) / sum(row["public_accepted_bil"] for row in observations),
        "range": [min(shares), max(shares)],
        "window": list(FRN_AUCTION_SHARE_WINDOW),
        "auctions": sum(len(row["tranche_issue_dates"]) for row in observations),
        "rule": (
            "per FRN whose tranches all settle in the window: N-MFP money-fund holdings one month-end after its "
            "last tranche over the public amount accepted across its tranches; amount-weighted over FRNs"
        ),
        "observations": observations,
        "auctions_source": {"file": auctions_path.name, "sha256": _sha256(auctions_path)},
    }


def _apply_mmf_frn_holdings(candidate_rows: list[dict], targets: dict[str, float]) -> dict:
    """Give money funds their filed FRN holdings by CUSIP for an equal par of their bills.

    Sector totals do not change. Within each FRN the notes come from domestic nonbanks
    first, then from foreign holders and banks for the shortfall, each pro rata to its
    holdings of the CUSIP; the givers take money-fund bills pro rata across the bill lots.
    """

    tolerance = 1e-9
    frn_items: dict[str, list[dict]] = {}
    for item in candidate_rows:
        if item["security_type"] == "FRN":
            frn_items.setdefault(item["cusip"], []).append(item)
    unknown = sorted(set(targets) - set(frn_items))
    if unknown:
        raise ValueError(f"N-MFP money-fund FRN holdings name CUSIPs that are not FRNs in the opening book: {unknown}")

    def held(items: list[dict], route: tuple[str, str]) -> float:
        return sum(item["route_faces"][route] for item in items)

    moved = {route: 0.0 for tier in FRN_OVERRIDE_GIVER_TIERS for route in tier}
    by_cusip = {}
    for cusip, items in sorted(frn_items.items()):
        before = held(items, MMF_OPENING_ROUTE)
        target = targets.get(cusip, 0.0)
        noncb = sum(held(items, route) for route in NONCB_OPENING_ROUTES)
        if target > noncb + tolerance:
            raise ValueError(f"money funds filed {target:.6f} bn of {cusip}, more than its non-Fed outstanding {noncb:.6f} bn")
        if target < before - tolerance:
            raise ValueError(
                f"money funds filed {target:.6f} bn of {cusip}, less than the {before:.6f} bn the opening split gives them; "
                "the override only moves FRNs to money funds"
            )
        remaining = max(0.0, target - before)
        takes = {}
        for tier in FRN_OVERRIDE_GIVER_TIERS:
            pool = sum(held(items, route) for route in tier)
            take = min(remaining, pool)
            for route in tier:
                takes[route] = take * held(items, route) / pool if pool > 0.0 else 0.0
            remaining -= take
        for route, take in takes.items():
            route_total = held(items, route)
            for item in items:
                amount = take * item["route_faces"][route] / route_total if route_total > 0.0 else 0.0
                item["route_faces"][route] = max(0.0, item["route_faces"][route] - amount)
                item["route_faces"][MMF_OPENING_ROUTE] += amount
            moved[route] += take
        by_cusip[cusip] = {
            "filed_bil": target,
            "pre_override_bil": before,
            "from_bil": {_route_key(route): take for route, take in takes.items()},
        }

    bill_items = [item for item in candidate_rows if item["class_key"] == "Bills"]
    mmf_bills = held(bill_items, MMF_OPENING_ROUTE)
    total_moved = sum(moved.values())
    if total_moved > mmf_bills + tolerance:
        raise ValueError(f"money funds need {total_moved:.6f} bn of bills to give for their filed FRNs but hold {mmf_bills:.6f} bn")
    if total_moved > 0.0:
        for item in bill_items:
            weight = item["route_faces"][MMF_OPENING_ROUTE] / mmf_bills
            item["route_faces"][MMF_OPENING_ROUTE] = max(0.0, item["route_faces"][MMF_OPENING_ROUTE] - total_moved * weight)
            for route, amount in moved.items():
                item["route_faces"][route] += amount * weight
    return {
        "claim": FRN_OVERRIDE_CLAIM,
        "rule": (
            "money funds hold their N-MFP FRN value, read as par, by CUSIP; domestic nonbanks give first, "
            "foreign holders and banks the shortfall, pro rata to their holdings of the CUSIP; "
            "each giver takes money-fund bills at equal par, pro rata across the bill lots"
        ),
        "frn_to_mmf_bil_by_sector": {_route_key(route): amount for route, amount in moved.items()},
        "bills_from_mmf_bil_by_sector": {_route_key(route): amount for route, amount in moved.items()},
        "by_cusip": by_cusip,
    }


def _check_mmf_mix(candidate_rows: list[dict], filed: pd.DataFrame, *, record_date: str) -> dict:
    """Compare the opening money-fund book with the N-MFP filings; raise outside the tolerance."""

    model = {"bills": 0.0, "frn": 0.0, "coupon_le397": 0.0, "coupon_gt397": 0.0}
    for item in candidate_rows:
        security_class = "frn" if item["security_type"] == "FRN" else "bill" if item["class_key"] == "Bills" else "coupon"
        model[_nmfp_bucket(security_class, item["maturity_date"], record_date)] += item["route_faces"][MMF_OPENING_ROUTE]
    filings = {bucket: 0.0 for bucket in (*model, "unresolved")}
    for row in filed.itertuples(index=False):
        filings[_nmfp_bucket(row.security_class, row.maturity_date, record_date)] += row.value_bil
    model["total"] = sum(model.values())
    filings["total"] = sum(filings.values())
    buckets = {}
    failed = []
    for bucket in NMFP_MIX_BUCKETS:
        if filings[bucket] > 0.0:
            gap = model[bucket] / filings[bucket] - 1.0
        else:
            gap = 0.0 if model[bucket] == 0.0 else math.inf
        buckets[bucket] = {"model_par_bil": model[bucket], "filed_value_bil": filings[bucket], "relative_gap": gap}
        if not abs(gap) <= NMFP_MIX_TOLERANCE:
            failed.append(f"{bucket} {model[bucket]:.3f} vs {filings[bucket]:.3f} bn")
    if failed:
        raise ValueError(
            f"opening money-fund book is more than {NMFP_MIX_TOLERANCE:.0%} from the N-MFP filings at {record_date}: "
            + "; ".join(failed)
        )
    return {
        "report_date": record_date,
        "tolerance": NMFP_MIX_TOLERANCE,
        "basis": "model book par against filed value; FRN value is read as par",
        "buckets": buckets,
        "filed_unresolved_value_bil": filings["unresolved"],
        "status": "pass",
    }


def load_soma_auction_add_ons(path: Path, *, after: str, through: str) -> tuple[int, dict[str, float]]:
    """Return the auction count and SOMA add-ons by CUSIP (USD millions) settling in (after, through].

    The add-ons are on the MSPD outstanding basis, so a TIPS add-on carries its index
    ratio on the issue date. The file must reach past ``through`` so a stale file
    cannot drop add-ons silently.
    """

    frame = pd.read_csv(path, dtype=str)
    required = ["issue_date", "cusip", "soma_accepted", "inflation_index_security", "index_ratio_on_issue_date"]
    missing = [column for column in required if column not in frame.columns]
    if missing or frame.empty:
        raise ValueError(f"auction file {path.name} is empty or lacks columns: {missing}")
    issue = pd.to_datetime(frame["issue_date"], errors="coerce")
    if issue.isna().any():
        raise ValueError(f"auction file {path.name} has {int(issue.isna().sum())} rows without a valid issue date")
    if issue.max() <= pd.Timestamp(through):
        raise ValueError(f"auction file {path.name} ends on {issue.max().date()}, so it may miss settlements through {through}")
    if issue.min() > pd.Timestamp(after):
        raise ValueError(f"auction file {path.name} starts on {issue.min().date()}, so it may miss settlements after {after}")
    rows = frame[(issue > pd.Timestamp(after)) & (issue <= pd.Timestamp(through))]
    if rows.duplicated(["issue_date", "cusip"]).any():
        raise ValueError(f"auction file {path.name} lists one CUSIP twice on one settlement date in ({after}, {through}]")
    cusips = rows["cusip"].fillna("").astype(str).str.strip()
    amount = pd.to_numeric(rows["soma_accepted"], errors="coerce")
    tips_flag = rows["inflation_index_security"].fillna("").astype(str).str.strip()
    ratio = pd.to_numeric(rows["index_ratio_on_issue_date"], errors="coerce").where(tips_flag.eq("Yes"), 1.0)
    bad = (
        ~np.isfinite(amount) | (amount < 0) | ~np.isfinite(ratio) | (ratio <= 0)
        | cusips.eq("") | ~tips_flag.isin(["Yes", "No"])
    )
    if bad.any():
        raise ValueError(f"auction file {path.name} has {int(bad.sum())} unusable SOMA rows settling in ({after}, {through}]")
    add_ons = (amount * ratio / 1.0e6).groupby(cusips).sum()
    return len(rows), {cusip: float(value) for cusip, value in sorted(add_ons.items())}


def load_wamest_noncb_weights(path: Path, *, record_date: str) -> dict:
    """Return WAMEST non-Fed holdings and shares by opening route for bills and nonbills.

    Uses the latest WAMEST quarter on or before ``record_date``. Only ``level``,
    ``bill_share`` and the level's valuation basis are read; the Fed row is excluded
    because CB comes from SOMA. The shares are the measured shares; the opening book
    converts the holdings to par itself, by basis (``WAMEST_LEVEL_BASES``).
    """

    frame = pd.read_csv(
        path,
        usecols=["date", "sector_key", "level", "bill_share", "publication_status", "revaluation_valuation_basis"],
        dtype=str,
    )
    dates = sorted(value for value in frame["date"].dropna().unique() if value <= record_date)
    if not dates:
        raise ValueError(f"WAMEST panel {path.name} has no quarter on or before {record_date}")
    rows = frame[frame["date"] == dates[-1]]
    sectors = rows["sector_key"].astype(str)
    unknown = sorted(set(sectors) - set(WAMEST_SECTOR_ROUTES))
    absent = sorted(set(WAMEST_SECTOR_ROUTES) - set(sectors))
    if unknown or absent or sectors.duplicated().any():
        raise ValueError(f"WAMEST sectors at {dates[-1]} do not match the route registry: unknown={unknown} absent={absent}")
    holdings = {route: {"bills": 0.0, "nonbills": 0.0} for route in NONCB_OPENING_ROUTES}
    by_basis = {
        route: {group: {basis: 0.0 for basis in sorted(set(WAMEST_LEVEL_BASES.values()))} for group in ("bills", "nonbills")}
        for route in NONCB_OPENING_ROUTES
    }
    for row in rows.itertuples(index=False):
        route = WAMEST_SECTOR_ROUTES[row.sector_key]
        if route is None:
            continue
        level = pd.to_numeric(row.level, errors="coerce")
        bill_share = pd.to_numeric(row.bill_share, errors="coerce")
        if row.publication_status != "published_estimate" or pd.isna(level) or level < 0 or not 0.0 <= bill_share <= 1.0:
            raise ValueError(f"WAMEST row {row.sector_key} at {dates[-1]} is not a usable published estimate")
        basis = WAMEST_LEVEL_BASES.get(row.revaluation_valuation_basis)
        if basis is None:
            raise ValueError(
                f"WAMEST row {row.sector_key} at {dates[-1]} has unknown level basis {row.revaluation_valuation_basis!r}"
            )
        for group, amount in (("bills", float(level) * float(bill_share)), ("nonbills", float(level) * (1.0 - float(bill_share)))):
            holdings[route][group] += amount
            by_basis[route][group][basis] += amount
    weights = {}
    for group in ("bills", "nonbills"):
        total = sum(holdings[route][group] for route in NONCB_OPENING_ROUTES)
        if total <= 0:
            raise ValueError(f"WAMEST non-Fed {group} holdings are not positive at {dates[-1]}")
        weights[group] = {route: holdings[route][group] / total for route in NONCB_OPENING_ROUTES}
    return {"wamest_date": dates[-1], "weights": weights, "holdings_usd_mil": holdings, "holdings_by_basis_usd_mil": by_basis}


def _normalize_holder_shares(raw: dict[str, float]) -> dict[str, float]:
    clipped = {holder: max(0.0, float(raw.get(holder, 0.0))) for holder in TDCSIM_HOLDERS}
    total = sum(clipped.values())
    if total <= 0:
        return {holder: 1.0 / len(TDCSIM_HOLDERS) for holder in TDCSIM_HOLDERS}
    return {holder: value / total for holder, value in clipped.items()}


def _normalize_private_subbucket_shares(raw: dict[str, float]) -> dict[str, float]:
    clipped = {
        PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: max(
            0.0,
            float(raw.get(PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, 0.0)),
        ),
        PRIVATE_SUBBUCKET_MMF: max(0.0, float(raw.get(PRIVATE_SUBBUCKET_MMF, 0.0))),
    }
    total = sum(clipped.values())
    if total <= 0.0:
        return {PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: 1.0, PRIVATE_SUBBUCKET_MMF: 0.0}
    return {key: value / total for key, value in clipped.items()}


def _blend_private_subbucket_shares(
    left: dict[str, float],
    right: dict[str, float] | None,
    *,
    right_weight: float,
) -> dict[str, float]:
    if right is None:
        return _normalize_private_subbucket_shares(left)
    return _normalize_private_subbucket_shares(
        {
            route: (1.0 - right_weight) * left.get(route, 0.0)
            + right_weight * right.get(route, 0.0)
            for route in PRIVATE_SUBBUCKET_SOURCE_BUCKETS
        }
    )


def _blend_shares(
    left: dict[str, float],
    right: dict[str, float],
    *,
    right_weight: float,
) -> dict[str, float]:
    return _normalize_holder_shares(
        {
            holder: (1.0 - right_weight) * left.get(holder, 0.0)
            + right_weight * right.get(holder, 0.0)
            for holder in TDCSIM_HOLDERS
        }
    )


def _default_holder_absorption_calibration() -> dict[str, float | list[float]]:
    return {
        "scenario_shift_lambda_grid": list(SCENARIO_SHIFT_LAMBDA_GRID),
        "scenario_shift_lambda_default": SCENARIO_SHIFT_LAMBDA_DEFAULT,
    }


def _load_holder_absorption_calibration(config_path: Path) -> dict[str, float | list[float]]:
    defaults = _default_holder_absorption_calibration()
    if not config_path.exists():
        return defaults
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = loaded.get("holder_absorption_calibration", {})
    if not isinstance(section, dict):
        return defaults
    grid_raw = section.get("scenario_shift_lambda_grid", defaults["scenario_shift_lambda_grid"])
    if not isinstance(grid_raw, list) or not grid_raw:
        grid = list(SCENARIO_SHIFT_LAMBDA_GRID)
    else:
        grid = [float(value) for value in grid_raw]
    return {
        "scenario_shift_lambda_grid": grid,
        "scenario_shift_lambda_default": _safe_float(
            section.get("scenario_shift_lambda_default"),
            float(defaults["scenario_shift_lambda_default"]),
        ),
    }


def _load_tdcmix_targets(tdcmix_prior_path: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(tdcmix_prior_path)
    rows = {str(row["bucket"]): row for _, row in frame.iterrows()}

    def collapse(value_field: str) -> dict[str, float]:
        raw = {
            "CB": _safe_float(rows.get("fed", {}).get(value_field)),
            "Banks": _safe_float(rows.get("banks", {}).get(value_field)),
            "Foreign": _safe_float(rows.get("row", {}).get(value_field)),
            "Private": _safe_float(rows.get("domestic_nonbank", {}).get(value_field))
            + _safe_float(rows.get("mmf", {}).get(value_field)),
        }
        out = _normalize_holder_shares(raw)
        if sum(out.values()) <= 0:
            raise ValueError(f"No positive absorption priors in {tdcmix_prior_path}")
        return out

    baseline = collapse("default_prior_value")
    du_heavy = collapse("upper_prior_value")
    ru_raw = {
        "CB": _safe_float(rows.get("fed", {}).get("upper_prior_value")),
        "Banks": _safe_float(rows.get("banks", {}).get("upper_prior_value")),
        "Foreign": _safe_float(rows.get("row", {}).get("upper_prior_value")),
        "Private": _safe_float(rows.get("domestic_nonbank", {}).get("lower_prior_value"))
        + _safe_float(rows.get("mmf", {}).get("lower_prior_value")),
    }
    ru_heavy = _normalize_holder_shares(ru_raw)
    return {
        "current_mix_baseline": baseline,
        "domestic_nonbank_absorption_shift": du_heavy,
        "reserve_user_absorption_shift": ru_heavy,
    }


def _load_tdcmix_private_subbucket_targets(tdcmix_prior_path: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(tdcmix_prior_path)
    rows = {str(row["bucket"]): row for _, row in frame.iterrows()}

    def route_split(dn_field: str, mmf_field: str) -> dict[str, float]:
        return _normalize_private_subbucket_shares(
            {
                PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: _safe_float(
                    rows.get("domestic_nonbank", {}).get(dn_field)
                ),
                PRIVATE_SUBBUCKET_MMF: _safe_float(rows.get("mmf", {}).get(mmf_field)),
            }
        )

    return {
        "current_mix_baseline": route_split("default_prior_value", "default_prior_value"),
        "domestic_nonbank_absorption_shift": route_split("upper_prior_value", "upper_prior_value"),
        "reserve_user_absorption_shift": route_split("lower_prior_value", "lower_prior_value"),
    }


def _latest_z1_absorption_shares(z1_absorption_path: Path) -> tuple[dict[str, float] | None, float]:
    if not z1_absorption_path.exists():
        return None, 0.0
    frame = pd.read_csv(z1_absorption_path)
    if "quarter" not in frame.columns:
        return None, 0.0
    required = [
        "positive_absorption_total",
        "pos_abs_share_fed",
        "pos_abs_share_banks",
        "pos_abs_share_dealer_bridge",
        "pos_abs_share_row",
        "pos_abs_share_mmf",
        "pos_abs_share_domestic_nonbank",
    ]
    if not set(required) <= set(frame.columns):
        return None, 0.0
    frame["quarter_date"] = pd.to_datetime(frame["quarter"], errors="coerce")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["quarter_date"])
        .dropna(subset=required)
        .sort_values("quarter_date")
        .tail(Z1_LOOKBACK_QUARTERS)
    )
    if frame.empty:
        return None, 0.0
    core_share = (
        frame["pos_abs_share_fed"]
        + frame["pos_abs_share_banks"]
        + frame["pos_abs_share_row"]
        + frame["pos_abs_share_mmf"]
        + frame["pos_abs_share_domestic_nonbank"]
    )
    eff_q = frame["positive_absorption_total"].clip(lower=0.0) * core_share.clip(lower=0.0, upper=1.0)
    positive_eff = eff_q[eff_q > 0.0]
    if positive_eff.empty:
        return None, 0.0
    floor = max(
        Z1_POSITIVE_ABSORPTION_FLOOR_BN_SAAR,
        Z1_FLOOR_MEDIAN_MULTIPLIER * float(positive_eff.median()),
    )
    quarter_credit = (eff_q / floor).clip(lower=0.0, upper=1.0)
    n_eff = float(quarter_credit.sum())
    if n_eff <= 0.0:
        return None, 0.0

    weighted = {holder: 0.0 for holder in TDCSIM_HOLDERS}
    for idx, row in frame.iterrows():
        credit = float(quarter_credit.loc[idx])
        if credit <= 0.0:
            continue
        shares = _normalize_holder_shares(
            {
                "CB": _safe_float(row["pos_abs_share_fed"]),
                "Banks": _safe_float(row["pos_abs_share_banks"]),
                "Foreign": _safe_float(row["pos_abs_share_row"]),
                "Private": _safe_float(row["pos_abs_share_domestic_nonbank"])
                + _safe_float(row["pos_abs_share_mmf"]),
            }
        )
        for holder in TDCSIM_HOLDERS:
            weighted[holder] += credit * shares[holder]
    return _normalize_holder_shares({holder: value / n_eff for holder, value in weighted.items()}), n_eff


def _latest_z1_private_subbucket_absorption_shares(z1_absorption_path: Path) -> tuple[dict[str, float] | None, float]:
    if not z1_absorption_path.exists():
        return None, 0.0
    frame = pd.read_csv(z1_absorption_path)
    if "quarter" not in frame.columns:
        return None, 0.0
    required = [
        "positive_absorption_total",
        "pos_abs_share_mmf",
        "pos_abs_share_domestic_nonbank",
    ]
    if not set(required) <= set(frame.columns):
        return None, 0.0
    frame["quarter_date"] = pd.to_datetime(frame["quarter"], errors="coerce")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["quarter_date"])
        .dropna(subset=required)
        .sort_values("quarter_date")
        .tail(Z1_LOOKBACK_QUARTERS)
    )
    if frame.empty:
        return None, 0.0
    private_eff = frame["positive_absorption_total"].clip(lower=0.0) * (
        frame["pos_abs_share_mmf"].clip(lower=0.0)
        + frame["pos_abs_share_domestic_nonbank"].clip(lower=0.0)
    )
    positive_eff = private_eff[private_eff > 0.0]
    if positive_eff.empty:
        return None, 0.0
    floor = max(
        Z1_POSITIVE_ABSORPTION_FLOOR_BN_SAAR,
        Z1_FLOOR_MEDIAN_MULTIPLIER * float(positive_eff.median()),
    )
    quarter_credit = (private_eff / floor).clip(lower=0.0, upper=1.0)
    n_eff = float(quarter_credit.sum())
    if n_eff <= 0.0:
        return None, 0.0
    weighted = {route: 0.0 for route in PRIVATE_SUBBUCKET_SOURCE_BUCKETS}
    for idx, row in frame.iterrows():
        credit = float(quarter_credit.loc[idx])
        if credit <= 0.0:
            continue
        shares = _normalize_private_subbucket_shares(
            {
                PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: _safe_float(row["pos_abs_share_domestic_nonbank"]),
                PRIVATE_SUBBUCKET_MMF: _safe_float(row["pos_abs_share_mmf"]),
            }
        )
        for route in PRIVATE_SUBBUCKET_SOURCE_BUCKETS:
            weighted[route] += credit * shares[route]
    return _normalize_private_subbucket_shares({route: value / n_eff for route, value in weighted.items()}), n_eff


def _latest_primary_market_instrument_shares(
    path: Path,
    *,
    baseline_scalar: dict[str, float],
) -> dict[str, float] | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"date", "buyer_class", "share"}
    if not required <= set(frame.columns):
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    recent_dates = sorted(frame["date"].dropna().unique())[-PRIMARY_MARKET_LOOKBACK_QUARTERS:]
    frame = frame[frame["date"].isin(recent_dates)]
    if frame.empty:
        return None
    grouped = frame.groupby("buyer_class")["share"].mean()
    dealer_bridge_share = float(grouped.get("dealers", 0.0))
    raw = {
        "Banks": dealer_bridge_share * baseline_scalar.get("Banks", 0.0),
        "CB": float(grouped.get("fed", 0.0)) + dealer_bridge_share * baseline_scalar.get("CB", 0.0),
        "Foreign": float(grouped.get("foreign_official", 0.0))
        + dealer_bridge_share * baseline_scalar.get("Foreign", 0.0),
        "Private": dealer_bridge_share * baseline_scalar.get("Private", 0.0),
    }
    return _normalize_holder_shares(raw)


def _load_absorption_scenarios(
    tdcmix_prior_path: Path,
    *,
    z1_absorption_path: Path,
    bills_primary_path: Path,
    coupons_primary_path: Path,
    holder_absorption_calibration: dict[str, float | list[float]] | None = None,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, str], dict[str, dict[str, dict[str, float]]]]:
    calibration = holder_absorption_calibration or _default_holder_absorption_calibration()
    scenario_shift_lambda_default = float(calibration["scenario_shift_lambda_default"])
    scenario_shift_lambda_grid = [float(value) for value in calibration["scenario_shift_lambda_grid"]]
    tdcmix_targets = _load_tdcmix_targets(tdcmix_prior_path)
    tdcmix_private_targets = _load_tdcmix_private_subbucket_targets(tdcmix_prior_path)
    z1_recent, z1_n_eff = _latest_z1_absorption_shares(z1_absorption_path)
    z1_private_recent, z1_private_n_eff = _latest_z1_private_subbucket_absorption_shares(z1_absorption_path)
    z1_weight = z1_n_eff / (z1_n_eff + Z1_SHRINKAGE_K_QUARTERS) if z1_n_eff > 0.0 else 0.0
    z1_private_weight = (
        z1_private_n_eff / (z1_private_n_eff + Z1_SHRINKAGE_K_QUARTERS)
        if z1_private_n_eff > 0.0
        else 0.0
    )
    baseline_scalar = tdcmix_targets["current_mix_baseline"]
    if z1_recent is not None:
        baseline_scalar = _blend_shares(baseline_scalar, z1_recent, right_weight=z1_weight)
    baseline_private_split = _blend_private_subbucket_shares(
        tdcmix_private_targets["current_mix_baseline"],
        z1_private_recent,
        right_weight=z1_private_weight,
    )

    bills_primary = _latest_primary_market_instrument_shares(
        bills_primary_path,
        baseline_scalar=baseline_scalar,
    )
    coupons_primary = _latest_primary_market_instrument_shares(
        coupons_primary_path,
        baseline_scalar=baseline_scalar,
    )
    baseline_by_category: dict[str, dict[str, float]] = {}
    baseline_private_by_category: dict[str, dict[str, float]] = {}
    for category in PREFERENCE_CATEGORIES:
        instrument_prior = bills_primary if category == "bills" else coupons_primary
        shares = baseline_scalar
        if instrument_prior is not None:
            shares = _blend_shares(shares, instrument_prior, right_weight=PRIMARY_MARKET_OVERLAY_WEIGHT)
        baseline_by_category[category] = shares
        baseline_private_by_category[category] = baseline_private_split

    scenarios: dict[str, dict[str, dict[str, float]]] = {}
    private_subbucket_scenarios: dict[str, dict[str, dict[str, float]]] = {}
    registry = _ratewall_scenarios()
    holder_target_order = {
        target: index
        for index, target in enumerate(dict.fromkeys(row.holder_target for row in registry))
    }
    curve_order = {curve_id: index for index, curve_id in enumerate(RATE_SCENARIOS)}
    holder_scenarios = sorted(
        registry,
        key=lambda row: (
            holder_target_order[row.holder_target],
            curve_order[row.curve_id],
        ),
    )
    for scenario in holder_scenarios:
        if scenario.holder_target == "current_mix_baseline":
            scenarios[scenario.scenario_id] = baseline_by_category
            private_subbucket_scenarios[scenario.scenario_id] = baseline_private_by_category
            continue
        if scenario.holder_target not in tdcmix_targets:
            raise ValueError(
                f"RateWall scenario {scenario.scenario_id!r} references unknown holder target "
                f"{scenario.holder_target!r}."
            )
        target = tdcmix_targets[scenario.holder_target]
        scenarios[scenario.scenario_id] = {
            category: _blend_shares(shares, target, right_weight=scenario_shift_lambda_default)
            for category, shares in baseline_by_category.items()
        }
        target_private = tdcmix_private_targets[scenario.holder_target]
        private_subbucket_scenarios[scenario.scenario_id] = {
            category: _blend_private_subbucket_shares(
                baseline_private_by_category[category],
                target_private,
                right_weight=scenario_shift_lambda_default,
            )
            for category in PREFERENCE_CATEGORIES
        }

    metadata = {
        "baseline_sources": (
            "tdcmix_prior_contract"
            + (";z1_exact_recent_positive_absorption" if z1_recent is not None else "")
            + (";primary_market_instrument_buyer_composition" if bills_primary or coupons_primary else "")
        ),
        "dealer_bridge_mapping": "dealer_bridge_redistributed_by_pre_overlay_baseline_not_mapped_to_banks",
        "z1_effective_quarters_n_eff": f"{z1_n_eff:.12f}",
        "z1_shrinkage_weight": f"{z1_weight:.12f}",
        "z1_private_subbucket_effective_quarters_n_eff": f"{z1_private_n_eff:.12f}",
        "z1_private_subbucket_shrinkage_weight": f"{z1_private_weight:.12f}",
        "z1_shrinkage_k_quarters": f"{Z1_SHRINKAGE_K_QUARTERS:.1f}",
        "primary_market_overlay_weight": f"{PRIMARY_MARKET_OVERLAY_WEIGHT:.2f}",
        "primary_market_overlay_coverage": (
            "low_coverage_only_dealers_fed_foreign_official_share_weighted"
            if bills_primary or coupons_primary
            else "not_available"
        ),
        "scenario_shift_lambda_default": f"{scenario_shift_lambda_default:.2f}",
        "scenario_shift_lambda_grid": ",".join(f"{value:.2f}" for value in scenario_shift_lambda_grid),
        "tic_foreign_cross_check_status": "DEFERRED_new_plumbing_QA_only_not_blocking_blend_weight_grounding",
    }
    return scenarios, metadata, private_subbucket_scenarios


def _opening_price_curve(path: Path, *, record_date: str) -> tuple[list[float], list[float], str]:
    """Return the governed baseline curve on or before ``record_date`` and its curve date."""

    years, rates, label = curve_for_date(
        load_yield_curve_surface(path), record_date, scenario_id=OPENING_PRICE_CURVE["scenario_id"]
    )
    prefix = "dynamic_curve_surface:"
    if not years or not label.startswith(prefix):
        raise ValueError(f"{path.name} has no {OPENING_PRICE_CURVE['scenario_id']} curve on or before {record_date}: {label}")
    return years, rates, label[len(prefix):]


def _clean_price_ratio(item: dict, *, settlement: pd.Timestamp, curve_years: list[float], curve_rates: list[float]) -> float | None:
    """Clean market value per unit of book par at ``settlement``; None for a TIPS.

    Matured-unpaid debt is paid at par on the next business day. FRNs price at par. A
    TIPS needs a real curve, and the input directory governs none, so it is left out of
    the factor sample rather than priced on the nominal curve.
    """

    if item["maturity_date"] <= settlement:
        return 1.0
    if item["security_type"] == "TIPS":
        return None
    coupon_rate = item["coupon_rate"]
    if coupon_rate > 0.0 and item["issue_date"] > settlement:
        raise ValueError(f"{item['cusip']} pays a coupon but is issued after {settlement.date()}")
    years_to_maturity = (item["maturity_date"] - settlement).total_seconds() / (DAYS_PER_YEAR_ACTUAL * 24 * 60 * 60)
    market_yield = evaluate_nominal_yield(
        years_to_maturity,
        curve_years,
        curve_rates,
        method=OPENING_PRICE_CURVE["interpolation_method"],
        floor_zero=OPENING_PRICE_CURVE["floor_zero"],
        shock=None,
    )
    if not math.isfinite(float(market_yield)):
        raise ValueError(f"{item['cusip']} has no finite market yield on the opening curve")
    clean = float(
        value_treasury_security(
            settlement_date=settlement,
            maturity_date=item["maturity_date"],
            coupon_rate=coupon_rate,
            discount_yield=float(market_yield),
            security_type=item["security_type"],
            face_value=1.0,
            issue_date=item["issue_date"],
        )["clean"]
    )
    if not math.isfinite(clean) or clean <= 0.0:
        raise ValueError(f"{item['cusip']} produced an invalid clean price {clean}")
    return clean


def _route_key(route: tuple[str, str]) -> str:
    return "/".join(route).rstrip("/")


def _measured_par_bil(by_basis_mil: dict, *, factor: float, group: str) -> float:
    """Measured holdings at par: market values convert with the group factor; book values
    of bills too, while coupon securities at book are read as par."""

    book_factor = factor if group == "bills" else 1.0
    return (by_basis_mil["market_value"] * factor + by_basis_mil["book_value"] * book_factor) / 1000.0


def _opening_route_par(measured_mil: dict, *, noncb_par_bil: float, factor: float, group: str) -> dict:
    """Convert measured route holdings to par; the residual route takes the rest of the group's par."""

    if noncb_par_bil <= 0.0:
        raise ValueError(f"opening book has no non-Fed {group} par to allocate")
    par = {
        route: _measured_par_bil(measured_mil[route], factor=factor, group=group)
        for route in NONCB_OPENING_ROUTES
        if route != NONCB_RESIDUAL_ROUTE
    }
    par[NONCB_RESIDUAL_ROUTE] = noncb_par_bil - sum(par.values())
    negative = {_route_key(route): value for route, value in par.items() if value < 0.0}
    if negative:
        raise ValueError(
            f"opening {group} par holdings would be negative: {negative}; measured holdings at par "
            f"exceed the non-Fed {group} par of {noncb_par_bil:.3f} bn"
        )
    return {route: par[route] for route in NONCB_OPENING_ROUTES}


def _bill_discount_price(discount_yield: float, issue_date: pd.Timestamp, maturity_date: pd.Timestamp) -> float:
    """Auction price per unit face of a bill sold at ``discount_yield`` (actual/360)."""

    day_count = max(1.0, (maturity_date - issue_date).days)
    return max(0.75, min(1.0, 1.0 - discount_yield * day_count / 360.0))


def _bill_tranche_price(cusip: str, tranches: list[tuple[float, float]], outstanding_mil: float) -> float:
    """Face-weighted issue price over every auction tranche of one bill.

    The tranches must account for the bill's whole outstanding amount, so a missing or
    duplicated reopening fails here rather than mispricing the issue's proceeds.
    """

    face = sum(amount for amount, _ in tranches)
    if not tranches or not math.isfinite(outstanding_mil) or abs(face - outstanding_mil) > MSPD_TOTAL_TOLERANCE_MIL:
        raise ValueError(
            f"MSPD Table 3 bill tranches of {cusip} sum to {face} million, not its outstanding {outstanding_mil}"
        )
    return sum(amount * price for amount, price in tranches) / face


def build_initial_portfolio(
    output_path: Path,
    *,
    record_date: str,
    table1_rows: list[dict],
    table3_rows: list[dict],
    soma_path: Path,
    wamest_path: Path,
    yield_curve_path: Path,
    auctions_path: Path,
    nmfp_holdings_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Write the MSPD security-level opening book with CUSIP-level CB and WAMEST-split remainders.

    MSPD Table 3 securities, plus one cohort per class for debt that matured on a
    non-business record date and is paid on the next business day, are scaled to the
    Table 1 debt-held-by-public class total. CB holds the SOMA amount of each CUSIP
    (and SOMA's matured-unpaid amounts in the matured cohorts), plus the SOMA add-ons of
    auctions settling after the SOMA as-of date and by the record date. The rest is
    split among non-Fed holders by bills and nonbills: banks, foreign holders and money
    funds keep their WAMEST holdings converted to par by level basis
    (``WAMEST_LEVEL_BASES``) with the group's par-to-market factor on the governed
    baseline curve; domestic nonbanks,
    the Financial Accounts' residual sector, take the remaining non-Fed par. The split
    is initially uniform within bills and within nonbills. WAMEST has a dedicated
    money_market_funds sector, from which this builder uses the level and bill
    share. The uniform CUSIP split does not use that sector's detailed maturity
    profile. Money funds exchange every ineligible lot for an equal amount of
    eligible domestic nonbank paper, pro rata across the eligible lots. The
    exchange uses book par (inflation-adjusted outstanding for TIPS). Money funds
    then hold their SEC N-MFP floating-rate notes by CUSIP at the record date, for an
    equal par of their bills (``_apply_mmf_frn_holdings``), and the book's money-fund
    mix must match the filings within ``NMFP_MIX_TOLERANCE``. Money funds' share of new
    FRN supply is calibrated from the same filings (``calibrate_mmf_frn_auction_share``)
    and recorded with the new-issuance weights.
    """

    soma_as_of, soma = load_soma_cusip_holdings(soma_path)
    if soma_as_of > record_date:
        raise ValueError(f"SOMA as-of date {soma_as_of} is after MSPD record date {record_date}")
    noncb = load_wamest_noncb_weights(wamest_path, record_date=record_date)
    add_on_auctions, soma_add_ons = load_soma_auction_add_ons(auctions_path, after=soma_as_of, through=record_date)
    curve_years, curve_rates, curve_date = _opening_price_curve(yield_curve_path, record_date=record_date)
    targets = {
        row["security_class_desc"]: _safe_float(row.get("debt_held_public_mil_amt"))
        for row in table1_rows
        if row.get("security_type_desc") == "Marketable"
    }

    candidate_rows: list[dict] = []
    skipped: dict[str, int] = {}
    class_totals: dict[str, float] = {}
    unmatured_totals: dict[str, float] = {}
    # Every auction tranche of each bill, as (face outstanding, issue price per unit face).
    # A reopened bill was sold at a different price at each auction; its historical
    # proceeds are the sum over tranches, not the first tranche's price on the whole issue.
    bill_tranches: dict[str, list[tuple[float, float]]] = {}
    for row in table3_rows:
        class_desc = str(row.get("security_class1_desc", ""))
        class_key = _class_key(class_desc)
        sec_type = _security_type(class_desc)
        if not class_key or not sec_type or class_desc == "Total Marketable":
            continue
        label = str(row.get("security_class2_desc", "")).strip()
        outstanding = _safe_float(row.get("outstanding_amt"), math.nan)
        if label.startswith("Total "):
            # Subtotal rows. A "Total Matured" amount is derived as total less unmatured,
            # because MSPD prints "*" for amounts under $500k.
            if "Unmatured" in label:
                unmatured_totals[class_key] = outstanding
            elif "Matured" not in label:
                class_totals[class_key] = outstanding
            continue
        issue_date = pd.to_datetime(row.get("issue_date"), errors="coerce")
        maturity_date = pd.to_datetime(row.get("maturity_date"), errors="coerce")
        if class_key == "Bills" and not pd.isna(issue_date) and not pd.isna(maturity_date):
            tranche_face = _safe_float(row.get("issued_amt"), math.nan) + _safe_float(row.get("redeemed_amt"), 0.0)
            tranche_yield = _safe_float(row.get("yield_pct"), math.nan) / 100.0
            if not math.isfinite(tranche_face) or not math.isfinite(tranche_yield):
                raise ValueError(f"MSPD Table 3 bill tranche {label} issued {issue_date.date()} lacks an amount or yield")
            bill_tranches.setdefault(label, []).append(
                (tranche_face, _bill_discount_price(tranche_yield, issue_date, maturity_date))
            )
        if not math.isfinite(outstanding):
            # Reopening detail rows repeat a security without an outstanding amount.
            skipped["no_outstanding_amount"] = skipped.get("no_outstanding_amount", 0) + 1
            continue
        if outstanding <= 0 or pd.isna(issue_date) or pd.isna(maturity_date):
            raise ValueError(f"MSPD Table 3 security row {class_desc}:{label} has an amount without valid dates or a positive amount")
        candidate_rows.append(
            {
                "source_row": row,
                "class_key": class_key,
                "security_type": sec_type,
                "outstanding": outstanding,
                "issue_date": issue_date,
                "maturity_date": maturity_date,
                "original_maturity_years": max(0.01, (maturity_date - issue_date).days / 365.25),
                "cusip": label,
            }
        )

    raw_totals: dict[str, float] = {}
    cusip_outstanding: dict[str, float] = {}
    for item in candidate_rows:
        raw_totals[item["class_key"]] = raw_totals.get(item["class_key"], 0.0) + item["outstanding"]
        cusip_outstanding[item["cusip"]] = cusip_outstanding.get(item["cusip"], 0.0) + item["outstanding"]
    matured_mil: dict[str, float] = {}
    for class_key, raw_total in raw_totals.items():
        class_total = class_totals.get(class_key, math.nan)
        unmatured = unmatured_totals.get(class_key, class_total)
        if not math.isfinite(class_total) or not math.isfinite(unmatured):
            raise ValueError(f"MSPD Table 3 has no class total for {class_key}")
        if abs(raw_total - unmatured) > MSPD_TOTAL_TOLERANCE_MIL:
            raise ValueError(f"MSPD Table 3 {class_key} securities sum to {raw_total} but the unmatured total is {unmatured}")
        if class_total - unmatured < -MSPD_TOTAL_TOLERANCE_MIL:
            raise ValueError(f"MSPD Table 3 {class_key} unmatured total exceeds the class total")
        matured_mil[class_key] = max(0.0, class_total - unmatured)

    soma_amount = dict(zip(soma["cusip"], soma["amount_mil"]))
    # SOMA is weekly; the Fed's add-ons at auctions settling after its as-of date and by
    # the record date are Fed holdings at the record date.
    absent_add_ons = sorted(set(soma_add_ons) - set(cusip_outstanding))
    if absent_add_ons:
        raise ValueError(f"auctions settling in ({soma_as_of}, {record_date}] are absent from MSPD: {absent_add_ons}")
    for cusip, amount in soma_add_ons.items():
        soma_amount[cusip] = soma_amount.get(cusip, 0.0) + amount
    unmatched = soma[~soma["cusip"].isin(cusip_outstanding)]
    unmatched_live = unmatched[unmatched["maturity_date"] > record_date]
    if not unmatched_live.empty:
        raise ValueError(f"SOMA CUSIPs outstanding after {record_date} are absent from MSPD: {sorted(unmatched_live['cusip'])}")
    # SOMA securities maturing on a non-business record date are still outstanding as
    # matured-unpaid debt; earlier maturities were paid before the record date.
    matured_unpaid = unmatched[unmatched["maturity_date"] == record_date]
    paid_before_record = unmatched[unmatched["maturity_date"] < record_date]
    matured_cb_mil = {class_key: 0.0 for class_key in matured_mil}
    for soma_type, amount in matured_unpaid.groupby("security_type")["amount_mil"].sum().items():
        classes = [key for key in SOMA_TYPE_CLASSES.get(soma_type, ()) if matured_mil.get(key, 0.0) > 0.0]
        if not classes:
            raise ValueError(f"SOMA holds matured {soma_type} on {record_date} but MSPD shows no matured amount for it")
        pool = sum(matured_mil[key] for key in classes)
        for key in classes:
            matured_cb_mil[key] += amount * matured_mil[key] / pool
    record_ts = pd.Timestamp(record_date)
    for class_key, amount in matured_mil.items():
        if amount <= 0.0:
            continue
        candidate_rows.append(
            {
                "source_row": {"security_class1_desc": class_key},
                "class_key": class_key,
                "security_type": "Fixed" if class_key in {"Bills", "Notes", "Bonds"} else _security_type(class_key),
                "outstanding": amount,
                "issue_date": record_ts - pd.Timedelta(days=1),
                "maturity_date": record_ts,
                "original_maturity_years": 1.0 / 365.25,
                "cusip": f"matured_unpaid_{class_key.lower().replace(' ', '_').replace('-', '_')}",
                "matured_cb_mil": matured_cb_mil[class_key],
                "maturity_category": MATURED_CLASS_CATEGORIES.get(class_key, ""),
            }
        )

    groups = ("bills", "nonbills")
    noncb_par = {group: 0.0 for group in groups}
    priced = {group: {"par": 0.0, "market": 0.0} for group in groups}
    unpriced_par = {group: 0.0 for group in groups}
    for item in candidate_rows:
        target_total = targets.get(item["class_key"], 0.0)
        if target_total <= 0:
            raise ValueError(f"MSPD Table 1 has no debt-held-by-public total for {item['class_key']}")
        # Table 1 excludes intragovernmental holdings; scale each class's Table 3 total to it.
        class_scale = target_total / class_totals[item["class_key"]]
        cusip = item["cusip"]
        item["coupon_rate"] = _safe_float(item["source_row"].get("interest_rate_pct")) / 100.0
        if "matured_cb_mil" not in item and item["security_type"] in {"Fixed", "TIPS"} and item["class_key"] != "Bills":
            raw_rate = _safe_float(item["source_row"].get("interest_rate_pct"), math.nan)
            if not math.isfinite(raw_rate) or raw_rate <= 0.0:
                raise ValueError(f"MSPD Table 3 coupon security {cusip} has no valid interest rate")
        # MSPD and SOMA amounts are in millions of dollars; contract columns use billions.
        item["public_face"] = item["outstanding"] * class_scale / 1000.0
        if "matured_cb_mil" in item:
            item["cb_face"] = item["matured_cb_mil"] / 1000.0
        else:
            item["cb_face"] = soma_amount.get(cusip, 0.0) * item["outstanding"] / cusip_outstanding[cusip] / 1000.0
        if item["cb_face"] > item["public_face"] * (1.0 + 1e-9) + 1e-9:
            raise ValueError(
                f"SOMA (with auction add-ons) holds more of {cusip} than MSPD debt held by the public: "
                f"{item['cb_face']} > {item['public_face']}"
            )
        item["remainder"] = max(0.0, item["public_face"] - item["cb_face"])
        group = item["group"] = "bills" if item["class_key"] == "Bills" else "nonbills"
        noncb_par[group] += item["remainder"]
        if item["remainder"] <= 0.0:
            continue
        ratio = _clean_price_ratio(item, settlement=record_ts, curve_years=curve_years, curve_rates=curve_rates)
        if ratio is None:
            unpriced_par[group] += item["remainder"]
        else:
            priced[group]["par"] += item["remainder"]
            priced[group]["market"] += item["remainder"] * ratio
    factors = {}
    route_par = {}
    for group in groups:
        if priced[group]["market"] <= 0.0:
            raise ValueError(f"opening book has no priced non-Fed {group} to convert WAMEST holdings to par")
        factors[group] = priced[group]["par"] / priced[group]["market"]
        route_par[group] = _opening_route_par(
            {route: noncb["holdings_by_basis_usd_mil"][route][group] for route in NONCB_OPENING_ROUTES},
            noncb_par_bil=noncb_par[group],
            factor=factors[group],
            group=group,
        )
    route_weights = {
        group: {route: route_par[group][route] / noncb_par[group] for route in NONCB_OPENING_ROUTES} for group in groups
    }
    for item in candidate_rows:
        item["route_faces"] = {
            route: item["remainder"] * share for route, share in route_weights[item["group"]].items()
        }
    eligibility = mmf_eligible_mask(
        pd.DataFrame(
            {"SecurityType": item["security_type"], "MaturityDate": item["maturity_date"]}
            for item in candidate_rows
        ),
        record_date,
    )
    mmf_offered = sum(
        item["route_faces"][MMF_OPENING_ROUTE] for item, eligible in zip(candidate_rows, eligibility) if not eligible
    )
    nonbank_available = sum(
        item["route_faces"][NONCB_RESIDUAL_ROUTE] for item, eligible in zip(candidate_rows, eligibility) if eligible
    )
    if mmf_offered > nonbank_available:
        raise ValueError(
            f"opening money-fund swap needs {mmf_offered:.6f} billion of eligible domestic nonbank paper, "
            f"but only {nonbank_available:.6f} billion is available"
        )
    swap_share = mmf_offered / nonbank_available if mmf_offered else 0.0
    swap_by_class: dict[str, dict[str, float]] = {"from_mmf": {}, "to_mmf": {}}
    for item, eligible in zip(candidate_rows, eligibility):
        faces = item["route_faces"]
        class_name = item["class_key"]
        if eligible:
            amount = faces[NONCB_RESIDUAL_ROUTE] * swap_share
            faces[MMF_OPENING_ROUTE] += amount
            faces[NONCB_RESIDUAL_ROUTE] -= amount
            direction = "to_mmf"
        else:
            amount = faces[MMF_OPENING_ROUTE]
            faces[MMF_OPENING_ROUTE] = 0.0
            faces[NONCB_RESIDUAL_ROUTE] += amount
            direction = "from_mmf"
        if amount > 0.0:
            totals = swap_by_class[direction]
            totals[class_name] = totals.get(class_name, 0.0) + amount
    nmfp, nmfp_provenance = load_nmfp_mmf_holdings(nmfp_holdings_path)
    filed = nmfp.loc[nmfp["report_date"].eq(record_date)]
    if filed.empty:
        raise ValueError(f"{nmfp_holdings_path.name} has no money-fund holdings reported at {record_date}")
    frn_override = _apply_mmf_frn_holdings(
        candidate_rows, filed.loc[filed["security_class"].eq("frn")].set_index("cusip")["value_bil"].to_dict()
    )
    frn_override["source"] = {
        "file": nmfp_holdings_path.name,
        "sha256": _sha256(nmfp_holdings_path),
        "report_date": record_date,
        "producer": nmfp_provenance.get("producer"),
        "filed_through": nmfp_provenance["filed_through"],
        "value_basis": nmfp_provenance.get("value_basis"),
        "zips": nmfp_provenance["zips"],
    }
    mix_check = _check_mmf_mix(candidate_rows, filed, record_date=record_date)
    frn_auction_share = calibrate_mmf_frn_auction_share(nmfp, auctions_path)
    frn_auction_share["nmfp_source"] = {"file": nmfp_holdings_path.name, "sha256": _sha256(nmfp_holdings_path)}

    output_rows: list[dict] = []
    bond_id = 1
    for item in candidate_rows:
        is_matured = "matured_cb_mil" in item
        source_row = item["source_row"]
        class_desc = str(source_row.get("security_class1_desc", ""))
        cusip = item["cusip"]
        coupon_rate = item["coupon_rate"]
        issue_yield = _safe_float(source_row.get("yield_pct"), coupon_rate * 100.0) / 100.0
        maturity_years = item["original_maturity_years"]
        maturity_category = item.get("maturity_category") or _maturity_category(class_desc, maturity_years)
        is_bill = maturity_category == "bills" and coupon_rate <= 1e-12
        price_ratio = 1.0
        if is_bill and is_matured:
            # Matured-unpaid bills are one synthetic row per class with no auction tranches;
            # they redeem at the opening at their own declared yield.
            price_ratio = _bill_discount_price(issue_yield, item["issue_date"], item["maturity_date"])
        elif is_bill:
            price_ratio = _bill_tranche_price(cusip, bill_tranches.get(cusip, []), cusip_outstanding.get(cusip, math.nan))
        group = item["group"]
        has_add_on = soma_add_ons.get(cusip, 0.0) > 0.0
        frn_observed = item["security_type"] == "FRN"
        frn_override_leg = frn_observed or item["class_key"] == "Bills"
        routes = [
            (holder, subbucket, face) for (holder, subbucket), face in item["route_faces"].items()
        ]
        routes.insert(1, ("CB", "", item["cb_face"]))
        for holder, holder_subbucket, face_value in routes:
            if face_value <= 1e-9:
                continue
            is_cb = holder == "CB"
            is_mmf_frn = frn_observed and (holder, holder_subbucket) == MMF_OPENING_ROUTE
            original_principal = face_value if item["security_type"] == "TIPS" else 0.0
            output_rows.append(
                {
                    "BondID": bond_id,
                    "SecurityType": item["security_type"],
                    "IssueDate": item["issue_date"].date().isoformat(),
                    "MaturityDate": item["maturity_date"].date().isoformat(),
                    "OriginalMaturityYears": f"{maturity_years:.8f}",
                    "FaceValue": f"{face_value:.12f}",
                    "CouponRate": f"{coupon_rate:.12f}",
                    "HolderType": holder,
                    "HolderSubBucket": holder_subbucket,
                    "Status": "Active",
                    "MaturityCategory": maturity_category or "",
                    "OriginalPrincipal": f"{original_principal:.12f}",
                    "AdjustedPrincipal": f"{original_principal:.12f}",
                    "ReferenceCPI_Issue": "100.0" if item["security_type"] == "TIPS" else "0.0",
                    "IndexRatio": "1.0" if item["security_type"] == "TIPS" else "0.0",
                    "FixedSpread": "0.0005" if item["security_type"] == "FRN" else "0.0",
                    "AccruedInterest_FRN": "0.0",
                    "BenchmarkRate_FRN": f"{issue_yield:.12f}" if item["security_type"] == "FRN" else "0.0",
                    "LastAccrualDate": item["issue_date"].date().isoformat() if item["security_type"] == "FRN" else "",
                    "IssuePriceRatio": f"{price_ratio:.12f}",
                    "IssueProceeds": f"{face_value * price_ratio:.12f}",
                    "IssueYieldAtIssue": f"{issue_yield:.12f}",
                    "TimeToMaturity": "",
                    "DiscountYield": "",
                    "CleanPrice": "",
                    "AccruedInterest": "",
                    "DirtyValue": "",
                    "DirtyPriceRatio": "",
                    "cohort_id": f"mspd_{record_date}_{cusip}_{bond_id}",
                    "cusip_or_synthetic_id": cusip,
                    "issue_proceeds_bil": f"{face_value * price_ratio:.12f}",
                    "face_value_bil": f"{face_value:.12f}",
                    "adjusted_principal_bil": f"{original_principal:.12f}",
                    "holder_bucket": holder,
                    "holder_subbucket": holder_subbucket,
                    "source_holder_bucket": (
                        "domestic_nonbank"
                        if holder_subbucket == PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
                        else "mmf"
                        if holder_subbucket == PRIVATE_SUBBUCKET_MMF
                        else holder
                    ),
                    "ratewall_perimeter": "DU" if holder == "Private" else "RU",
                    "source_family": (
                        "fiscaldata_mspd_table_3_market;nyfed_soma_cusip_holdings"
                        + (";fiscaldata_auctions_soma_accepted" if has_add_on else "")
                        if is_cb
                        else "fiscaldata_mspd_table_3_market;sec_nmfp_mmf_holdings"
                        if is_mmf_frn
                        else "fiscaldata_mspd_table_3_market;wamest_sector_bill_share"
                    ),
                    "source_key": f"{record_date}:{class_desc}:{cusip}",
                    "observability_tier": (
                        "class_matured_unpaid_total"
                        if is_matured
                        else "security_level_observed_holder"
                        if is_cb or is_mmf_frn
                        else "security_cohort_remainder_with_sector_bill_nonbill_allocation"
                    ),
                    "central_default_eligible": "true",
                    "sensitivity_only": "false",
                    "source_status": (
                        ("mspd_matured_unpaid_class_amount;" if is_matured else "")
                        + "mspd_million_amounts_scaled_to_table1_class_total_and_converted_to_bil;"
                        + (
                            f"cb_from_nyfed_soma_asof_{soma_as_of}"
                            + (f"_plus_auction_soma_add_ons_through_{record_date}" if has_add_on else "")
                            if is_cb
                            else f"mmf_frn_holdings_from_sec_nmfp_{record_date}_value_read_as_par"
                            if is_mmf_frn
                            else f"noncb_remainder_split_by_wamest_{noncb['wamest_date']}_{group}_holdings_at_par_"
                            "domestic_nonbank_residual_not_cusip_holder_evidence"
                            + (
                                ";mmf_eligibility_swap"
                                if (holder, holder_subbucket) in (MMF_OPENING_ROUTE, NONCB_RESIDUAL_ROUTE)
                                else ""
                            )
                            + (";mmf_nmfp_frn_holdings_override" if frn_override_leg else "")
                        )
                    ),
                    "claim_boundary": RATEWALL_CLAIM_BOUNDARY,
                }
            )
            bond_id += 1

    frame = pd.DataFrame(output_rows)
    frame.to_csv(output_path, index=False)
    holder_totals: dict[str, float] = {}
    for row in output_rows:
        key = f"{row['HolderType']}/{row['HolderSubBucket']}".rstrip("/")
        holder_totals[key] = holder_totals.get(key, 0.0) + float(row["FaceValue"])
    metadata = {
        "mspd_record_date": record_date,
        "rows": len(frame),
        "mspd_rows_skipped": dict(sorted(skipped.items())),
        "source_url_table_1": f"{FISCALDATA_BASE}/v1/debt/mspd/mspd_table_1",
        "source_url_table_3_market": f"{FISCALDATA_BASE}/v1/debt/mspd/mspd_table_3_market",
        "cb_source": "nyfed_soma_treasury_holdings_by_cusip",
        "soma_as_of_date": soma_as_of,
        "soma_file_sha256": _sha256(soma_path),
        "soma_matured_unpaid_at_record_date": {
            "cusips": int(len(matured_unpaid)),
            "amount_bil": float(matured_unpaid["amount_mil"].sum() / 1000.0),
        },
        # The Fed reinvests these at auctions MSPD records under new CUSIPs; the add-ons
        # below put those new holdings with CB.
        "soma_paid_before_record_date": {
            "cusips": int(len(paid_before_record)),
            "amount_bil": float(paid_before_record["amount_mil"].sum() / 1000.0),
        },
        "soma_auction_add_ons": {
            "settlement_window": f"({soma_as_of}, {record_date}]",
            "auctions": add_on_auctions,
            "cusips_with_add_on": sum(1 for value in soma_add_ons.values() if value > 0.0),
            "amount_bil": sum(soma_add_ons.values()) / 1000.0,
            "amount_bil_by_cusip": {cusip: value / 1000.0 for cusip, value in soma_add_ons.items()},
            "source": auctions_path.name,
            "source_sha256": _sha256(auctions_path),
        },
        "matured_unpaid_bil": {key: value / 1000.0 for key, value in sorted(matured_mil.items()) if value > 0.0},
        "noncb_source": WAMEST_CANONICAL_RELATIVE_PATH,
        "noncb_source_sha256": _sha256(wamest_path),
        "wamest_date": noncb["wamest_date"],
        # New-issuance holder shares read these measured shares, except that money funds
        # take the N-MFP-calibrated share of new FRNs; the opening book's own pre-swap split
        # is opening_allocation.groups.*.route_weights.
        "noncb_weights": {
            **{
                group: {_route_key(route): share for route, share in shares.items()}
                for group, shares in noncb["weights"].items()
            },
            FRN_MMF_AUCTION_SHARE_KEY: frn_auction_share["share"],
        },
        "noncb_weights_basis": (
            "wamest_measured_market_value_shares_for_new_issuance_not_opening_split;"
            "frn_money_fund_share_calibrated_from_sec_nmfp"
        ),
        "frn_mmf_auction_share_calibration": frn_auction_share,
        "opening_allocation": {
            "rule": OPENING_ALLOCATION_RULE,
            "measured_holdings_basis": "wamest_level_times_bill_share_by_level_basis",
            "level_basis_field": "revaluation_valuation_basis",
            "level_bases": dict(sorted(WAMEST_LEVEL_BASES.items())),
            "market_value_treatment": "converted_with_group_factor",
            "book_value_treatment": "bills_converted_with_bill_factor_coupon_securities_read_as_par",
            "residual_route": _route_key(NONCB_RESIDUAL_ROUTE),
            "price_basis": "clean_market_value_per_unit_of_book_par",
            "pricing_date": record_date,
            "curve_file": yield_curve_path.name,
            "curve_sha256": _sha256(yield_curve_path),
            **{f"curve_{key}": value for key, value in OPENING_PRICE_CURVE.items()},
            "curve_date": curve_date,
            "frn_treatment": "priced_at_par",
            "matured_unpaid_treatment": "valued_at_par",
            "tips_treatment": "excluded_from_factor_sample_no_governed_real_curve_take_group_factor",
            "tips_par_basis": "mspd_outstanding_including_inflation_adjustment",
            "money_fund_eligibility_swap": {
                "wamest_sector": "money_market_funds",
                "wamest_inputs": "sector level and bill share; the pre-swap CUSIP split is uniform within bills and nonbills",
                "rule": "exchange every ineligible money-fund lot for eligible domestic nonbank lots pro rata at book par",
                "as_of_date": record_date,
                "face_bil_by_class": swap_by_class,
            },
            "money_fund_frn_holdings_override": frn_override,
            "groups": {
                group: {
                    "par_to_market_factor": factors[group],
                    "noncb_par_bil": noncb_par[group],
                    "priced_par_bil": priced[group]["par"],
                    "priced_market_value_bil": priced[group]["market"],
                    "unpriced_tips_par_bil": unpriced_par[group],
                    "measured_bil_by_basis": {
                        _route_key(route): {
                            basis: amount / 1000.0
                            for basis, amount in noncb["holdings_by_basis_usd_mil"][route][group].items()
                        }
                        for route in NONCB_OPENING_ROUTES
                    },
                    "par_bil": {_route_key(route): route_par[group][route] for route in NONCB_OPENING_ROUTES},
                    "domestic_nonbank_residual_bil": route_par[group][NONCB_RESIDUAL_ROUTE],
                    # Non-Fed par that no measured holding at par accounts for; the residual
                    # route absorbs it on top of its own measured holdings at par.
                    "residual_gap_bil": noncb_par[group]
                    - sum(
                        _measured_par_bil(noncb["holdings_by_basis_usd_mil"][route][group], factor=factors[group], group=group)
                        for route in NONCB_OPENING_ROUTES
                    ),
                    "route_weights": {_route_key(route): route_weights[group][route] for route in NONCB_OPENING_ROUTES},
                }
                for group in groups
            },
        },
        "holder_totals_bil": dict(sorted(holder_totals.items())),
        "nmfp_mix_check": mix_check,
    }
    return frame, metadata


def _read_xlsx(path: Path) -> dict[str, list[list[str]]]:
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{main_ns}}}si"):
                shared_strings.append(_clean("".join(node.text or "" for node in item.iter(f"{{{main_ns}}}t"))))
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{{{pkg_rel_ns}}}Relationship")
        }
        sheets: dict[str, list[list[str]]] = {}
        for sheet in workbook.findall(f".//{{{main_ns}}}sheet"):
            name = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{rel_ns}}}id"]
            target = rel_targets[rel_id].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            sheet_root = ET.fromstring(archive.read(target))
            rows: list[list[str]] = []
            for row_node in sheet_root.findall(f".//{{{main_ns}}}row"):
                values: dict[int, str] = {}
                max_col = -1
                for cell in row_node.findall(f"{{{main_ns}}}c"):
                    col_idx = _cell_col_index(cell.attrib.get("r", ""))
                    if col_idx < 0:
                        continue
                    values[col_idx] = _cell_value(cell, shared_strings, main_ns)
                    max_col = max(max_col, col_idx)
                rows.append([values.get(idx, "") for idx in range(max_col + 1)])
            sheets[name] = rows
    return sheets


def _cell_col_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return -1
    col_idx = 0
    for char in match.group(1):
        col_idx = col_idx * 26 + (ord(char) - ord("A") + 1)
    return col_idx - 1


def _cell_value(cell: ET.Element, shared_strings: list[str], main_ns: str) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return _clean("".join(node.text or "" for node in cell.iter(f"{{{main_ns}}}t")))
    value_node = cell.find(f"{{{main_ns}}}v")
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text.strip()
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (IndexError, ValueError):
            return raw
    return _clean(raw)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _cbo_budget_values(path: Path) -> dict[int, dict[str, float]]:
    workbook = _read_xlsx(path)
    rows = workbook["Table 1-1"]
    year_cols = _year_cols(rows)
    values: dict[int, dict[str, float]] = {}
    section = ""
    for row in rows:
        label = _clean(row[0] if row else "").lower()
        joined = _clean(" ".join(row)).lower()
        if "in billions of dollars" in joined:
            section = "billions"
            continue
        if "as a percentage of gdp" in joined:
            section = "percent_of_gdp"
            continue
        if section != "billions":
            continue
        metric = None
        if label == "net interest":
            metric = "net_interest_bil"
        elif label == "total deficit (-)":
            metric = "deficit_bil"
        elif label == "debt held by the public":
            metric = "debt_held_public_bil"
        if metric is None:
            continue
        for col_idx, year in year_cols.items():
            if col_idx >= len(row) or not row[col_idx]:
                continue
            values.setdefault(int(year), {})[metric] = _safe_float(row[col_idx])
    return values


def _cbo_rate_values(path: Path) -> dict[int, dict[str, float]]:
    workbook = _read_xlsx(path)
    rows = workbook["Table 2-1"]
    header = rows[6]
    year_cols = {
        idx: value
        for idx, value in enumerate(header)
        if value and (re.fullmatch(r"20[0-9]{2}", value) or "–" in value or "-" in value)
    }
    values: dict[int, dict[str, float]] = {}
    for row in rows:
        label = _clean(row[0] if row else "").lower()
        if label not in {"3-month treasury bills", "10-year treasury notes"}:
            continue
        key = "cbo_3m_pct" if label.startswith("3-month") else "cbo_10y_pct"
        for col_idx, year_label in year_cols.items():
            if col_idx >= len(row) or not row[col_idx]:
                continue
            years = _expand_year_label(year_label)
            for year in years:
                values.setdefault(year, {})[key] = _safe_float(row[col_idx])
    return values


def _year_cols(rows: list[list[str]]) -> dict[int, str]:
    for row in rows:
        years = {
            idx: _clean(value)
            for idx, value in enumerate(row)
            if re.fullmatch(r"20[0-9]{2}", _clean(value))
        }
        if "2026" in years.values() and "2036" in years.values():
            return years
    return {}


def _expand_year_label(label: str) -> list[int]:
    label = _clean(label).replace(" ", "")
    if re.fullmatch(r"20[0-9]{2}", label):
        return [int(label)]
    match = re.fullmatch(r"(20[0-9]{2})[–-](20[0-9]{2})", label)
    if match:
        return list(range(int(match.group(1)), int(match.group(2)) + 1))
    return []


def _federal_fiscal_year_calendar_quarters(year: int) -> list[str]:
    return [f"{year - 1}Q4", f"{year}Q1", f"{year}Q2", f"{year}Q3"]


def build_primary_flow_path(
    output_path: Path,
    *,
    cbo_budget_path: Path,
    scenario_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    selected_scenario_ids = (
        scenario_ids
        if scenario_ids is not None
        else [scenario.scenario_id for scenario in _ratewall_scenarios()]
    )
    values = _cbo_budget_values(cbo_budget_path)
    rows: list[dict] = []
    for year, metrics in sorted(values.items()):
        if year < 2026:
            continue
        deficit_magnitude = -metrics.get("deficit_bil", 0.0)
        net_interest = metrics.get("net_interest_bil", 0.0)
        primary_deficit = deficit_magnitude - net_interest
        for quarter in _federal_fiscal_year_calendar_quarters(year):
            for scenario_id in selected_scenario_ids:
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "quarter": quarter,
                        "primary_fiscal_flow_to_du_bil": f"{primary_deficit / 4.0:.12f}",
                        "non_debt_outlays_to_du_bil": "",
                        "receipts_from_du_bil": "",
                        "primary_flow_status": "aggregate_cash_proxy_from_cbo_total_deficit_less_net_interest",
                        "aggregate_primary_deficit_bil": f"{primary_deficit:.12f}",
                        "du_primary_flow_share_assumption": "1.0",
                        "recipient_mapping_status": "recipient_mapping_unavailable_aggregate_proxy",
                        "assumption_mode": "true",
                        "central_default_eligible": "true",
                        "source_family": "cbo_budget_projection",
                        "source_key": "51118-2026-02-Budget-Projections.xlsx:Table 1-1",
                        "source_status": "source_backed_federal_fiscal_year_mapped_to_calendar_quarters_aggregate_primary_deficit_proxy_not_recipient_mapped",
                        "claim_boundary": RATEWALL_CLAIM_BOUNDARY,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame, {
        "cbo_budget_path": str(cbo_budget_path),
        "fiscal_year_calendar_mapping": "federal_fy_y_to_calendar_y_minus_1_q4_through_y_q3",
        "rows": len(frame),
    }


def _latest_treasury_curve() -> tuple[str, dict[float, float]]:
    bodies = []
    now = pd.Timestamp.now(tz="UTC")
    for offset in range(0, 8):
        month = now - pd.DateOffset(months=offset)
        url = f"{TREASURY_CURVE_XML}&field_tdr_date_value_month={month.strftime('%Y%m')}"
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
            body = response.read().decode("utf-8", errors="replace")
        bodies.append(body)
        if "BC_10YEAR" in body:
            break
    body = "\n".join(bodies)
    root = ET.fromstring(body)
    tenor_map = {
        "BC_3MONTH": 0.25,
        "BC_6MONTH": 0.5,
        "BC_1YEAR": 1.0,
        "BC_2YEAR": 2.0,
        "BC_5YEAR": 5.0,
        "BC_10YEAR": 10.0,
        "BC_20YEAR": 20.0,
        "BC_30YEAR": 30.0,
    }
    latest_date = ""
    latest_rates: dict[float, float] = {}
    for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        props = None
        for node in entry.iter():
            if node.tag.endswith("properties"):
                props = node
                break
        if props is None:
            continue
        values = {child.tag.split("}")[-1]: child.text for child in list(props)}
        date_text = str(values.get("NEW_DATE", ""))[:10]
        if date_text <= latest_date:
            continue
        rates = {
            tenor: _safe_float(values.get(field)) / 100.0
            for field, tenor in tenor_map.items()
            if values.get(field) not in {None, ""}
        }
        if rates:
            latest_date = date_text
            latest_rates = rates
    if not latest_rates:
        raise ValueError("Treasury yield curve XML did not contain rate rows.")
    return latest_date, latest_rates


def _interp_rate(curve: dict[float, float], tenor: float) -> float:
    if tenor in curve:
        return curve[tenor]
    points = sorted(curve)
    if tenor <= points[0]:
        return curve[points[0]]
    if tenor >= points[-1]:
        return curve[points[-1]]
    for left, right in zip(points, points[1:]):
        if left <= tenor <= right:
            weight = (tenor - left) / (right - left)
            return curve[left] + weight * (curve[right] - curve[left])
    return curve[points[-1]]


def _shape_preserving_curve(base_curve: dict[float, float], *, cbo_3m: float, cbo_10y: float) -> dict[float, float]:
    base_3m = _interp_rate(base_curve, 0.25)
    base_10y = _interp_rate(base_curve, 10.0)
    short_shift = cbo_3m - base_3m
    long_shift = cbo_10y - base_10y
    out: dict[float, float] = {}
    for tenor in TENORS:
        if tenor <= 1.0:
            shift = short_shift
        elif tenor >= 10.0:
            shift = long_shift
        else:
            shift = short_shift + (long_shift - short_shift) * ((tenor - 1.0) / 9.0)
        out[tenor] = max(0.0, _interp_rate(base_curve, tenor) + shift)
    return out


def build_yield_curve_surface(output_path: Path, *, cbo_economic_path: Path) -> tuple[pd.DataFrame, dict]:
    current_date, current_curve = _latest_treasury_curve()
    cbo_rates = _cbo_rate_values(cbo_economic_path)
    rows: list[dict] = []
    for scenario_id, scenario in RATE_SCENARIOS.items():
        for year, rates in sorted(cbo_rates.items()):
            if year < 2026:
                continue
            cbo_3m_investment_rate = discount_rate_to_investment_rate(
                rates["cbo_3m_pct"] / 100.0,
                CBO_3M_BILL_DAYS,
            )
            curve = _shape_preserving_curve(
                current_curve,
                cbo_3m=cbo_3m_investment_rate + float(scenario["short_shift"]),
                cbo_10y=rates["cbo_10y_pct"] / 100.0
                + float(scenario["long_shift"]),
            )
            for quarter_idx in range(1, 5):
                curve_date = pd.Timestamp(year=year, month=(quarter_idx - 1) * 3 + 1, day=1).date().isoformat()
                for tenor, rate in curve.items():
                    rows.append(
                        {
                            "scenario_id": scenario_id,
                            "curve_date": curve_date,
                            "tenor_years": f"{tenor:.2f}",
                            "nominal_rate": f"{rate:.12f}",
                            "rate_type": "nominal",
                            "curve_basis": "treasury_par_shape_preserving_cbo_3m_10y_anchor",
                            "source_family": "treasury_daily_yield_curve;cbo_economic_projection",
                            "source_key": f"{current_date};51135-2026-02-Economic-Projections.xlsx:Table 2-1",
                            "interpolation_method": (
                                "current_curve_shape_shifted_to_cbo_3m_10y_annual_anchors;"
                                f"short_shift={scenario['short_shift']};"
                                f"long_shift={scenario['long_shift']}"
                            ),
                            "central_default_eligible": str(
                                scenario["central_default_eligible"]
                            ),
                            "sensitivity_only": str(scenario["sensitivity_only"]),
                            "source_status": str(scenario["source_status"]),
                            "claim_boundary": RATEWALL_CLAIM_BOUNDARY,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame, {"treasury_curve_date": current_date, "rows": len(frame)}


def build_holder_absorption_path(
    output_path: Path,
    *,
    tdcmix_prior_path: Path,
    z1_absorption_path: Path,
    bills_primary_path: Path,
    coupons_primary_path: Path,
    holder_absorption_calibration: dict[str, float | list[float]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    scenarios, metadata, private_subbucket_scenarios = _load_absorption_scenarios(
        tdcmix_prior_path,
        z1_absorption_path=z1_absorption_path,
        bills_primary_path=bills_primary_path,
        coupons_primary_path=coupons_primary_path,
        holder_absorption_calibration=holder_absorption_calibration,
    )
    rows: list[dict] = []
    for scenario_id, category_shares in scenarios.items():
        for year in range(2026, 2037):
            for quarter_idx in range(1, 5):
                quarter = f"{year}Q{quarter_idx}"
                for holder in TDCSIM_HOLDERS:
                    holder_rows = [("", holder, None)]
                    if holder == "Private":
                        holder_rows = [
                            (
                                PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
                                "domestic_nonbank",
                                1.0,
                            ),
                            (
                                PRIVATE_SUBBUCKET_MMF,
                                "mmf",
                                MMF_DEPOSIT_PASS_THROUGH_DEFAULT,
                            ),
                        ]
                    for holder_subbucket, source_holder_bucket, pass_through in holder_rows:
                        values = {}
                        for category in PREFERENCE_CATEGORIES:
                            holder_share = category_shares[category].get(holder, 0.0)
                            if holder == "Private":
                                route_share = private_subbucket_scenarios[scenario_id][category].get(
                                    holder_subbucket,
                                    0.0,
                                )
                                holder_share *= route_share
                            values[category] = holder_share
                        rows.append(
                            {
                                "scenario_id": scenario_id,
                                "quarter": quarter,
                                "holder_type": holder,
                                "holder_subbucket": holder_subbucket,
                                "source_holder_bucket": source_holder_bucket,
                                "route_class": (
                                    "du_deposit_funded"
                                    if holder_subbucket == PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
                                    else "mmf_cash_fund_plumbing"
                                    if holder_subbucket == PRIVATE_SUBBUCKET_MMF
                                    else ""
                                ),
                                "deposit_pass_through": (
                                    "" if pass_through is None else f"{pass_through:.2f}"
                                ),
                                "mmf_deposit_pass_through": (
                                    f"{MMF_DEPOSIT_PASS_THROUGH_DEFAULT:.2f}" if holder == "Private" else ""
                                ),
                                "mmf_ru_plumbing_pass_through": (
                                    f"{1.0 - MMF_DEPOSIT_PASS_THROUGH_DEFAULT:.2f}"
                                    if holder == "Private"
                                    else ""
                                ),
                                "mmf_deposit_pass_through_status": (
                                    MMF_DEPOSIT_PASS_THROUGH_STATUS if holder == "Private" else ""
                                ),
                                "mmf_deposit_pass_through_sensitivity_grid": (
                                    "|".join(
                                        f"{value:.2f}"
                                        for value in MMF_DEPOSIT_PASS_THROUGH_SENSITIVITY_GRID
                                    )
                                    if holder == "Private"
                                    else ""
                                ),
                                "bills_pct": f"{values['bills']:.12f}",
                                "notes_pct": f"{values['notes']:.12f}",
                                "bonds_pct": f"{values['bonds']:.12f}",
                                "tips_pct": f"{values['tips']:.12f}",
                                "frn_pct": f"{values['frn']:.12f}",
                                "nonmarketable_pct": "0.0",
                                "source_family": (
                                    "tdcmix_holder_absorption_prior_contract;"
                                    "tdcmix_z1_exact_recent_absorption;"
                                    "tsyparty_primary_market_buyer_composition"
                                ),
                                "source_key": (
                                    f"{tdcmix_prior_path.name};"
                                    f"{z1_absorption_path.name};"
                                    f"{bills_primary_path.name};"
                                    f"{coupons_primary_path.name}"
                                ),
                                "source_status": (
                                    "instrument_specific_holder_absorption_prior_not_holder_allocation_evidence;"
                                    f"{metadata['dealer_bridge_mapping']}"
                                    f";primary_market_overlay_{metadata['primary_market_overlay_coverage']}"
                                    + (
                                        ";mmf_split_from_du_private_subbucket_explicit"
                                        if holder == "Private"
                                        else ""
                                    )
                                ),
                                "mmf_collapsed_into_du": "false",
                                "central_default_eligible": (
                                    "true" if scenario_id == "current_mix_baseline" else "false"
                                ),
                                "sensitivity_only": (
                                    "false" if scenario_id == "current_mix_baseline" else "true"
                                ),
                                "blocked_unmapped_share": "0.0",
                                "claim_boundary": RATEWALL_CLAIM_BOUNDARY,
                            }
                        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame, {
        "tdcmix_prior_path": str(tdcmix_prior_path),
        "z1_absorption_path": str(z1_absorption_path),
        "bills_primary_path": str(bills_primary_path),
        "coupons_primary_path": str(coupons_primary_path),
        "rows": len(frame),
        **metadata,
    }


def build_interest_component_contract(output_path: Path) -> tuple[pd.DataFrame, dict]:
    rows = [
        ("coupon_interest", "true", "true", "false", "Fixed coupon and non-TIPS coupon cash paid to DU."),
        ("bill_discount_interest", "true", "true", "false", "Treasury bill discount separated from principal return."),
        ("frn_reset_interest", "true", "true", "false", "FRN coupon cash from benchmark reset plus spread."),
        ("tips_coupon_interest", "true", "true", "false", "TIPS coupon cash only."),
        ("tips_inflation_compensation", "false", "false", "true", "TIPS inflation compensation kept separate from default overlap."),
        ("principal_return", "false", "false", "false", "Principal redemption is TDC support, not direct-interest overlap."),
    ]
    frame = pd.DataFrame(
        [
            {
                "cash_component_key": key,
                "enters_tdc_interest_to_du": enters_tdc,
                "enters_direct_interest_overlap_default": overlap,
                "separate_review_required": review,
                "component_note": note,
                "source_family": "tdcsim_ratewall_interest_component_contract",
                "source_status": "contract_boundary_not_empirical_estimate",
                "claim_boundary": RATEWALL_CLAIM_BOUNDARY,
            }
            for key, enters_tdc, overlap, review, note in rows
        ]
    )
    frame.to_csv(output_path, index=False)
    return frame, {"rows": len(frame)}


def build_ratewall_config(
    output_path: Path,
    *,
    input_dir: Path,
    contract_output_dir: Path,
    holder_absorption_calibration: dict[str, float | list[float]] | None = None,
) -> None:
    calibration = holder_absorption_calibration or _default_holder_absorption_calibration()
    base_curve = {
        "use_static": True,
        "years": TENORS,
        "rates": [0.04, 0.04, 0.04, 0.04, 0.042, 0.043, 0.045, 0.046],
    }
    issuance_profile = {
        "bills": {"category_cutoff_years": 1.0, "target_percentage_of_remainder": 0.22, "maturities": [0.25, 0.5, 1.0], "maturity_distribution": [0.40, 0.35, 0.25]},
        "notes": {"category_cutoff_years": 10.0, "target_percentage_of_remainder": 0.58, "maturities": [2.0, 5.0, 10.0], "maturity_distribution": [0.35, 0.35, 0.30]},
        "bonds": {"category_cutoff_years": 999.0, "target_percentage_of_remainder": 0.20, "maturities": [20.0, 30.0], "maturity_distribution": [0.45, 0.55]},
        "TIPS": {"target_percentage": 0.06, "maturities": [5.0, 10.0, 30.0], "maturity_distribution": [0.40, 0.35, 0.25]},
        "FRN": {"target_percentage": 0.04, "maturities": [2.0], "maturity_distribution": [1.0]},
        "remainder_maturity_years": 0.5,
    }
    base_prefs = {
        holder: {
            f"{category}_pct": (1.0 / len(TDCSIM_HOLDERS))
            for category in MARKETABLE_PREFERENCE_CATEGORIES
        }
        for holder in TDCSIM_HOLDERS
    }
    scenario_rows = [
        {
            "name": scenario_id,
            "overrides": {
                "yield_curve_surface": {
                    "file": str(
                        (input_dir / "tdcsim_yield_curve_surface.csv").relative_to(
                            output_path.parent
                        )
                    ),
                    "scenario_id": curve_id,
                }
            },
        }
        for scenario_id, curve_id in (
            (scenario.scenario_id, scenario.curve_id)
            for scenario in _ratewall_scenarios()
        )
    ]
    config = {
        "simulation_period": {"start_date": "2026-04-30", "end_date": "2036-12-31", "frequency": "W", "enable_preference_trading": False, "coarse_frequency_action": "warn"},
        "initial_values": {"reserves": 2500.0, "tdc_level": 0.0, "tga": 750.0},
        "initial_portfolio": {"mode": "file", "file": str((input_dir / "tdcsim_initial_portfolio_cohorts.csv").relative_to(output_path.parent))},
        "tga_params": {"target_balance": 750.0, "floor": 100.0},
        "fiscal_params": {"initial_weekly_spending": 0.0, "initial_weekly_taxes": 0.0, "spending_growth_qtr": 0.0, "tax_growth_qtr": 0.0},
        "treasury_issuance_profile": issuance_profile,
        "yield_curve": base_curve,
        "yield_curve_surface": {"file": str((input_dir / "tdcsim_yield_curve_surface.csv").relative_to(output_path.parent)), "scenario_id": "cbo_shape_preserving_baseline"},
        "sector_preferences": base_prefs,
        "auction_absorption_preferences": base_prefs,
        "secondary_target_preferences": base_prefs,
        "private_mmf_split": {
            "enabled": True,
            "mmf_deposit_pass_through": MMF_DEPOSIT_PASS_THROUGH_DEFAULT,
            "mmf_deposit_pass_through_status": MMF_DEPOSIT_PASS_THROUGH_STATUS,
            "mmf_ru_plumbing_pass_through": 1.0 - MMF_DEPOSIT_PASS_THROUGH_DEFAULT,
            "sensitivity_grid": MMF_DEPOSIT_PASS_THROUGH_SENSITIVITY_GRID,
        },
        "ratewall_input_paths": {
            "primary_flow_to_du_file": str((input_dir / "tdcsim_primary_flow_to_du_path.csv").relative_to(output_path.parent)),
            "holder_absorption_path_file": str((input_dir / "tdcsim_holder_absorption_path.csv").relative_to(output_path.parent)),
        },
        "holder_absorption_calibration": calibration,
        "tips_params": {"cpi_start_level": 100.0, "cpi_annual_inflation": 0.02, "ref_cpi_lag_months": 3, "default_real_coupon_rate": 0.005},
        "frn_params": {"benchmark_maturity_years": 0.25, "default_fixed_spread": 0.0005},
        "nonmarketable_params": {"interest_rate_basis_maturities": [5.0, 10.0], "interest_crediting_frequency": "semi-annual", "initial_holder": "TrustFunds", "rate_setting_method": "yield_curve_points"},
        "other_flows": {"reserve_transfer": 0.0, "cb_net_expense": 0.0, "money_minting_transfers": 0.0},
        "rate_sensitive_demand": {"enabled": False, "min_multiplier": 0.25, "anchor_maturity_years": 5.0, "slope_short_maturity_years": 2.0, "slope_long_maturity_years": 10.0, "auction": {}, "secondary": {}},
        "financing_cost_options": {"include_tips_inflation_accretion": False},
        "ratewall_contract": {
            "enabled": True,
            "output_dir": str(contract_output_dir.relative_to(output_path.parent)),
            "input_manifest": str((input_dir / "tdcsim_ratewall_input_manifest.json").relative_to(output_path.parent)),
            "private_route_sensitivity_file": "../tdcest/data/processed/tdc_tdcsim_private_route_allocation_sensitivity.csv",
        },
        "parallel_execution": {
            "group_workers": "auto",
            "note": "Override with TDCSIM_GROUP_WORKERS for independent scenario groups.",
        },
        "scenario_parallel_workers": "auto",
        "scenario_groups": [
            {
                "group_name": "RateWall Source Backed TDC Forward",
                "group_output_settings": {"save_results_csv": True, "save_results_filename": "output/ratewall_source_backed_tdc", "output_mode": "single_file", "graphs_to_show": []},
                "scenarios": scenario_rows,
            }
        ],
    }
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def build_all(
    *,
    output_dir: Path,
    config_path: Path,
    ratewall_root: Path,
    project_root: Path,
    auctions_query: Path,
) -> dict:
    _ensure_dir(output_dir)
    contract_output_dir = project_root / "output" / "ratewall_contract_source_backed"
    tdcmix_prior_path = ratewall_root.parent / "tdcmix" / "data" / "processed" / "holder_absorption_prior_contract.csv"
    z1_absorption_path = ratewall_root.parent / "tdcmix" / "data" / "processed" / "z1_exact_holder_absorption_panel.csv"
    bills_primary_path = ratewall_root.parent / "tsyparty" / "data" / "interim" / "bills_quarterly_composition.csv"
    coupons_primary_path = ratewall_root.parent / "tsyparty" / "data" / "interim" / "nominal_coupons_quarterly_composition.csv"
    cbo_budget_path = ratewall_root / "data" / "raw" / "cbo" / "51118-2026-02-Budget-Projections.xlsx"
    cbo_economic_path = ratewall_root / "data" / "raw" / "cbo" / "51135-2026-02-Economic-Projections.xlsx"
    holder_absorption_calibration = _load_holder_absorption_calibration(config_path)

    artifacts = {}
    record_date = _latest_record_date("v1/debt/mspd/mspd_table_1")
    soma_path = output_dir / SOMA_HOLDINGS_FILENAME
    soma_url = fetch_soma_treasury_holdings(record_date, soma_path)
    # The opening book prices its non-Fed securities on this surface.
    _, artifacts["yield_curve"] = build_yield_curve_surface(
        output_dir / YIELD_CURVE_SURFACE_FILENAME,
        cbo_economic_path=cbo_economic_path,
    )
    _, artifacts["initial_portfolio"] = build_initial_portfolio(
        output_dir / "tdcsim_initial_portfolio_cohorts.csv",
        record_date=record_date,
        table1_rows=_fetch_fiscaldata_rows("v1/debt/mspd/mspd_table_1", record_date=record_date),
        table3_rows=_fetch_fiscaldata_rows("v1/debt/mspd/mspd_table_3_market", record_date=record_date),
        soma_path=soma_path,
        wamest_path=ratewall_root.parent / WAMEST_CANONICAL_RELATIVE_PATH,
        yield_curve_path=output_dir / YIELD_CURVE_SURFACE_FILENAME,
        auctions_path=auctions_query,
        nmfp_holdings_path=output_dir / NMFP_HOLDINGS_FILENAME,
    )
    artifacts["initial_portfolio"]["soma_source_url"] = soma_url
    _, artifacts["primary_flow"] = build_primary_flow_path(
        output_dir / "tdcsim_primary_flow_to_du_path.csv",
        cbo_budget_path=cbo_budget_path,
    )
    _, artifacts["holder_absorption"] = build_holder_absorption_path(
        output_dir / "tdcsim_holder_absorption_path.csv",
        tdcmix_prior_path=tdcmix_prior_path,
        z1_absorption_path=z1_absorption_path,
        bills_primary_path=bills_primary_path,
        coupons_primary_path=coupons_primary_path,
        holder_absorption_calibration=holder_absorption_calibration,
    )
    _, artifacts["interest_component"] = build_interest_component_contract(
        output_dir / "tdcsim_interest_component_contract.csv"
    )
    build_ratewall_config(
        config_path,
        input_dir=output_dir,
        contract_output_dir=contract_output_dir,
        holder_absorption_calibration=holder_absorption_calibration,
    )

    files = sorted(output_dir.glob("*.csv")) + [(output_dir / NMFP_HOLDINGS_FILENAME).with_suffix(".provenance.json"), config_path]
    manifest = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": RATEWALL_CLAIM_BOUNDARY,
        "artifacts": artifacts,
        "files": {path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size} for path in files},
        "source_hierarchy": {
            "bond_history": BOND_HISTORY_SOURCE,
            "yield_curve": "Treasury daily par curve shape shifted to CBO 3-month and 10-year projected anchors",
            "primary_flow": "CBO total deficit less net interest as aggregate cash proxy",
            "holder_absorption": (
                "instrument-specific holder absorption priors from tdcmix, "
                "recent z1 exact absorption, and tsyparty primary-market "
                "buyer composition; not holder-allocation evidence"
            ),
        },
    }
    manifest_path = output_dir / "tdcsim_ratewall_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def rebuild_opening_book(
    output_dir: Path,
    *,
    record_date: str,
    mspd_table_1: Path,
    mspd_table_3_market: Path,
    soma_holdings: Path,
    wamest_canonical: Path,
    auctions_query: Path,
) -> dict:
    """Rebuild only the opening book in an existing input directory from local source files.

    The SOMA file is copied beside the book, and the manifest's opening-book entry and
    file hashes are replaced; the other artifacts in the directory are left as they are.
    The book is priced on the directory's yield curve surface, which must be the one the
    manifest records, and reads the directory's N-MFP money-fund holdings
    (``NMFP_HOLDINGS_FILENAME``), whose hashes the manifest then records.
    """

    manifest_path = output_dir / "tdcsim_ratewall_input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    curve_path = output_dir / YIELD_CURVE_SURFACE_FILENAME
    recorded_curve = manifest.get("files", {}).get(YIELD_CURVE_SURFACE_FILENAME, {}).get("sha256")
    if not curve_path.exists() or recorded_curve != _sha256(curve_path):
        raise ValueError(f"{curve_path.name} is missing or does not match the manifest's recorded hash")
    tables = {}
    for name, path in (("table1", mspd_table_1), ("table3", mspd_table_3_market)):
        frame = pd.read_csv(path, dtype=str)
        rows = frame[frame["record_date"] == record_date].fillna("").to_dict("records")
        if not rows:
            raise ValueError(f"{path.name} has no rows for record date {record_date}")
        tables[name] = rows
    soma_path = output_dir / SOMA_HOLDINGS_FILENAME
    if soma_holdings.resolve() != soma_path.resolve():
        soma_path.write_bytes(soma_holdings.read_bytes())
    book_path = output_dir / "tdcsim_initial_portfolio_cohorts.csv"
    nmfp_path = output_dir / NMFP_HOLDINGS_FILENAME
    _, metadata = build_initial_portfolio(
        book_path,
        record_date=record_date,
        table1_rows=tables["table1"],
        table3_rows=tables["table3"],
        soma_path=soma_path,
        wamest_path=wamest_canonical,
        yield_curve_path=curve_path,
        auctions_path=auctions_query,
        nmfp_holdings_path=nmfp_path,
    )
    metadata["mspd_source_sha256"] = {path.name: _sha256(path) for path in (mspd_table_1, mspd_table_3_market)}
    manifest["artifacts"]["initial_portfolio"] = metadata
    for path in (book_path, soma_path, nmfp_path, nmfp_path.with_suffix(".provenance.json")):
        manifest["files"][path.name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    manifest["source_hierarchy"]["bond_history"] = BOND_HISTORY_SOURCE
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source-backed tdcsim inputs for RateWall.")
    parser.add_argument("--output-dir", default="data/ratewall_inputs")
    parser.add_argument("--config-path", default="tdc_config_ratewall_source_backed.yaml")
    parser.add_argument("--ratewall-root", default="../ratewall")
    parser.add_argument(
        "--auctions-query",
        type=Path,
        required=True,
        help="FiscalData auctions file; the Fed's add-ons settling by the record date go to CB",
    )
    opening = parser.add_argument_group("rebuild only the opening book from local files")
    opening.add_argument("--opening-book-only", action="store_true")
    opening.add_argument("--record-date")
    opening.add_argument("--mspd-table-1", type=Path)
    opening.add_argument("--mspd-table-3-market", type=Path)
    opening.add_argument("--soma-holdings", type=Path)
    opening.add_argument("--wamest-canonical", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.opening_book_only:
        required = ("record_date", "mspd_table_1", "mspd_table_3_market", "soma_holdings")
        missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"--opening-book-only requires {', '.join(missing)}")
        manifest = rebuild_opening_book(
            (project_root / args.output_dir).resolve(),
            record_date=args.record_date,
            mspd_table_1=args.mspd_table_1,
            mspd_table_3_market=args.mspd_table_3_market,
            soma_holdings=args.soma_holdings,
            wamest_canonical=args.wamest_canonical
            or (project_root / args.ratewall_root).resolve().parent / WAMEST_CANONICAL_RELATIVE_PATH,
            auctions_query=args.auctions_query,
        )
        print(json.dumps({"status": "ok", "initial_portfolio": manifest["artifacts"]["initial_portfolio"]}, indent=2, sort_keys=True))
        return
    manifest = build_all(
        output_dir=(project_root / args.output_dir).resolve(),
        config_path=(project_root / args.config_path).resolve(),
        ratewall_root=(project_root / args.ratewall_root).resolve(),
        project_root=project_root,
        auctions_query=args.auctions_query,
    )
    print(json.dumps({"status": "ok", "files": manifest["files"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
