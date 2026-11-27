"""Settlement-route registry and Treasury component of reserves (TRC) ledger.

Reserve incidence follows the claim debited or credited at settlement, never a
holder's legal sector. Each holder maps to settlement routes through one routing
policy; each route names the claim, whether that claim is a covered nonbank
deposit (TDC membership) and whether its settlement moves aggregate reserve
balances (TRC membership).

Cash moving between a reserve-settled claim and a Federal Reserve nonreserve
account (the TGA, SOMA, ON RRP, foreign official deposits or the foreign repo
pool) changes reserves; cash moving between two reserve-settled claims, or
between two nonreserve accounts, does not. TRC is therefore the sum of each
event's signed holder cash multiplied by the reserve membership of the holder's
settlement route. The legacy holder-based reserve path treats every Bank,
Private and Foreign flow as reserve-settled, so for every period

    legacy reserve change = TRC change + change in route-attributed Fed
                            nonreserve balances,

which the ledger enforces at each daily close.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from tdc_shared import (
    HOLDER_TYPES,
    INTRAGOV_HOLDERS,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
    PRIVATE_SUBBUCKETS,
)

TRC_ROUTING_SCHEMA_VERSION = "tdcsim_trc_routing_v1"
TRC_RESULTS_SCHEMA_VERSION = "tdcsim_trc_results_v1"
TRC_CLOSURE_TOLERANCE_BIL = 1.0e-8
TRC_STATE_TOLERANCE_BIL = 1.0e-9

# Flows recorded by the ledger, in output order. FiscalFlow covers primary
# spending and taxes; the others follow the engine's settlement legs.
TRC_FAMILIES = (
    "FiscalFlow",
    "DebtService",
    "Buyback",
    "AuctionSettlement",
    "FedSecondary",
    "SecondaryTrades",
    "TTandL",
    "OtherTreasuryFed",
)

TRC_FAMILY_SLUGS = MappingProxyType(
    {
        "FiscalFlow": "fiscal_flow",
        "DebtService": "debt_service",
        "Buyback": "buyback",
        "AuctionSettlement": "auction_settlement",
        "FedSecondary": "fed_secondary",
        "SecondaryTrades": "secondary_trades",
        "TTandL": "tt_and_l",
        "OtherTreasuryFed": "other_treasury_fed",
    }
)
if set(TRC_FAMILY_SLUGS) != set(TRC_FAMILIES):
    raise RuntimeError("TRC_FAMILY_SLUGS must name every TRC family exactly once")

# Federal Reserve nonreserve balances that a route can change. The keys are the
# output column names.
TRC_ROUTE_STATES = (
    "ONRRP_MMF",
    "ForeignOfficialFedDeposits",
    "ForeignOfficialFedRRP",
)


class TrcRoutingError(ValueError):
    """The TRC routing policy or ledger closure is invalid."""


@dataclass(frozen=True)
class SettlementRoute:
    route_id: str
    settlement_claim: str
    fed_account: str
    covered_deposit_member: float
    reserve_settled_member: float
    route_state: str | None
    modeled_in_tdcsim: bool
    description: str


SETTLEMENT_ROUTES: Mapping[str, SettlementRoute] = MappingProxyType(
    {
        route.route_id: route
        for route in (
            SettlementRoute(
                "covered_deposit_settled",
                "covered_domestic_nonbank_deposit",
                "reserve_balances_via_issuing_bank",
                1.0,
                1.0,
                None,
                True,
                "Domestic nonbank pays or receives through a covered bank deposit.",
            ),
            SettlementRoute(
                "bank_reserve_settled",
                "depository_institution_own_reserve_balance",
                "reserve_balances",
                0.0,
                1.0,
                None,
                True,
                "A depository institution settles on its own account.",
            ),
            SettlementRoute(
                "foreign_bank_settled",
                "us_bank_correspondent_balance",
                "reserve_balances_via_correspondent_bank",
                0.0,
                1.0,
                None,
                True,
                "Foreign holder settles through a US bank outside covered deposits.",
            ),
            SettlementRoute(
                "on_rrp_mmf",
                "fed_on_rrp_money_fund_claim",
                "on_rrp",
                0.0,
                0.0,
                "ONRRP_MMF",
                True,
                "Money fund funds or places cash through its ON RRP claim.",
            ),
            SettlementRoute(
                "on_rrp_gse",
                "fed_on_rrp_gse_claim",
                "on_rrp",
                0.0,
                0.0,
                None,
                False,
                "GSE funds or places cash through ON RRP; no GSE holder is modeled.",
            ),
            SettlementRoute(
                "foreign_official_deposit",
                "fed_foreign_official_deposit",
                "foreign_official_deposits",
                0.0,
                0.0,
                "ForeignOfficialFedDeposits",
                True,
                "Foreign official holder settles from its Federal Reserve deposit.",
            ),
            SettlementRoute(
                "foreign_official_rrp",
                "fed_foreign_repo_pool_claim",
                "foreign_repo_pool",
                0.0,
                0.0,
                "ForeignOfficialFedRRP",
                True,
                "Foreign official holder settles from the foreign repo pool.",
            ),
            SettlementRoute(
                "fed_soma",
                "federal_reserve_own_account",
                "soma",
                0.0,
                0.0,
                None,
                True,
                "Federal Reserve on its own account; add-ons and redemptions net against the TGA.",
            ),
            SettlementRoute(
                "tt_and_l",
                "treasury_tax_and_loan_balance_at_commercial_bank",
                "reserve_balances_via_depository_bank",
                0.0,
                1.0,
                None,
                False,
                "Treasury cash at a commercial bank; TDCSim's TGA proxy does not model it.",
            ),
            SettlementRoute(
                "intragovernmental",
                "treasury_internal_account",
                "tga",
                0.0,
                0.0,
                None,
                True,
                "Trust funds and other intragovernmental holders; payments are TGA washes.",
            ),
            SettlementRoute(
                "other_named_fed_account",
                "other_named_federal_reserve_nonreserve_account",
                "other_fed_nonreserve",
                0.0,
                0.0,
                None,
                False,
                "Any other named Fed nonreserve account; admitted only with a matched event.",
            ),
        )
    }
)

FOREIGN_SETTLEMENT_ROUTES = (
    "foreign_bank_settled",
    "foreign_official_deposit",
    "foreign_official_rrp",
)
MMF_ROUTE_KEYS = {"on_rrp_share", "source"}
_FIXED_HOLDER_ROUTES = MappingProxyType(
    {
        "Banks": (("bank_reserve_settled", 1.0),),
        "CB": (("fed_soma", 1.0),),
        **{holder: (("intragovernmental", 1.0),) for holder in sorted(INTRAGOV_HOLDERS)},
    }
)
_REQUIRED_KEYS = {
    "schema_version",
    "foreign_settlement_mix",
    "mmf_cash_fund_route",
    "opening_fed_nonreserve_balances_bil",
    "opening_balance_sources",
}
_OPTIONAL_KEYS = {"description", "assumption_status"}


def _finite_share(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TrcRoutingError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise TrcRoutingError(f"{label} must lie in [0, 1]")
    return number


@dataclass(frozen=True)
class TrcRoutingPolicy:
    """Holder-to-route shares and opening Fed nonreserve balances.

    Ordinary money-fund cash settles through covered deposits. An explicit
    money-fund ON RRP share can replace part or all of that route. The two
    declared shares account for all money-fund settlement.
    """

    foreign_mix: Mapping[str, float]
    mmf_on_rrp_share: float
    mmf_on_rrp_source: str
    opening_balances: Mapping[str, float]
    opening_balance_sources: Mapping[str, str]
    mmf_deposit_pass_through: float
    assumption_status: str
    description: str

    def routes_for(self, holder: str, private_route: str | None = None) -> tuple[tuple[str, float], ...]:
        if holder in _FIXED_HOLDER_ROUTES:
            return _FIXED_HOLDER_ROUTES[holder]
        if holder == "Foreign":
            return tuple(
                (route_id, share)
                for route_id, share in self.foreign_mix.items()
                if share > 0.0
            )
        if holder == "Private":
            if private_route == PRIVATE_SUBBUCKET_MMF:
                routes = (
                    ("covered_deposit_settled", self.mmf_deposit_pass_through),
                    ("on_rrp_mmf", self.mmf_on_rrp_share),
                )
                return tuple((route_id, share) for route_id, share in routes if share > 0.0)
            if private_route in (PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, None):
                return (("covered_deposit_settled", 1.0),)
            raise TrcRoutingError(f"unknown Private settlement sub-route: {private_route!r}")
        raise TrcRoutingError(f"holder has no settlement-route policy: {holder!r}")

    def reserve_share(self, holder: str, private_route: str | None = None) -> float:
        return sum(
            share * SETTLEMENT_ROUTES[route_id].reserve_settled_member
            for route_id, share in self.routes_for(holder, private_route)
        )

    def covered_deposit_share(self, holder: str, private_route: str | None = None) -> float:
        return sum(
            share * SETTLEMENT_ROUTES[route_id].covered_deposit_member
            for route_id, share in self.routes_for(holder, private_route)
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": TRC_ROUTING_SCHEMA_VERSION,
            "foreign_settlement_mix": dict(self.foreign_mix),
            "mmf_cash_fund_route": {
                "on_rrp_share": self.mmf_on_rrp_share,
                "source": self.mmf_on_rrp_source,
                "covered_deposit_share_from_pass_through": self.mmf_deposit_pass_through,
            },
            "opening_fed_nonreserve_balances_bil": dict(self.opening_balances),
            "opening_balance_sources": dict(self.opening_balance_sources),
            "assumption_status": self.assumption_status,
            "description": self.description,
            "fixed_holder_routes": {
                holder: [list(item) for item in routes]
                for holder, routes in _FIXED_HOLDER_ROUTES.items()
            },
        }


def parse_trc_routing(config: Mapping[str, Any], *, mmf_deposit_pass_through: float) -> TrcRoutingPolicy:
    """Validate a ``trc_routing`` block; every share and opening balance is explicit."""

    if not isinstance(config, Mapping):
        raise TrcRoutingError("trc_routing must be a mapping")
    keys = set(config)
    missing = _REQUIRED_KEYS - keys
    unknown = keys - _REQUIRED_KEYS - _OPTIONAL_KEYS
    if missing or unknown:
        raise TrcRoutingError(
            f"trc_routing keys differ: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if config["schema_version"] != TRC_ROUTING_SCHEMA_VERSION:
        raise TrcRoutingError(f"trc_routing schema_version must be {TRC_ROUTING_SCHEMA_VERSION}")

    mix_raw = config["foreign_settlement_mix"]
    if not isinstance(mix_raw, Mapping) or set(mix_raw) != set(FOREIGN_SETTLEMENT_ROUTES):
        raise TrcRoutingError(
            f"foreign_settlement_mix must name exactly {list(FOREIGN_SETTLEMENT_ROUTES)}"
        )
    foreign_mix = {
        route_id: _finite_share(mix_raw[route_id], f"foreign_settlement_mix.{route_id}")
        for route_id in FOREIGN_SETTLEMENT_ROUTES
    }
    if abs(sum(foreign_mix.values()) - 1.0) > 1.0e-12:
        raise TrcRoutingError("foreign_settlement_mix shares must sum to one")

    mmf_cfg = config["mmf_cash_fund_route"]
    if not isinstance(mmf_cfg, Mapping) or set(mmf_cfg) != MMF_ROUTE_KEYS:
        raise TrcRoutingError(f"mmf_cash_fund_route must name exactly {sorted(MMF_ROUTE_KEYS)}")
    pass_through = _finite_share(mmf_deposit_pass_through, "mmf_deposit_pass_through")
    mmf_on_rrp_share = _finite_share(mmf_cfg["on_rrp_share"], "mmf_cash_fund_route.on_rrp_share")
    mmf_on_rrp_source = str(mmf_cfg["source"] or "").strip()
    if not mmf_on_rrp_source:
        raise TrcRoutingError("mmf_cash_fund_route.on_rrp_share requires a named source")
    if pass_through + mmf_on_rrp_share != 1.0:
        raise TrcRoutingError(
            "money-fund covered-deposit and ON RRP shares must sum to one"
        )

    balances_raw = config["opening_fed_nonreserve_balances_bil"]
    sources_raw = config["opening_balance_sources"]
    if not isinstance(balances_raw, Mapping) or set(balances_raw) != set(TRC_ROUTE_STATES):
        raise TrcRoutingError(
            f"opening_fed_nonreserve_balances_bil must name exactly {list(TRC_ROUTE_STATES)}"
        )
    if not isinstance(sources_raw, Mapping) or set(sources_raw) != set(TRC_ROUTE_STATES):
        raise TrcRoutingError(
            f"opening_balance_sources must name exactly {list(TRC_ROUTE_STATES)}"
        )
    balances: dict[str, float] = {}
    for state in TRC_ROUTE_STATES:
        try:
            value = float(balances_raw[state])
        except (TypeError, ValueError) as exc:
            raise TrcRoutingError(f"opening balance {state} must be numeric") from exc
        if not math.isfinite(value) or value < 0.0:
            raise TrcRoutingError(f"opening balance {state} must be finite and nonnegative")
        source = str(sources_raw[state] or "").strip()
        if not source:
            raise TrcRoutingError(f"opening balance {state} requires a named source")
        balances[state] = value

    return TrcRoutingPolicy(
        foreign_mix=MappingProxyType(foreign_mix),
        mmf_on_rrp_share=mmf_on_rrp_share,
        mmf_on_rrp_source=mmf_on_rrp_source,
        opening_balances=MappingProxyType(balances),
        opening_balance_sources=MappingProxyType({k: str(sources_raw[k]) for k in TRC_ROUTE_STATES}),
        mmf_deposit_pass_through=pass_through,
        assumption_status=str(config.get("assumption_status") or "scenario_assumption"),
        description=str(config.get("description") or ""),
    )


def route_registry_rows() -> list[dict[str, Any]]:
    return [
        {
            "route_id": route.route_id,
            "settlement_claim": route.settlement_claim,
            "fed_account": route.fed_account,
            "covered_deposit_member": route.covered_deposit_member,
            "reserve_settled_member": route.reserve_settled_member,
            "route_state": route.route_state or "",
            "modeled_in_tdcsim": route.modeled_in_tdcsim,
            "description": route.description,
        }
        for route in SETTLEMENT_ROUTES.values()
    ]


class TrcLedger:
    """Accumulate route-aware reserve incidence and Fed nonreserve route balances."""

    def __init__(self, policy: TrcRoutingPolicy, *, opening_reserves: float) -> None:
        self.policy = policy
        self.opening_reserves = float(opening_reserves)
        self.balances = {state: float(policy.opening_balances[state]) for state in TRC_ROUTE_STATES}
        self.cumulative = 0.0
        self.legacy_cumulative = 0.0
        self.unrouted_private_abs = 0.0
        self.rows: list[dict[str, Any]] = []
        self._reset_period()

    def _reset_period(self) -> None:
        self._period = {family: 0.0 for family in TRC_FAMILIES}
        self._state_delta = {state: 0.0 for state in TRC_ROUTE_STATES}
        self._foreign_gross = 0.0

    def _book(self, family: str, holder: str, private_route: str | None, amount: float) -> None:
        if family not in self._period:
            raise TrcRoutingError(f"unknown TRC family: {family}")
        if amount == 0.0:
            return
        if not math.isfinite(amount):
            raise TrcRoutingError(f"nonfinite TRC cash amount for {holder} in {family}")
        if holder == "Foreign":
            self._foreign_gross += amount
        for route_id, share in self.policy.routes_for(holder, private_route):
            route = SETTLEMENT_ROUTES[route_id]
            part = amount * share
            self._period[family] += part * route.reserve_settled_member
            if route.route_state is not None:
                self._state_delta[route.route_state] += part

    def record_holder_cash(
        self,
        family: str,
        holder_cash: Mapping[str, float],
        private_route_cash: Mapping[str, float] | None = None,
    ) -> None:
        """Book cash credited to (+) or debited from (-) holders by a Fed-side account."""

        for holder in HOLDER_TYPES:
            amount = float(holder_cash.get(holder, 0.0) or 0.0)
            if holder != "Private":
                self._book(family, holder, None, amount)
                continue
            routed = 0.0
            for route in PRIVATE_SUBBUCKETS:
                route_amount = float((private_route_cash or {}).get(route, 0.0) or 0.0)
                routed += route_amount
                self._book(family, "Private", route, route_amount)
            unrouted = amount - routed
            if abs(unrouted) > TRC_STATE_TOLERANCE_BIL * max(1.0, abs(amount)):
                # Legacy portfolios without a Private sub-bucket settle through
                # covered deposits, matching the holder-based reserve path.
                self.unrouted_private_abs += abs(unrouted)
                self._book(family, "Private", None, unrouted)
        extra = set(holder_cash) - set(HOLDER_TYPES)
        if extra:
            raise TrcRoutingError(f"unknown holders in TRC cash: {sorted(extra)}")

    def record_public_cash(self, family: str, amount: float, route_id: str) -> None:
        """Book a non-holder Treasury cash flow settled through one named route."""

        route = SETTLEMENT_ROUTES[route_id]
        if route.route_state is not None:
            raise TrcRoutingError("public cash flows cannot change a Fed nonreserve route state")
        self._period[family] += float(amount) * route.reserve_settled_member

    def record_trade(
        self,
        family: str,
        *,
        buyer: str,
        buyer_private_route: str | None,
        seller: str,
        seller_private_route: str | None,
        amount: float,
    ) -> None:
        """Book a holder-to-holder Treasury-security settlement: buyer pays seller."""

        self._book(family, seller, seller_private_route if seller == "Private" else None, float(amount))
        self._book(family, buyer, buyer_private_route if buyer == "Private" else None, -float(amount))

    def close_period(self, date: Any, *, legacy_reserve_change: float, legacy_reserves: float) -> dict[str, Any]:
        trc_change = sum(self._period.values())
        state_change = sum(self._state_delta.values())
        legacy = float(legacy_reserve_change)
        scale = max(1.0, abs(legacy), abs(trc_change), abs(state_change))
        gap = legacy - trc_change - state_change
        if abs(gap) > TRC_CLOSURE_TOLERANCE_BIL * scale:
            raise TrcRoutingError(
                f"TRC closure failed at {date}: legacy reserve change {legacy!r} "
                f"!= TRC {trc_change!r} + route state change {state_change!r}"
            )
        for state in TRC_ROUTE_STATES:
            self.balances[state] += self._state_delta[state]
            if self.balances[state] < -TRC_STATE_TOLERANCE_BIL:
                raise TrcRoutingError(
                    f"Fed nonreserve route balance {state} is negative at the {date} close: "
                    f"{self.balances[state]!r}; the routed settlement exceeds available capacity"
                )
        self.cumulative += trc_change
        self.legacy_cumulative += legacy
        synthetic = self.opening_reserves + self.cumulative
        route_balance_change = sum(
            self.balances[state] - self.policy.opening_balances[state] for state in TRC_ROUTE_STATES
        )
        alias_gap = float(legacy_reserves) - (synthetic + route_balance_change)
        if abs(alias_gap) > TRC_CLOSURE_TOLERANCE_BIL * max(1.0, abs(float(legacy_reserves)), len(self.rows) + 1.0):
            raise TrcRoutingError(
                f"legacy Reserves alias does not reconcile at {date}: gap {alias_gap!r}"
            )
        row = {"Date": date}
        row.update({f"TRC_{family}": value for family, value in self._period.items()})
        row.update(
            {
                "TRC_Change": trc_change,
                "TRC_Cumulative": self.cumulative,
                "TRC_RouteStateChange": state_change,
                "TRC_ForeignGrossSettlementCash": self._foreign_gross,
                "LegacyReserveChange_HolderRouted": legacy,
                "SyntheticReserveBalance_ExNonTreasuryFactors": synthetic,
                "TRC_LegacyClosureGap": gap,
                "TRC_LegacyAliasGap": alias_gap,
            }
        )
        row.update(dict(self.balances))
        self.rows.append(row)
        self._reset_period()
        return row

    def opening_row(self, date: Any) -> dict[str, Any]:
        row = {"Date": date}
        row.update({f"TRC_{family}": 0.0 for family in TRC_FAMILIES})
        row.update(
            {
                "TRC_Change": 0.0,
                "TRC_Cumulative": 0.0,
                "TRC_RouteStateChange": 0.0,
                "TRC_ForeignGrossSettlementCash": 0.0,
                "LegacyReserveChange_HolderRouted": 0.0,
                "SyntheticReserveBalance_ExNonTreasuryFactors": self.opening_reserves,
                "TRC_LegacyClosureGap": 0.0,
                "TRC_LegacyAliasGap": 0.0,
            }
        )
        row.update(dict(self.balances))
        return row


TRC_RESULT_COLUMNS = (
    "Date",
    *(f"TRC_{family}" for family in TRC_FAMILIES),
    "TRC_Change",
    "TRC_Cumulative",
    "TRC_RouteStateChange",
    "TRC_ForeignGrossSettlementCash",
    "LegacyReserveChange_HolderRouted",
    "SyntheticReserveBalance_ExNonTreasuryFactors",
    "TRC_LegacyClosureGap",
    "TRC_LegacyAliasGap",
    *TRC_ROUTE_STATES,
)


def trc_metadata(policy: TrcRoutingPolicy, ledger: TrcLedger) -> dict[str, Any]:
    return {
        "schema_version": TRC_RESULTS_SCHEMA_VERSION,
        "object": "treasury_component_of_reserves",
        "perimeter": "aggregate_reserve_balances_at_federal_reserve_banks",
        "units": "USD billions",
        "routing": policy.as_record(),
        "route_registry": route_registry_rows(),
        "unrouted_private_cash_abs_bil": ledger.unrouted_private_abs,
        "closure_identity": (
            "LegacyReserveChange_HolderRouted = TRC_Change + TRC_RouteStateChange; "
            "Reserves = SyntheticReserveBalance_ExNonTreasuryFactors + "
            "sum(route balance - opening route balance)"
        ),
        "structural_zeros": {
            "TRC_TTandL": "the TDCSim TGA proxies Treasury operating cash; TT&L is not modeled",
            "on_rrp_gse": "no GSE holder is modeled",
        },
        "legacy_reserves_alias": (
            "Reserves is a deprecated compatibility alias: opening reserves plus the "
            "cumulative holder-routed Treasury reserve path. It excludes every "
            "non-Treasury reserve factor and is not an observed or forecast reserve stock."
        ),
        "boundaries": [
            "Scenario accounting under stated routing assumptions; not a forecast or causal estimate",
            "Excludes Fed non-Treasury assets, currency, IORB, swaps and non-Treasury ON RRP dynamics",
        ],
    }
