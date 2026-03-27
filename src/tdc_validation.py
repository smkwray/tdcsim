"""Validation helpers for tdcsim.

These checks are intentionally strict enough to prevent silent misconfiguration,
but still lightweight enough to keep the existing simulator structure intact.
"""
from __future__ import annotations

import collections.abc
from typing import Dict, Iterable, List, Sequence, Tuple

from tdc_shared import HOLDER_TYPES, MATURITY_CATEGORIES, PREFERENCE_CATEGORIES, TGA_FLOOR_TOLERANCE

VALID_OVERRIDE_KEYS = frozenset({
    'fiscal_params', 'tga_params', 'yield_curve', 'treasury_issuance_profile',
    'sector_preferences', 'auction_absorption_preferences', 'secondary_target_preferences',
    'events', 'tips_params', 'frn_params', 'nonmarketable_params', 'other_flows',
    'simulation_period',
})

VALID_NONMARKETABLE_CREDITING_FREQUENCIES = frozenset({'semi-annual', 'annual'})

SUPPORTED_EVENT_PATH_PREFIXES: Tuple[Tuple[str, ...], ...] = (
    ('simulation_period', 'enable_preference_trading'),
    ('fiscal_params',),
    ('tga_params',),
    ('other_flows',),
    ('sector_preferences',),
    ('auction_absorption_preferences',),
    ('secondary_target_preferences',),
    ('yield_curve',),
    ('treasury_issuance_profile',),
    ('tips_params',),
    ('frn_params',),
    ('nonmarketable_params',),
)

# Valid leaf keys for each flat parameter block targeted by events.
VALID_EVENT_LEAF_KEYS: Dict[str, frozenset] = {
    'fiscal_params': frozenset({
        'initial_weekly_spending', 'initial_weekly_taxes',
        'spending_growth_qtr', 'tax_growth_qtr',
    }),
    'tga_params': frozenset({'target_balance', 'floor'}),
    'other_flows': frozenset({'reserve_transfer', 'cb_net_expense', 'money_minting_transfers'}),
    'tips_params': frozenset({
        'cpi_start_level', 'cpi_annual_inflation', 'ref_cpi_lag_months',
        'default_real_coupon_rate',
    }),
    'frn_params': frozenset({'benchmark_maturity_years', 'default_fixed_spread'}),
    'nonmarketable_params': frozenset({
        'interest_rate_basis_maturities', 'interest_crediting_frequency',
        'initial_holder', 'rate_setting_method',
        'marketable_basket_min_remaining_maturity', 'marketable_basket_types',
        'marketable_basket_weighting',
    }),
}

# Parameter blocks where paths follow holder.category_pct pattern
HOLDER_PREF_BLOCKS: frozenset = frozenset({
    'sector_preferences', 'auction_absorption_preferences', 'secondary_target_preferences',
})

# Parameter blocks with complex nested structure validated elsewhere
COMPLEX_REPLACEMENT_BLOCKS: frozenset = frozenset({
    'yield_curve', 'treasury_issuance_profile',
})

_VALID_PREF_CATEGORIES = frozenset({'bills', 'notes', 'bonds', 'tips', 'frn', 'nonmarketable'})
_HOLDER_TYPES_SET = frozenset(HOLDER_TYPES)


def is_supported_event_path(path_keys: Sequence[str]) -> bool:
    path_tuple = tuple(path_keys)
    return any(path_tuple[:len(prefix)] == prefix for prefix in SUPPORTED_EVENT_PATH_PREFIXES)


def validate_event_path(path_keys: Sequence[str]) -> Tuple[bool, str]:
    """Validate a full event parameter path. Returns (is_valid, error_message)."""
    if not path_keys:
        return False, "Empty parameter path."

    root = path_keys[0]

    # Exact match for simulation_period.enable_preference_trading
    if root == 'simulation_period':
        if tuple(path_keys) == ('simulation_period', 'enable_preference_trading'):
            return True, ""
        return False, "Only 'simulation_period.enable_preference_trading' is supported."

    # Full replacement (len == 1)
    if len(path_keys) == 1:
        if root in VALID_EVENT_LEAF_KEYS or root in HOLDER_PREF_BLOCKS or root in COMPLEX_REPLACEMENT_BLOCKS:
            return True, ""
        return False, f"Unknown parameter block '{root}'."

    # Flat leaf-key blocks
    if root in VALID_EVENT_LEAF_KEYS:
        if len(path_keys) == 2 and path_keys[1] in VALID_EVENT_LEAF_KEYS[root]:
            return True, ""
        return False, (
            f"Invalid leaf key '{'.'.join(path_keys[1:])}' for '{root}'. "
            f"Valid keys: {sorted(VALID_EVENT_LEAF_KEYS[root])}"
        )

    # Holder preference blocks: holder.category_pct
    if root in HOLDER_PREF_BLOCKS:
        if len(path_keys) == 2 and path_keys[1] in _HOLDER_TYPES_SET:
            return True, ""
        if len(path_keys) == 3:
            holder, pref_key = path_keys[1], path_keys[2]
            if holder in _HOLDER_TYPES_SET and pref_key.endswith('_pct'):
                cat = pref_key[:-4]
                if cat in _VALID_PREF_CATEGORIES:
                    return True, ""
        return False, f"Invalid path for '{root}': '{'.'.join(path_keys)}'."

    # Complex nested blocks — structural validation
    if root == 'yield_curve':
        if len(path_keys) == 1:
            return True, ""  # Full replacement
        _VALID_YC_KEYS = frozenset({'years', 'rates', 'use_static'})
        if len(path_keys) == 2 and path_keys[1] in _VALID_YC_KEYS:
            return True, ""
        return False, f"Invalid yield_curve path '{'.'.join(path_keys)}'. Valid sub-keys: {sorted(_VALID_YC_KEYS)}"

    if root == 'treasury_issuance_profile':
        if len(path_keys) == 1:
            return True, ""  # Full replacement
        _VALID_TIP_L1 = frozenset(MATURITY_CATEGORIES) | frozenset({'TIPS', 'FRN', 'NonMarketable', 'remainder_maturity_years'})
        if path_keys[1] not in _VALID_TIP_L1:
            return False, f"Invalid issuance profile category '{path_keys[1]}'"
        if len(path_keys) == 2:
            return True, ""  # Full category replacement
        _VALID_CAT_KEYS = frozenset({'category_cutoff_years', 'target_percentage_of_remainder',
                                     'maturities', 'maturity_distribution', 'target_percentage'})
        if len(path_keys) == 3 and path_keys[2] in _VALID_CAT_KEYS:
            return True, ""
        return False, f"Invalid issuance profile path '{'.'.join(path_keys)}'"

    return False, f"Unsupported parameter block '{root}'."


def _assert_nonnegative(value: float, name: str, errors: List[str]) -> None:
    try:
        if float(value) < -TGA_FLOOR_TOLERANCE:
            errors.append(f"{name} must be non-negative. Found {value}.")
    except Exception:
        errors.append(f"{name} must be numeric. Found {value!r}.")


def validate_yield_curve(yield_curve: dict, label: str = 'yield_curve') -> List[str]:
    errors: List[str] = []
    if not isinstance(yield_curve, collections.abc.Mapping):
        return [f"{label} must be a mapping."]
    years = yield_curve.get('years', [])
    rates = yield_curve.get('rates', [])
    if not isinstance(years, list) or not isinstance(rates, list):
        return [f"{label}.years and {label}.rates must be lists."]
    if len(years) == 0:
        errors.append(f"{label}.years is empty.")
    if len(years) != len(rates):
        errors.append(f"{label}.years and {label}.rates must have the same length.")
        return errors
    prev_year = None
    for idx, (year, rate) in enumerate(zip(years, rates)):
        _assert_nonnegative(year, f"{label}.years[{idx}]", errors)
        try:
            year = float(year)
            if prev_year is not None and year <= prev_year:
                errors.append(f"{label}.years must be strictly increasing. Found {year} after {prev_year}.")
            prev_year = year
        except Exception:
            errors.append(f"{label}.years[{idx}] must be numeric. Found {year!r}.")
        try:
            float(rate)
        except Exception:
            errors.append(f"{label}.rates[{idx}] must be numeric. Found {rate!r}.")
    return errors


def validate_issuance_profile(issuance_profile: dict, label: str = 'treasury_issuance_profile') -> List[str]:
    errors: List[str] = []
    if not isinstance(issuance_profile, collections.abc.Mapping):
        return [f"{label} must be a mapping."]

    fixed_remainder_sum = 0.0
    for cat in MATURITY_CATEGORIES:
        cat_cfg = issuance_profile.get(cat, {})
        if not isinstance(cat_cfg, collections.abc.Mapping):
            errors.append(f"{label}.{cat} must be a mapping.")
            continue
        target_pct = cat_cfg.get('target_percentage_of_remainder', 0.0)
        _assert_nonnegative(target_pct, f"{label}.{cat}.target_percentage_of_remainder", errors)
        try:
            fixed_remainder_sum += float(target_pct)
        except Exception:
            pass
        maturities = cat_cfg.get('maturities', [])
        distribution = cat_cfg.get('maturity_distribution', [])
        if maturities or distribution:
            if len(maturities) != len(distribution):
                errors.append(
                    f"{label}.{cat}.maturities and {label}.{cat}.maturity_distribution must have the same length."
                )
            if len(maturities) == 0:
                errors.append(f"{label}.{cat}.maturities cannot be empty when a distribution is provided.")
            dist_sum = 0.0
            for idx, weight in enumerate(distribution):
                _assert_nonnegative(weight, f"{label}.{cat}.maturity_distribution[{idx}]", errors)
                try:
                    dist_sum += float(weight)
                except Exception:
                    pass
            if distribution and dist_sum <= TGA_FLOOR_TOLERANCE:
                errors.append(f"{label}.{cat}.maturity_distribution must sum to a positive value.")

    if abs(fixed_remainder_sum - 1.0) > 1.0e-6:
        errors.append(
            f"{label} fixed-rate remainder shares must sum to 1.0 across bills/notes/bonds. Found {fixed_remainder_sum:.6f}."
        )

    for special_cat in ['TIPS', 'FRN', 'NonMarketable']:
        special_cfg = issuance_profile.get(special_cat, {})
        if isinstance(special_cfg, collections.abc.Mapping):
            target_pct = special_cfg.get('target_percentage', 0.0)
            _assert_nonnegative(target_pct, f"{label}.{special_cat}.target_percentage", errors)
            try:
                pct_val = float(target_pct)
            except (TypeError, ValueError):
                pct_val = 0.0
            if pct_val > TGA_FLOOR_TOLERANCE:
                mats = special_cfg.get('maturities', [])
                dist = special_cfg.get('maturity_distribution', [])
                if not mats:
                    errors.append(
                        f"{label}.{special_cat} has target_percentage={pct_val:.4f} but no 'maturities' defined. "
                        f"Issuance would be silently skipped."
                    )
                elif not dist:
                    errors.append(
                        f"{label}.{special_cat} has target_percentage={pct_val:.4f} but no 'maturity_distribution' defined."
                    )
                elif len(mats) != len(dist):
                    errors.append(
                        f"{label}.{special_cat}.maturities and maturity_distribution must have the same length."
                    )

    special_pct_sum = 0.0
    for special_cat in ['TIPS', 'FRN', 'NonMarketable']:
        special_cfg = issuance_profile.get(special_cat, {})
        if isinstance(special_cfg, collections.abc.Mapping):
            try:
                special_pct_sum += float(special_cfg.get('target_percentage', 0.0))
            except Exception:
                pass
    if special_pct_sum > 1.0 + 1.0e-6:
        errors.append(f"{label} special-security target percentages exceed 1.0 in total ({special_pct_sum:.6f}).")

    return errors


def validate_sector_preferences(
    sector_preferences: dict,
    issuance_profile: dict | None = None,
    label: str = 'sector_preferences',
    enforce_column_sums: bool = True,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(sector_preferences, collections.abc.Mapping):
        return [f"{label} must be a mapping."]

    for holder, holder_cfg in sector_preferences.items():
        if holder not in HOLDER_TYPES:
            errors.append(f"{label} contains unknown holder '{holder}'.")
            continue
        if not isinstance(holder_cfg, collections.abc.Mapping):
            errors.append(f"{label}.{holder} must be a mapping.")
            continue
        for pref_key, pref_value in holder_cfg.items():
            if not (pref_key.endswith('_pct') and pref_key[:-4] in _VALID_PREF_CATEGORIES):
                errors.append(
                    f"{label}.{holder} contains unexpected key '{pref_key}'. "
                    f"Valid keys: bills_pct, notes_pct, bonds_pct, tips_pct, frn_pct, nonmarketable_pct"
                )
            _assert_nonnegative(pref_value, f"{label}.{holder}.{pref_key}", errors)

    if not enforce_column_sums:
        return errors

    active_categories = set(PREFERENCE_CATEGORIES)
    if isinstance(issuance_profile, collections.abc.Mapping):
        # Only insist on unit sums for categories that are actually present or can be traded.
        active_categories = {'bills', 'notes', 'bonds'}
        for special_cat, pref_key in [('TIPS', 'tips'), ('FRN', 'frn'), ('NonMarketable', 'nonmarketable')]:
            cfg = issuance_profile.get(special_cat, {})
            if isinstance(cfg, collections.abc.Mapping):
                try:
                    if float(cfg.get('target_percentage', 0.0)) > TGA_FLOOR_TOLERANCE:
                        active_categories.add(pref_key)
                except Exception:
                    errors.append(f"{label}: could not parse target percentage for {special_cat}.")

    for pref_cat in sorted(active_categories):
        pref_key = f'{pref_cat}_pct'
        total = 0.0
        contributors = 0
        for holder in HOLDER_TYPES:
            holder_cfg = sector_preferences.get(holder, {})
            if not isinstance(holder_cfg, collections.abc.Mapping):
                continue
            if pref_key in holder_cfg:
                contributors += 1
                try:
                    total += float(holder_cfg.get(pref_key, 0.0))
                except Exception:
                    errors.append(f"{label}.{holder}.{pref_key} must be numeric.")
        if contributors > 0 and abs(total - 1.0) > 1.0e-6:
            errors.append(
                f"{label} column '{pref_key}' must sum to 1.0 across holders. Found {total:.6f}."
            )
        elif contributors == 0 and pref_cat in active_categories:
            # Nonmarketable securities are never traded on the secondary market
            if not (label.startswith('secondary_target') and pref_cat == 'nonmarketable'):
                errors.append(
                    f"{label} column '{pref_key}' has no contributors but category "
                    f"'{pref_cat}' is active. At least one holder must specify a preference."
                )
    return errors


def validate_nonmarketable_params(params: dict, label: str = 'nonmarketable_params') -> List[str]:
    errors: List[str] = []
    if not isinstance(params, collections.abc.Mapping):
        return errors
    freq = params.get('interest_crediting_frequency')
    if freq is not None and freq not in VALID_NONMARKETABLE_CREDITING_FREQUENCIES:
        errors.append(
            f"{label}.interest_crediting_frequency '{freq}' is not supported. "
            f"Valid values: {sorted(VALID_NONMARKETABLE_CREDITING_FREQUENCIES)}"
        )
    return errors


def validate_events(events: Iterable[dict], label: str = 'events') -> List[str]:
    import pandas as _pd
    errors: List[str] = []
    if events is None:
        return errors
    if not isinstance(events, list):
        return [f"{label} must be a list."]
    for idx, event_def in enumerate(events):
        if not isinstance(event_def, collections.abc.Mapping):
            errors.append(f"{label}[{idx}] must be a mapping.")
            continue
        date_str = event_def.get('date')
        if not date_str:
            errors.append(f"{label}[{idx}] is missing 'date'.")
        else:
            try:
                _pd.to_datetime(date_str)
            except Exception:
                errors.append(f"{label}[{idx}] has unparseable date '{date_str}'.")
        actions = event_def.get('actions')
        if not isinstance(actions, list) or not actions:
            errors.append(f"{label}[{idx}] must contain a non-empty 'actions' list.")
            continue
        for action_idx, action in enumerate(actions):
            if not isinstance(action, collections.abc.Mapping):
                errors.append(f"{label}[{idx}].actions[{action_idx}] must be a mapping.")
                continue
            parameter_path = action.get('parameter_path')
            if not parameter_path:
                errors.append(f"{label}[{idx}].actions[{action_idx}] is missing 'parameter_path'.")
                continue
            if 'new_value' not in action:
                errors.append(f"{label}[{idx}].actions[{action_idx}] is missing 'new_value'.")
            path_keys = str(parameter_path).split('.')
            is_valid, err_msg = validate_event_path(path_keys)
            if not is_valid:
                errors.append(
                    f"{label}[{idx}].actions[{action_idx}]: {err_msg} "
                    f"(parameter_path='{parameter_path}')"
                )
    return errors


def validate_config(base_config: dict) -> List[str]:
    errors: List[str] = []
    if not isinstance(base_config, collections.abc.Mapping):
        return ['Configuration root must be a mapping.']

    if 'nonmarketable_params' in base_config:
        errors.extend(validate_nonmarketable_params(base_config.get('nonmarketable_params', {})))
    if 'yield_curve' in base_config and base_config.get('yield_curve'):
        errors.extend(validate_yield_curve(base_config.get('yield_curve', {})))
    if 'treasury_issuance_profile' in base_config and base_config.get('treasury_issuance_profile'):
        errors.extend(validate_issuance_profile(base_config.get('treasury_issuance_profile', {})))
    if 'sector_preferences' in base_config and base_config.get('sector_preferences'):
        errors.extend(
            validate_sector_preferences(
                base_config.get('sector_preferences', {}),
                issuance_profile=base_config.get('treasury_issuance_profile', {}),
            )
        )
    if 'auction_absorption_preferences' in base_config:
        errors.extend(
            validate_sector_preferences(
                base_config.get('auction_absorption_preferences', {}),
                issuance_profile=base_config.get('treasury_issuance_profile', {}),
                label='auction_absorption_preferences',
            )
        )
    if 'secondary_target_preferences' in base_config:
        errors.extend(
            validate_sector_preferences(
                base_config.get('secondary_target_preferences', {}),
                issuance_profile=base_config.get('treasury_issuance_profile', {}),
                label='secondary_target_preferences',
            )
        )
    errors.extend(validate_events(base_config.get('events', [])))

    scenario_groups = base_config.get('scenario_groups', [])
    if isinstance(scenario_groups, list):
        for group_idx, group in enumerate(scenario_groups):
            if not isinstance(group, collections.abc.Mapping):
                errors.append(f"scenario_groups[{group_idx}] must be a mapping.")
                continue
            for scen_idx, scenario in enumerate(group.get('scenarios', [])):
                if not isinstance(scenario, collections.abc.Mapping):
                    errors.append(f"scenario_groups[{group_idx}].scenarios[{scen_idx}] must be a mapping.")
                    continue
                overrides = scenario.get('overrides', {})
                if not isinstance(overrides, collections.abc.Mapping):
                    errors.append(f"Scenario '{scenario.get('name', scen_idx)}' overrides must be a mapping.")
                    continue
                unknown_keys = set(overrides.keys()) - VALID_OVERRIDE_KEYS
                if unknown_keys:
                    scen_name = scenario.get('name', scen_idx)
                    errors.append(
                        f"Scenario '{scen_name}' has unknown override keys: {sorted(unknown_keys)}. "
                        f"Valid keys: {sorted(VALID_OVERRIDE_KEYS)}"
                    )
                if 'yield_curve' in overrides:
                    errors.extend(validate_yield_curve(overrides['yield_curve'], label=f"scenario_groups[{group_idx}].scenarios[{scen_idx}].overrides.yield_curve"))
                if 'treasury_issuance_profile' in overrides:
                    errors.extend(validate_issuance_profile(overrides['treasury_issuance_profile'], label=f"scenario_groups[{group_idx}].scenarios[{scen_idx}].overrides.treasury_issuance_profile"))
                if 'sector_preferences' in overrides:
                    profile_for_prefs = overrides.get('treasury_issuance_profile', base_config.get('treasury_issuance_profile', {}))
                    errors.extend(
                        validate_sector_preferences(
                            overrides['sector_preferences'],
                            issuance_profile=profile_for_prefs,
                            label=f"scenario_groups[{group_idx}].scenarios[{scen_idx}].overrides.sector_preferences",
                        )
                    )
                for pref_block in ['auction_absorption_preferences', 'secondary_target_preferences']:
                    if pref_block in overrides:
                        profile_for_prefs = overrides.get('treasury_issuance_profile', base_config.get('treasury_issuance_profile', {}))
                        errors.extend(
                            validate_sector_preferences(
                                overrides[pref_block],
                                issuance_profile=profile_for_prefs,
                                label=f"scenario_groups[{group_idx}].scenarios[{scen_idx}].overrides.{pref_block}",
                            )
                        )
                if 'events' in overrides:
                    errors.extend(validate_events(overrides['events'], label=f"scenario_groups[{group_idx}].scenarios[{scen_idx}].overrides.events"))
    return errors
