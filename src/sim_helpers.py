
"""Helper utilities and validation for the Treasury funding-chain simulator."""

import collections.abc
import copy

import pandas as pd

from tdc_validation import (
    VALID_OVERRIDE_KEYS,
    validate_events,
    validate_financing_cost_options,
    validate_issuance_profile,
    validate_rate_sensitive_demand,
    validate_sector_preferences,
    validate_simulation_period,
    validate_yield_curve,
)

OUTPUT_COLUMN_RENAMES = {
    'DebtHeld_Private': 'DebtHeld_DomesticNonBanks',
    'DebtHeld_CB': 'DebtHeld_CentralBank',
}

def _set_nested_value(target_dict, path_keys, value):
    """
    Sets a value in a nested dictionary using a list of keys.
    Modifies the dictionary in place. Returns True if successful, False otherwise.
    """
    d = target_dict
    last_key = path_keys[-1]
    try:
        for key in path_keys[:-1]:
            if key not in d or not isinstance(d[key], collections.abc.Mapping):
                return False
            d = d[key]
        if not isinstance(d, collections.abc.Mapping):
            return False
        if last_key not in d:
            return False
        d[last_key] = value
        return True
    except Exception as e:
        return False

def update_dict_recursive(d, u):
    """Recursively update nested dictionaries."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            if k in d and isinstance(d.get(k), collections.abc.Mapping):
                r = update_dict_recursive(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = v
        else:
            d[k] = v
    return d

def apply_event_actions(actions, dynamic_params_state, scenario_name, current_date, propagate_legacy_sector_prefs_to_auction=False, propagate_legacy_sector_prefs_to_secondary=False):
    """Apply a list of event actions. Raises on unsupported or failed actions."""
    for action in actions:
        path_keys = action['path_keys']
        new_value = action['new_value']
        param_root_key = path_keys[0]
        update_success = False
        if len(path_keys) == 1:
            if param_root_key in dynamic_params_state:
                dynamic_params_state[param_root_key] = copy.deepcopy(new_value)
                update_success = True
        elif len(path_keys) == 2 and param_root_key == 'simulation_period' and (path_keys[1] == 'enable_preference_trading'):
            dynamic_params_state['simulation_period']['enable_preference_trading'] = bool(new_value)
            update_success = True
        elif param_root_key in dynamic_params_state and isinstance(dynamic_params_state[param_root_key], collections.abc.Mapping):
            update_success = _set_nested_value(dynamic_params_state[param_root_key], path_keys[1:], new_value)
        if not update_success:
            raise ValueError(f"[{scenario_name}@{pd.to_datetime(current_date).date()}] Failed to apply event action for path {'.'.join(path_keys)}.")
        if param_root_key == 'sector_preferences':
            if propagate_legacy_sector_prefs_to_auction and 'auction_absorption_preferences' in dynamic_params_state:
                _set_nested_value(dynamic_params_state['auction_absorption_preferences'], path_keys[1:], new_value)
            if propagate_legacy_sector_prefs_to_secondary and 'secondary_target_preferences' in dynamic_params_state:
                _set_nested_value(dynamic_params_state['secondary_target_preferences'], path_keys[1:], new_value)

def validate_run_params(params, scenario_name='Scenario'):
    errors = []
    for required_block in ('fiscal_params', 'yield_curve', 'treasury_issuance_profile'):
        if required_block not in params or not params[required_block]:
            errors.append(f"Required parameter block '{required_block}' is missing or empty.")
    errors.extend(validate_simulation_period(params.get('simulation_period', {}), label='simulation_period'))
    errors.extend(validate_yield_curve(params.get('yield_curve', {}), label='yield_curve'))
    errors.extend(validate_issuance_profile(params.get('treasury_issuance_profile', {}), label='treasury_issuance_profile'))
    auction_prefs = params.get('auction_absorption_preferences', params.get('sector_preferences', {}))
    secondary_prefs = params.get('secondary_target_preferences', params.get('sector_preferences', {}))
    errors.extend(validate_sector_preferences(auction_prefs, issuance_profile=params.get('treasury_issuance_profile', {}), label='auction_absorption_preferences'))
    errors.extend(validate_sector_preferences(secondary_prefs, issuance_profile=params.get('treasury_issuance_profile', {}), label='secondary_target_preferences'))
    errors.extend(validate_rate_sensitive_demand(params.get('rate_sensitive_demand', {}), label='rate_sensitive_demand'))
    errors.extend(validate_financing_cost_options(params.get('financing_cost_options', {}), label='financing_cost_options'))
    errors.extend(validate_events(params.get('events', []), label='events'))
    if errors:
        msg = '\n - '.join(errors[:20])
        more = '' if len(errors) <= 20 else f'\n - ... and {len(errors) - 20} more'
        raise ValueError(f'[{scenario_name}] Configuration validation failed:\n - {msg}{more}')


__all__ = [
    'VALID_OVERRIDE_KEYS',
    'OUTPUT_COLUMN_RENAMES',
    '_set_nested_value',
    'update_dict_recursive',
    'apply_event_actions',
    'validate_run_params',
]
