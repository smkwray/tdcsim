
"""Portfolio valuation and secondary-market preference trading."""

import traceback

import numpy as np
import pandas as pd

from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    DAYS_PER_YEAR_ACTUAL,
    HOLDER_TYPES,
    INTRAGOV_HOLDERS,
    PORTFOLIO_DTYPES,
    SECURITY_TYPES,
    TGA_FLOOR_TOLERANCE,
)
from sim_pricing import (
    calculate_accrued_interest,
    calculate_bond_market_price,
    get_security_category_for_prefs,
    get_yield_for_maturity,
)

def calculate_portfolio_value_and_composition(portfolio_df, current_date, yield_curve_years, yield_curve_rates):
    """
    Calculates total dirty market value and composition by holder and security type.
    """
    if portfolio_df is None or portfolio_df.empty:
        empty_val = {h: 0.0 for h in HOLDER_TYPES}
        empty_comp = {h: {st: 0.0 for st in SECURITY_TYPES} for h in HOLDER_TYPES}
        empty_type = {st: 0.0 for st in SECURITY_TYPES}
        return (empty_val, empty_comp, empty_type)
    active_portfolio = portfolio_df[portfolio_df['Status'] == 'Active'].copy()
    if active_portfolio.empty:
        empty_val = {h: 0.0 for h in HOLDER_TYPES}
        empty_comp = {h: {st: 0.0 for st in SECURITY_TYPES} for h in HOLDER_TYPES}
        empty_type = {st: 0.0 for st in SECURITY_TYPES}
        return (empty_val, empty_comp, empty_type)
    required_cols = ['TimeToMaturity', 'DiscountYield', 'CleanPrice', 'AccruedInterest', 'DirtyValue', 'DirtyPriceRatio']
    recalculate_pricing = False
    for col in required_cols:
        if col not in active_portfolio.columns or active_portfolio[col].isnull().any():
            recalculate_pricing = True
            break
    if recalculate_pricing:
        active_portfolio['TimeToMaturity'] = (active_portfolio['MaturityDate'] - current_date).dt.total_seconds() / (DAYS_PER_YEAR_ACTUAL * 24 * 60 * 60)
        active_portfolio['TimeToMaturity'] = active_portfolio['TimeToMaturity'].clip(lower=0.0)
        active_portfolio['DiscountYield'] = active_portfolio['TimeToMaturity'].apply(lambda ttm: get_yield_for_maturity(ttm, yield_curve_years, yield_curve_rates, method='pchip') if ttm > TGA_FLOOR_TOLERANCE else np.nan)

        def calculate_row_prices(row):
            frequency = 4 if row['SecurityType'] == 'FRN' else 2
            clean_price = calculate_bond_market_price(row['FaceValue'], row['CouponRate'], row['MaturityDate'], current_date, row['DiscountYield'], row['SecurityType'], row.get('AdjustedPrincipal'), row.get('OriginalPrincipal'), row.get('AccruedInterest_FRN'), frequency)
            accrued_interest = calculate_accrued_interest(row['FaceValue'], row['CouponRate'], current_date, row['IssueDate'], row['SecurityType'], row.get('AdjustedPrincipal'), row.get('AccruedInterest_FRN'), frequency)
            dirty_value = clean_price + accrued_interest
            dirty_price_ratio = dirty_value / row['FaceValue'] if row['FaceValue'] > TGA_FLOOR_TOLERANCE else 1.0
            return pd.Series([clean_price, accrued_interest, dirty_value, dirty_price_ratio])
        price_results = active_portfolio.apply(calculate_row_prices, axis=1)
        price_results.columns = ['CleanPrice', 'AccruedInterest', 'DirtyValue', 'DirtyPriceRatio']
        active_portfolio[['CleanPrice', 'AccruedInterest', 'DirtyValue', 'DirtyPriceRatio']] = price_results
        active_portfolio['CleanPrice'] = active_portfolio['CleanPrice'].fillna(0.0)
        active_portfolio['AccruedInterest'] = active_portfolio['AccruedInterest'].fillna(0.0)
        active_portfolio['DirtyValue'] = active_portfolio['DirtyValue'].fillna(0.0)
        active_portfolio['DirtyPriceRatio'] = active_portfolio['DirtyPriceRatio'].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    active_portfolio['DirtyValue'] = active_portfolio['DirtyValue'].clip(lower=0.0)
    value_by_holder = active_portfolio.groupby('HolderType')['DirtyValue'].sum().to_dict()
    value_by_type = active_portfolio.groupby('SecurityType')['DirtyValue'].sum().to_dict()
    composition_by_holder = {}
    for holder in HOLDER_TYPES:
        holder_total_value = value_by_holder.get(holder, 0.0)
        composition = {}
        if holder_total_value > TGA_FLOOR_TOLERANCE:
            holder_portfolio = active_portfolio[active_portfolio['HolderType'] == holder]
            type_values = holder_portfolio.groupby('SecurityType')['DirtyValue'].sum()
            for sec_type in SECURITY_TYPES:
                composition[sec_type] = type_values.get(sec_type, 0.0) / holder_total_value
        else:
            for sec_type in SECURITY_TYPES:
                composition[sec_type] = 0.0
        composition_by_holder[holder] = composition
    for holder in HOLDER_TYPES:
        value_by_holder.setdefault(holder, 0.0)
    for sec_type in SECURITY_TYPES:
        value_by_type.setdefault(sec_type, 0.0)
    try:
        if portfolio_df.index.equals(active_portfolio.index):
            cols_to_update_cache = ['TimeToMaturity', 'DiscountYield', 'CleanPrice', 'AccruedInterest', 'DirtyValue', 'DirtyPriceRatio']
            portfolio_df.update(active_portfolio[cols_to_update_cache])
    except Exception as e:
        pass
    return (value_by_holder, composition_by_holder, value_by_type)

def execute_preference_trades(bond_portfolio, current_date, yield_curve_years, yield_curve_rates, sector_target_prefs, issuance_profile, scenario_name):
    """
    Executes secondary market trades between holders based on deviations from their target portfolio preferences.
    Uses CURRENT sector_target_prefs passed in.
    """
    tradeable_mask = (bond_portfolio['SecurityType'] != 'NonMarketable') & (bond_portfolio['Status'] == 'Active')
    if not tradeable_mask.any():
        return (bond_portfolio, {'reserve_change': 0.0, 'deposit_change': 0.0, 'tga_change': 0.0, 'tga_drain': 0.0})
    try:
        bond_portfolio = bond_portfolio.astype(PORTFOLIO_DTYPES, errors='ignore')
    except Exception as e:
        return (bond_portfolio, {'reserve_change': 0.0, 'deposit_change': 0.0, 'tga_change': 0.0, 'tga_drain': 0.0})
    tradeable_portfolio = bond_portfolio[tradeable_mask].copy()
    calculate_portfolio_value_and_composition(tradeable_portfolio, current_date, yield_curve_years, yield_curve_rates)
    if 'DirtyValue' not in tradeable_portfolio.columns or tradeable_portfolio['DirtyValue'].isnull().any():
        return (bond_portfolio, {'reserve_change': 0.0, 'deposit_change': 0.0, 'tga_change': 0.0, 'tga_drain': 0.0})
    value_by_holder_tradeable = tradeable_portfolio.groupby('HolderType')['DirtyValue'].sum().to_dict()
    imbalance_values = {}
    for holder in HOLDER_TYPES:
        holder_total_value = value_by_holder_tradeable.get(holder, 0.0)
        holder_portfolio_subset = tradeable_portfolio[tradeable_portfolio['HolderType'] == holder]
        holder_imbalances = {}
        target_prefs = sector_target_prefs.get(holder, {})
        target_bills_pct = target_prefs.get('bills_pct', 0.0)
        target_notes_pct = target_prefs.get('notes_pct', 0.0)
        target_bonds_pct = target_prefs.get('bonds_pct', 0.0)
        target_tips_pct = target_prefs.get('tips_pct', 0.0)
        target_frn_pct = target_prefs.get('frn_pct', 0.0)
        actual_values_by_cat = {cat: 0.0 for cat in ['bills', 'notes', 'bonds', 'tips', 'frn']}
        for _, row in holder_portfolio_subset.iterrows():
            cat_key = get_security_category_for_prefs(row['SecurityType'], row['OriginalMaturityYears'], issuance_profile)
            if cat_key in actual_values_by_cat:
                actual_values_by_cat[cat_key] += row['DirtyValue']
        actual_bills_value = actual_values_by_cat['bills']
        actual_notes_value = actual_values_by_cat['notes']
        actual_bonds_value = actual_values_by_cat['bonds']
        actual_tips_value = actual_values_by_cat['tips']
        actual_frn_value = actual_values_by_cat['frn']
        if holder_total_value > TGA_FLOOR_TOLERANCE:
            holder_imbalances['bills'] = actual_bills_value - holder_total_value * target_bills_pct
            holder_imbalances['notes'] = actual_notes_value - holder_total_value * target_notes_pct
            holder_imbalances['bonds'] = actual_bonds_value - holder_total_value * target_bonds_pct
            holder_imbalances['tips'] = actual_tips_value - holder_total_value * target_tips_pct
            holder_imbalances['frn'] = actual_frn_value - holder_total_value * target_frn_pct
        else:
            for cat in ['bills', 'notes', 'bonds', 'tips', 'frn']:
                holder_imbalances[cat] = 0.0
        imbalance_values[holder] = holder_imbalances
    potential_trades = []
    for seller, seller_imbalances in imbalance_values.items():
        for buyer, buyer_imbalances in imbalance_values.items():
            if seller == buyer:
                continue
            for category, seller_surplus in seller_imbalances.items():
                buyer_deficit = buyer_imbalances.get(category, 0.0)
                if seller_surplus > TGA_FLOOR_TOLERANCE and buyer_deficit < -TGA_FLOOR_TOLERANCE:
                    trade_magnitude = min(seller_surplus, abs(buyer_deficit))
                    potential_trades.append({'seller': seller, 'buyer': buyer, 'category': category, 'magnitude_value': trade_magnitude})
    potential_trades.sort(key=lambda x: x['magnitude_value'], reverse=True)
    net_reserve_change = 0.0
    net_deposit_change = 0.0
    net_tga_change_inflow = 0.0
    net_tga_drain_outflow = 0.0
    traded_this_round = False
    max_trades_per_round = 250
    trade_count = 0
    modified_portfolio = bond_portfolio.copy()
    tradeable_portfolio_indexed = tradeable_portfolio.reset_index(drop=True)
    for trade_info in potential_trades:
        if trade_count >= max_trades_per_round:
            break
        seller = trade_info['seller']
        buyer = trade_info['buyer']
        category = trade_info['category']
        current_seller_surplus = imbalance_values.get(seller, {}).get(category, 0.0)
        current_buyer_deficit = imbalance_values.get(buyer, {}).get(category, 0.0)
        if not (current_seller_surplus > TGA_FLOOR_TOLERANCE and current_buyer_deficit < -TGA_FLOOR_TOLERANCE):
            continue
        seller_bonds_mask = tradeable_portfolio_indexed['HolderType'] == seller

        def check_cat(row):
            return get_security_category_for_prefs(row['SecurityType'], row['OriginalMaturityYears'], issuance_profile) == category
        if category in ['bills', 'notes', 'bonds']:
            seller_bonds_mask &= tradeable_portfolio_indexed.apply(check_cat, axis=1)
        elif category == 'tips':
            seller_bonds_mask &= tradeable_portfolio_indexed['SecurityType'] == 'TIPS'
        elif category == 'frn':
            seller_bonds_mask &= tradeable_portfolio_indexed['SecurityType'] == 'FRN'
        else:
            continue
        candidate_bonds_priced = tradeable_portfolio_indexed[seller_bonds_mask].sort_values(['DirtyValue', 'TimeToMaturity'], ascending=[False, False])
        if candidate_bonds_priced.empty:
            continue
        bond_to_trade_priced = candidate_bonds_priced.iloc[0]
        bond_id_to_trade = bond_to_trade_priced['BondID']
        dirty_price_ratio = bond_to_trade_priced.get('DirtyPriceRatio', 1.0)
        if pd.isna(dirty_price_ratio) or dirty_price_ratio < TGA_FLOOR_TOLERANCE:
            continue
        original_tranche_mask = (modified_portfolio['BondID'] == bond_id_to_trade) & (modified_portfolio['HolderType'] == seller) & (modified_portfolio['Status'] == 'Active')
        original_indices = modified_portfolio[original_tranche_mask].index
        if original_indices.empty:
            continue
        original_tranche_idx = original_indices[0]
        seller_tranche_face_value = modified_portfolio.loc[original_tranche_idx, 'FaceValue']
        if seller_tranche_face_value < TGA_FLOOR_TOLERANCE:
            continue
        seller_frn_accrued_before_trade = 0.0
        if modified_portfolio.loc[original_tranche_idx, 'SecurityType'] == 'FRN':
            seller_frn_accrued_before_trade = modified_portfolio.loc[original_tranche_idx, 'AccruedInterest_FRN']
            seller_frn_accrued_before_trade = 0.0 if pd.isna(seller_frn_accrued_before_trade) else float(seller_frn_accrued_before_trade)
        value_to_trade = min(current_seller_surplus, abs(current_buyer_deficit), bond_to_trade_priced['DirtyValue'])
        if value_to_trade < TGA_FLOOR_TOLERANCE:
            continue
        face_value_to_trade = min(value_to_trade / dirty_price_ratio, seller_tranche_face_value)
        if face_value_to_trade < TGA_FLOOR_TOLERANCE:
            continue
        actual_dirty_value_traded = face_value_to_trade * dirty_price_ratio
        if actual_dirty_value_traded < TGA_FLOOR_TOLERANCE:
            continue
        frn_accrued_to_transfer = 0.0
        if modified_portfolio.loc[original_tranche_idx, 'SecurityType'] == 'FRN' and seller_tranche_face_value > TGA_FLOOR_TOLERANCE:
            frn_accrued_to_transfer = seller_frn_accrued_before_trade * (face_value_to_trade / seller_tranche_face_value)
        _proportional_fields = ['OriginalPrincipal', 'AdjustedPrincipal', 'IssueProceeds']
        _seller_orig_vals = {}
        for _fld in _proportional_fields:
            _v = modified_portfolio.loc[original_tranche_idx, _fld]
            _seller_orig_vals[_fld] = 0.0 if pd.isna(_v) else float(_v)
        _transfer_ratio = face_value_to_trade / seller_tranche_face_value
        try:
            modified_portfolio.loc[original_tranche_idx, 'FaceValue'] -= face_value_to_trade
            new_seller_fv = modified_portfolio.loc[original_tranche_idx, 'FaceValue']
            if modified_portfolio.loc[original_tranche_idx, 'SecurityType'] == 'FRN':
                modified_portfolio.loc[original_tranche_idx, 'AccruedInterest_FRN'] = max(0.0, seller_frn_accrued_before_trade - frn_accrued_to_transfer)
            for _fld in _proportional_fields:
                _orig = _seller_orig_vals[_fld]
                if abs(_orig) > TGA_FLOOR_TOLERANCE:
                    modified_portfolio.loc[original_tranche_idx, _fld] = _orig * (1.0 - _transfer_ratio)
            buyer_tranche_mask = (modified_portfolio['BondID'] == bond_id_to_trade) & (modified_portfolio['HolderType'] == buyer) & (modified_portfolio['Status'] == 'Active')
            existing_buyer_tranche = modified_portfolio[buyer_tranche_mask]
            if not existing_buyer_tranche.empty:
                buyer_idx = existing_buyer_tranche.index[0]
                modified_portfolio.loc[buyer_idx, 'FaceValue'] += face_value_to_trade
                if modified_portfolio.loc[buyer_idx, 'SecurityType'] == 'FRN':
                    current_buyer_accrued = modified_portfolio.loc[buyer_idx, 'AccruedInterest_FRN']
                    current_buyer_accrued = 0.0 if pd.isna(current_buyer_accrued) else float(current_buyer_accrued)
                    modified_portfolio.loc[buyer_idx, 'AccruedInterest_FRN'] = current_buyer_accrued + frn_accrued_to_transfer
                    if pd.isna(modified_portfolio.loc[buyer_idx, 'LastAccrualDate']):
                        modified_portfolio.loc[buyer_idx, 'LastAccrualDate'] = modified_portfolio.loc[original_tranche_idx, 'LastAccrualDate']
                for _fld in _proportional_fields:
                    _orig = _seller_orig_vals[_fld]
                    if abs(_orig) > TGA_FLOOR_TOLERANCE:
                        _cur = modified_portfolio.loc[buyer_idx, _fld]
                        _cur = 0.0 if pd.isna(_cur) else float(_cur)
                        modified_portfolio.loc[buyer_idx, _fld] = _cur + _orig * _transfer_ratio
            else:
                new_buyer_tranche_dict = modified_portfolio.loc[original_tranche_idx].to_dict()
                new_buyer_tranche_dict['HolderType'] = buyer
                new_buyer_tranche_dict['FaceValue'] = face_value_to_trade
                new_buyer_tranche_dict['Status'] = 'Active'
                if new_buyer_tranche_dict['SecurityType'] == 'FRN':
                    new_buyer_tranche_dict['AccruedInterest_FRN'] = frn_accrued_to_transfer
                    new_buyer_tranche_dict['LastAccrualDate'] = modified_portfolio.loc[original_tranche_idx, 'LastAccrualDate']
                for _fld in _proportional_fields:
                    new_buyer_tranche_dict[_fld] = _seller_orig_vals[_fld] * _transfer_ratio
                for col in ['TimeToMaturity', 'DiscountYield', 'CleanPrice', 'AccruedInterest', 'DirtyValue', 'DirtyPriceRatio']:
                    new_buyer_tranche_dict[col] = np.nan
                new_row_df = pd.DataFrame([new_buyer_tranche_dict], columns=BOND_PORTFOLIO_COLS)
                new_row_df = new_row_df.astype(PORTFOLIO_DTYPES, errors='ignore')
                modified_portfolio = pd.concat([modified_portfolio, new_row_df], ignore_index=True)
            traded_this_round = True
            trade_count += 1
            buyer_is_gov = buyer in INTRAGOV_HOLDERS
            seller_is_gov = seller in INTRAGOV_HOLDERS
            if buyer_is_gov and seller_is_gov:
                net_tga_drain_outflow += actual_dirty_value_traded
                net_tga_change_inflow += actual_dirty_value_traded
            elif buyer_is_gov:
                net_tga_drain_outflow += actual_dirty_value_traded
                if seller == 'Banks':
                    net_reserve_change += actual_dirty_value_traded
                elif seller == 'Private':
                    net_deposit_change += actual_dirty_value_traded
                    net_reserve_change += actual_dirty_value_traded
                elif seller == 'Foreign':
                    net_reserve_change += actual_dirty_value_traded
                elif seller == 'CB':
                    pass
            elif seller_is_gov:
                net_tga_change_inflow += actual_dirty_value_traded
                if buyer == 'Banks':
                    net_reserve_change -= actual_dirty_value_traded
                elif buyer == 'Private':
                    net_deposit_change -= actual_dirty_value_traded
                    net_reserve_change -= actual_dirty_value_traded
                elif buyer == 'Foreign':
                    net_reserve_change -= actual_dirty_value_traded
                elif buyer == 'CB':
                    pass
            elif buyer == 'Private' and seller == 'Banks':
                net_deposit_change -= actual_dirty_value_traded
            elif buyer == 'Banks' and seller == 'Private':
                net_deposit_change += actual_dirty_value_traded
            elif buyer == 'CB' and seller == 'Banks':
                net_reserve_change += actual_dirty_value_traded
            elif buyer == 'CB' and seller == 'Private':
                net_reserve_change += actual_dirty_value_traded
                net_deposit_change += actual_dirty_value_traded
            elif buyer == 'Banks' and seller == 'CB':
                net_reserve_change -= actual_dirty_value_traded
            elif buyer == 'Private' and seller == 'CB':
                net_reserve_change -= actual_dirty_value_traded
                net_deposit_change -= actual_dirty_value_traded
            elif buyer == 'Foreign' and seller == 'Banks':
                pass
            elif buyer == 'Banks' and seller == 'Foreign':
                pass
            elif buyer == 'Foreign' and seller == 'Private':
                net_deposit_change -= actual_dirty_value_traded
            elif buyer == 'Private' and seller == 'Foreign':
                net_deposit_change += actual_dirty_value_traded
            elif buyer == 'Foreign' and seller == 'CB':
                net_reserve_change -= actual_dirty_value_traded
            elif buyer == 'CB' and seller == 'Foreign':
                net_reserve_change += actual_dirty_value_traded
            imbalance_values[seller][category] -= actual_dirty_value_traded
            imbalance_values[buyer][category] += actual_dirty_value_traded
            tradeable_idx = bond_to_trade_priced.name
            if tradeable_idx in tradeable_portfolio_indexed.index:
                tradeable_portfolio_indexed.loc[tradeable_idx, 'FaceValue'] = new_seller_fv
                tradeable_portfolio_indexed.loc[tradeable_idx, 'DirtyValue'] = new_seller_fv * dirty_price_ratio
                if new_seller_fv < TGA_FLOOR_TOLERANCE:
                    tradeable_portfolio_indexed = tradeable_portfolio_indexed.drop(index=tradeable_idx)
        except KeyError as e:
            continue
        except Exception as e:
            print(f'ERROR [{scenario_name}@{current_date.date()}]: Unexpected error during preference trade execution: {e}')
            traceback.print_exc()
            continue
    final_portfolio = modified_portfolio[modified_portfolio['FaceValue'] > TGA_FLOOR_TOLERANCE].reset_index(drop=True)
    monetary_impact = {'reserve_change': net_reserve_change, 'deposit_change': net_deposit_change, 'tga_change': net_tga_change_inflow, 'tga_drain': net_tga_drain_outflow}
    try:
        final_portfolio = final_portfolio.astype(PORTFOLIO_DTYPES, errors='ignore')
    except Exception:
        pass
    return (final_portfolio, monetary_impact)


__all__ = [
    'calculate_portfolio_value_and_composition',
    'execute_preference_trades',
]
