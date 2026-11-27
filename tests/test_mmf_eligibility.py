"""Money funds may acquire only securities that Rule 2a-7 lets them hold.

A fund cannot buy a note or bond with more than 397 days left, so a model that hands
money funds new two-year notes or long bonds at auction gives them assets they could not
own and routes those securities' payments through the wrong deposit channel.
"""

import pandas as pd
import pytest

from mmf_eligibility import MMF_MAX_REMAINING_DAYS, MmfEligibilityError, mmf_eligible_mask, mmf_may_acquire

DAY = pd.Timestamp('2026-06-21')


def test_a_new_two_year_note_is_ineligible_but_the_same_note_near_maturity_is_eligible():
    assert not mmf_may_acquire('Fixed', DAY + pd.Timedelta(days=730), DAY)
    assert mmf_may_acquire('Fixed', DAY + pd.Timedelta(days=MMF_MAX_REMAINING_DAYS), DAY)
    assert not mmf_may_acquire('Fixed', DAY + pd.Timedelta(days=MMF_MAX_REMAINING_DAYS + 1), DAY)


def test_bills_are_eligible_and_frns_are_eligible_at_any_term():
    assert mmf_may_acquire('Fixed', DAY + pd.Timedelta(weeks=52), DAY)
    assert mmf_may_acquire('FRN', DAY + pd.Timedelta(days=730), DAY)


def test_tips_follow_the_397_day_limit():
    assert not mmf_may_acquire('TIPS', DAY + pd.Timedelta(days=5 * 365), DAY)
    assert mmf_may_acquire('TIPS', DAY + pd.Timedelta(days=300), DAY)


def test_the_frame_mask_matches_the_scalar_rule_row_by_row():
    frame = pd.DataFrame(
        {
            'SecurityType': ['Fixed', 'Fixed', 'FRN', 'TIPS', 'Fixed'],
            'MaturityDate': [
                DAY + pd.Timedelta(days=91),
                DAY + pd.Timedelta(days=3650),
                DAY + pd.Timedelta(days=700),
                DAY + pd.Timedelta(days=397),
                DAY + pd.Timedelta(days=398),
            ],
        }
    )
    expected = [mmf_may_acquire(t, m, DAY) for t, m in zip(frame['SecurityType'], frame['MaturityDate'])]
    assert mmf_eligible_mask(frame, DAY).tolist() == expected == [True, False, True, True, False]


def test_eligibility_fails_loud_on_nonmarketable_or_undated_securities():
    with pytest.raises(MmfEligibilityError):
        mmf_may_acquire('NonMarketable', DAY, DAY)
    with pytest.raises(MmfEligibilityError):
        mmf_may_acquire('Fixed', None, DAY)
    with pytest.raises(MmfEligibilityError):
        mmf_eligible_mask(pd.DataFrame({'SecurityType': ['TIPS'], 'MaturityDate': [None]}), DAY)
