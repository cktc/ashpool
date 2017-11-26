'''ashpool tests'''
import datetime

import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given
from hypothesis.strategies import text
from hypothesis.extra.pandas import column, data_frames


from ..ashpool import *

df_l = pd.DataFrame({'dt_1': {0: datetime.datetime(2017, 9, 29, 00, 00, 00),
                              1: datetime.datetime(2016, 3, 14, 00, 00, 00),
                              2: datetime.datetime(2017, 9, 29, 00, 00, 00),
                              3: datetime.datetime(2016, 3, 14, 00, 00, 00),
                              4: datetime.datetime(2017, 9, 29, 00, 00, 00),
                              5: datetime.datetime(2016, 3, 14, 00, 00, 00)},
                     'flt_1': {0: 12.300000000000001,
                               1: 23.100000000000001,
                               2: 43.200000000000003,
                               3: 25.800000000000001,
                               4: 0.10000000000000001,
                               5: np.nan},
                     'int_1': {0: 10, 1: 1, 2: 234, 3: 0, 4: 3, 5: 12},
                     'str_1': {0: 'A', 1: 'A', 2: 'A', 3: 'A', 4: 'A', 5: 'A'},
                     'str_2': {0: 'BC', 1: 'BC', 2: 'BC', 3: 'DE', 4: 'DE', 5: 'DE'},
                     'str_3': {0: 'z', 1: 'y', 2: 'x', 3: 'w', 4: 'v', 5: np.nan},
                     'int_2': {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3}})
df_r = pd.DataFrame({'dt_1': {0: datetime.datetime(2017, 9, 29, 00, 00, 00),
                              1: datetime.datetime(2016, 3, 14, 00, 00, 00),
                              2: datetime.datetime(2017, 9, 29, 00, 00, 00),
                              3: datetime.datetime(2016, 3, 14, 00, 00, 00),
                              4: datetime.datetime(2017, 9, 29, 00, 00, 00),
                              5: datetime.datetime(2016, 3, 14, 00, 00, 00)},
                     'flt_1': {0: 12.300000000000001,
                               1: 23.100000000000001,
                               2: 43.200000000000003,
                               3: 25.800000000000001,
                               4: 0.10000000000000001,
                               5: np.nan},
                     'int_1': {0: 10, 1: 1, 2: 234, 3: 0, 4: 3, 5: 12},
                     'str_1': {0: 'A', 1: 'A', 2: 'A', 3: 'A', 4: 'A', 5: 'A'},
                     'str_2': {0: 'BC', 1: 'BC', 2: 'BC', 3: 'FG', 4: 'FG', 5: 'FG'},
                     'str_3': {0: 'u', 1: 'y', 2: 'x', 3: 'w', 4: 'v', 5: np.nan},
                     'int_2': {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3}})

@given(text(min_size=3))
@example(' BAD label ')
@example(' BAD label !@#$!%!%')
def test_make_good_label(s):
    assert isinstance(make_good_label(s), str)
    assert make_good_label(' BAD label ') == 'bad_label'


def test_completeness():
    # assert completeness(df_l['flt_1']) == 0.8
    assert np.isclose(completeness(df_l['flt_1']), 0.833333)

def test_mash():
    df_result = mash(df_l, 'flt_1')
    assert df_result.shape[0] == 5
    df_result = mash(df_l, 'int_1')
    assert df_result.shape[0] == 5
    df_result = mash(df_l, 'int_1', keep_zeros=True)
    assert df_result.shape[0] == 6

def test_uniqueness():
    assert np.isclose(uniqueness(df_l['str_1']), 0.166666)
    assert np.isclose(uniqueness(df_l['str_2']), 0.333333)

def test_coveredness():
    assert np.isclose(coveredness(df_l['int_1'], df_l['int_2']), 0.333333)

def test_get_combos():
    assert len(get_combos(df_l['str_2'])) == 13

def test_attach_temp_id():
    assert attach_temp_id(df_l, field_list=['str_1', 'str_2'])['tempid'].tolist() == ['A_BC', 'A_BC', 'A_BC', 'A_DE', 'A_DE', 'A_DE']
    assert attach_temp_id(df_l, field_list=['str_1', 'dt_1'])

def test_rate_series():
    df_result = rate_series(df_l)
    assert df_result['fld'].tolist() == df_l.columns.tolist()


@given(data_frames([column('int_1', dtype=int), column('int_2', dtype=int), column('int_3', dtype=int), column('flt_1', dtype=float), column('flt_2', dtype=float)]))
def test_rate_series_2(df_l):
    df_result = rate_series(df_l)
    assert df_result['fld'].tolist() == df_l.columns.tolist()


def test_get_sorted_fields():
    result = get_sorted_fields(df_l)
    assert result == {'most_complete': ['str_1', 'str_2', 'dt_1', 'str_3'],
                     'most_unique': ['str_3', 'str_2', 'dt_1', 'str_1'],
                     'non_object': ['flt_1', 'int_1', 'int_2']}

def test_get_unique_fields():
    flds = get_sorted_fields(df_l[['str_2', 'int_1', 'int_2', 'dt_1']])
    u_flds = get_unique_fields(df_l, candidate_flds=flds['most_unique'], threshold=0.5)
    assert u_flds == ['str_2', 'dt_1']

def test_attach_unique_id():
    df_result = attach_unique_id(df_l, threshold=1)
    assert df_result['u_id'].tolist() == ['Z_A', 'Y_A', 'X_A', 'W_A', 'V_A', 'NAN_A']

def test_jaccard_similarity():
    assert isinstance(jaccard_similarity(df_l['int_1'], df_l['int_2']), float)

def test_has_name_match():
    assert isinstance(has_name_match(df_l['str_1'], df_r), bool)

def test_suggest_id_pairs():
    df_result = suggest_id_pairs(df_l, df_r)
    assert isinstance(df_result, pd.DataFrame)
    assert 'id_scr' in df_result.columns

def test_get_most_coveredness():
    assert isinstance(get_most_coveredness(df_l['dt_1'], df_r), list)

def test_check_coveredness():
    df_result = check_coveredness(df_l, df_r)
    assert isinstance(df_result, pd.DataFrame)
    assert 'coveredness' in df_result.columns

def test_cum_uniq():
    assert isinstance(cum_uniq(df_l, df_l.columns.tolist()), list)

def test_best_id_pair():
    assert isinstance(best_id_pair(df_l, df_r), pd.DataFrame)

def test_reconcile():
    df_result = reconcile(df_l, df_r, fields_l=['flt_1'], fields_r=['flt_1'])
    assert isinstance(df_result, pd.DataFrame)
    assert 'compid' in df_result.columns
    assert 'found' in df_result.columns
