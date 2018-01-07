from __future__ import (absolute_import, division, print_function, unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map,
                      next, oct, open, pow, range, round, str, super, zip)

import datetime
import string

import numpy as np
import pandas as pd
import pytest

from hypothesis import example, given
from hypothesis.extra.pandas import column, data_frames, indexes, series
from hypothesis.strategies import datetimes, lists, sampled_from, text

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
lst_text = lists(elements=text(alphabet=list(string.printable), min_size=4, max_size=10), min_size=1).example()


@given(text(min_size=3))
@example(' BAD label ')
@example(' BAD label !@#$!%!%')
def test_make_good_label(s):
    print(type(make_good_label(s)))
    assert isinstance(make_good_label(s), str)
    assert make_good_label(' BAD label ') == 'bad_label'


@given(series(dtype=np.unicode_))
@example(series(dtype=np.int_).example())
@example(series(dtype=float).example())
@example(series(dtype=bool).example())
def test_completeness(srs_t):
    print(srs_t, srs_t.dtype.kind)
    assert np.isclose(completeness(df_l['flt_1']), 0.833333)
    assert isinstance(completeness(srs_t), float)

def test_mash():
    df_result = mash(df_l, 'flt_1')
    assert df_result.shape[0] == 5
    df_result = mash(df_l, 'int_1')
    assert df_result.shape[0] == 5
    df_result = mash(df_l, 'int_1', keep_zeros=True)
    assert df_result.shape[0] == 6


@given(series(dtype=np.unicode_))
@example(series(dtype=np.int_).example())
@example(series(dtype=bool).example())
def test_uniqueness(srs_t):
    assert np.isclose(uniqueness(df_l['str_1']), 0.166666)
    assert np.isclose(uniqueness(df_l['str_2']), 0.333333)
    assert isinstance(uniqueness(srs_t), float)


@given(series(dtype=np.unicode_), series(dtype=np.unicode_))
@example(series(dtype=np.int_).example(), series(dtype=np.int_).example())
@example(series(dtype=float).example(), series(dtype=float).example())
@example(series(dtype=bool).example(), series(dtype=bool).example())
# @example(series(dtype=np.unicode_).example(), series(dtype=np.unicode_).example())
def test_coveredness(srs_t, srs_u):
    assert np.isclose(coveredness(df_l['int_1'], df_l['int_2']), 0.333333)
    assert isinstance(coveredness(srs_t, srs_u), float)


@given(lists(text(), max_size=11))
def test_get_combos(srs_t):
    assert len(get_combos(df_l['str_2'])) == 13
    assert isinstance(get_combos(srs_t), list)

def test_attach_temp_id():
    assert attach_temp_id(df_l, field_list=['str_1', 'str_2'])['tempid'].tolist() == ['A_BC', 'A_BC', 'A_BC', 'A_DE', 'A_DE', 'A_DE']
    assert isinstance(attach_temp_id(df_l, field_list=['str_1', 'dt_1']), pd.DataFrame)

def test_rate_series():
    df_result = rate_series(df_l)
    assert df_result['fld'].tolist() == df_l.columns.tolist()


@given(data_frames([column('txt_1', elements=sampled_from(lst_text)), column('str_1', dtype=np.unicode_), column('int_1', dtype=np.int_), column('int_2', dtype=np.int_), column('int_3', dtype=np.int_), column('flt_1', dtype=float), column('flt_2', dtype=float), column('dt_1', elements=datetimes())], index=indexes(dtype=np.int_, min_size=1)))
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

# def test_attach_unique_id():
#     df_result = attach_unique_id(df_l, threshold=1)
#     assert df_result['u_id'].tolist() == ['Z_A', 'Y_A', 'X_A', 'W_A', 'V_A', 'NAN_A']

def test_jaccard_similarity():
    assert isinstance(jaccard_similarity(df_l['int_1'], df_l['int_2']), float)

def test_has_name_match():
    assert isinstance(has_name_match(df_l['str_1'], df_r), bool)


@given(data_frames([column('txt_1', elements=sampled_from(lst_text)), column('str_1', dtype=np.unicode_), column('int_1', dtype=np.int_), column('flt_1', dtype=float), column('dt_1', elements=datetimes())], index=indexes(dtype=np.int_, min_size=1)), data_frames([column('txt_1', elements=sampled_from(lst_text)), column('str_1', dtype=np.unicode_), column('int_1', dtype=np.int_), column('flt_1', dtype=float), column('dt_1', elements=datetimes())], index=indexes(dtype=np.int_, min_size=1)))
def test_suggest_id_pairs(df_t, df_u):
    df_result = suggest_id_pairs(df_l, df_r)
    assert isinstance(df_result, pd.DataFrame)
    assert 'id_scr' in df_result.columns
    assert isinstance(suggest_id_pairs(df_t,df_u), pd.DataFrame)

def test_get_most_coveredness():
    assert isinstance(get_most_coveredness(df_l['dt_1'], df_r), list)

def test_check_coveredness():
    df_result = check_coveredness(df_l, df_r)
    assert isinstance(df_result, pd.DataFrame)
    assert 'coveredness' in df_result.columns

def test_cum_uniq():
    assert isinstance(cum_uniq(df_l, df_l.columns.tolist()), list)


@given(data_frames([column('txt_1', elements=sampled_from(lst_text)), column('str_1', dtype=np.unicode_), column('int_1', dtype=np.int_), column('flt_1', dtype=float), column('dt_1', elements=datetimes())], index=indexes(dtype=np.int_, min_size=1)), data_frames([column('txt_1', elements=sampled_from(lst_text)), column('str_1', dtype=np.unicode_), column('int_1', dtype=np.int_), column('flt_1', dtype=float), column('dt_1', elements=datetimes())], index=indexes(dtype=np.int_, min_size=1)))
def test_best_id_pair(df_t, df_u):
    assert isinstance(best_id_pair(df_l, df_r), pd.DataFrame)
    assert isinstance(best_id_pair(df_t, df_u, threshold=0.01), pd.DataFrame)


@given(data_frames([column('txt_1', elements=sampled_from(lst_text)), column('str_1', dtype=np.unicode_), column('int_1', dtype=np.int_), column('flt_1', dtype=float), column('dt_1', elements=datetimes())], index=indexes(dtype=np.int_, min_size=1)), data_frames([column('txt_1', elements=sampled_from(lst_text)), column('str_1', dtype=np.unicode_), column('int_1', dtype=np.int_), column('flt_1', dtype=float), column('dt_1', elements=datetimes())], index=indexes(dtype=np.int_, min_size=1)))
def test_reconcile(df_t, df_u):
    df_result = reconcile(df_l, df_r, fields_l=['flt_1'], fields_r=['flt_1'])
    assert isinstance(df_result, pd.DataFrame)
    assert 'compid' in df_result.columns
    assert 'found' in df_result.columns
    assert isinstance(reconcile(df_t, df_u, fields_l=['flt_1'], fields_r=['flt_1']), pd.DataFrame)


@given(data_frames([column('txt_1', elements=sampled_from(lst_text)), column('str_1', dtype=np.unicode_), column('int_1', dtype=np.int_), column('flt_1', dtype=float), column('dt_1', elements=datetimes())], index=indexes(dtype=np.int_, min_size=1)), data_frames([column('txt_1', elements=sampled_from(lst_text)), column('str_1', dtype=np.unicode_), column('int_1', dtype=np.int_), column('flt_1', dtype=float), column('dt_1', elements=datetimes())], index=indexes(dtype=np.int_, min_size=1)))
def test_differ(df_t, df_u):
    df_result = differ(df_t, df_u, left_on='txt_1', right_on='txt_1', fields_l=['str_1','flt_1'], fields_r=['str_1','flt_1'])
    assert isinstance(df_result, pd.DataFrame)
    assert 'compid' in df_result.columns
    assert 'found' in df_result.columns
