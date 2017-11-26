'''Ashpool'''

import string
import uuid
from itertools import chain, combinations
from operator import itemgetter

import numpy as np
import pandas as pd
from IPython.core.display import display

trantbl = string.maketrans(string.punctuation, ' ' * len(string.punctuation))

def completeness(srs):
    """return completeness score for series - i.e., the percentage of non-null values in a series.

    Arguments:
        srs {pandas.Series}

    Returns:
        float
    """
    return float(srs.notnull().sum()) / srs.shape[0]


def uniqueness(srs):
    '''return uniqueness score for series - i.e., percentage of unique values in series (excl. nulls).

    Arguments:
        srs {pandas.Series}

    Returns:
        float
    '''
    u_scr = float(srs.nunique()) / srs.count()
    if u_scr == 1:
        print '{} is perfectly unique and covers {} of rows'.format(srs.name, float(srs.notnull().sum()) / srs.shape[0])
    return u_scr


def longest_member(srs):
    '''return the max len() of any member in series.

    Arguments:
        srs {pandas.Series}

    Returns:
        float
    '''
    max_len = 0
    for each in srs:
        if len(str(each)) > max_len:
            max_len = len(str(each))
    return max_len


def get_combos(lst):
    """Returns list of combinations of members of list.

    Arguments:
        lst {list} -- List of strings

    Returns:
        list -- List of combinations
    """
    combos = []
    for each in xrange(len(lst)):
        if each > 0:
            combos.extend(combinations(lst, each + 1))
    return list(set(combos))


def mash(dframe, flds=None, keep_zeros=False):
    '''Returns df of non-null and non-zero on flds'''
    flds = [flds] if isinstance(flds, str) else flds
    df_work = dframe.copy()
    for each in flds:
        assert each in df_work.columns, '"{}" is not in df'.format(each)
        df_work = df_work[df_work[each].notnull()]
        if keep_zeros is False:
            df_work = df_work[df_work[each] != 0]
    return df_work


def attach_temp_id(dframe, field_list=None, id_label='tempid', append_uuid=False, prefix=''):
    '''Attach an column with ID created from field_list and optionally add uuid'''
    df_work = dframe.copy()
    df_work['_prefix_'] = prefix.strip()
    df_work['_uuid_'] = ''

    if len(field_list) > 1:
        flds_x = [df_work[x].tolist() for x in field_list[1:]]
        df_work['_work_'] = df_work[field_list[0]].str.cat(flds_x, sep='_', na_rep='nan')
    else:
        flds_x = field_list
        df_work['_work_'] = df_work[field_list]

    if append_uuid:
        df_work['_uuid_'] = pd.Series(
            ['_' + uuid.uuid4().hex[:3].upper() for x in xrange(len(df_work))])
        df_work['_work_'] = df_work['_work_'] + df_work['_uuid_']
    if prefix:
        df_work['_work_'] = df_work['_prefix_'] + '_' + df_work['_work_']
    df_work['_work_'] = df_work['_work_'].apply(make_good_label).str.upper()
    df_work = df_work.drop(['_prefix_', '_uuid_'], axis=1)
    df_work = df_work.rename(columns={'_work_': id_label})
    return df_work


def make_good_label(x_value):
    """Return something that is a better label.

    Arguments:
        x_value {string} -- or something that can be converted to a string
    """
    # trantbl = string.maketrans(string.punctuation, ' ' * len(string.punctuation))
    if isinstance(x_value, unicode):
        x_value = x_value.encode('ascii', 'ignore')
    return '_'.join(str(x_value).translate(trantbl).split()).lower()


def rate_series(dframe):
    '''return ratings of fields for completeness and uniqueness'''
    df_work = pd.DataFrame({'fld': dframe.columns.tolist()}, )
    df_work['obj_type'] = df_work.fld.apply(lambda x: dframe[x].dtype)
    df_work['obj_kind'] = df_work.fld.apply(lambda x: dframe[x].dtype.kind)
    df_work['completeness'] = df_work['fld'].apply(lambda x: completeness(dframe[x]))
    df_work['uniqueness'] = df_work['fld'].apply(lambda x: uniqueness(dframe[x]))
    df_work['longest_member'] = df_work['fld'].apply(lambda x: longest_member(dframe[x]))
    df_work.sort_values(['obj_type', 'completeness', 'uniqueness'], ascending=[False, False, False])
    return df_work


def get_sorted_fields(dframe):
    '''return lists of fields sorted by most_complete, most_unique, and non_object'''
    df_work = rate_series(dframe)
    fltr_om = df_work['obj_kind'].isin(['O', 'M'])
    flds_srt_completeness = df_work[fltr_om].sort_values(
        ['completeness', 'longest_member'], ascending=[False, True])['fld'].tolist()
    flds_srt_uniqueness = df_work[fltr_om].sort_values(
        ['uniqueness', 'completeness', 'longest_member'], ascending=[False, False, True])['fld'].tolist()
    flds_non_object = df_work[~fltr_om]['fld'].tolist()
    return {'most_complete': flds_srt_completeness, 'most_unique': flds_srt_uniqueness, 'non_object': flds_non_object}


def get_unique_fields(dframe, candidate_flds, threshold=1.0, max_member_length=30, show_all=False):
    '''returns list of fields that combine to create an id that has uniqueness >= threshold'''
    summary = []
    flds_work = [x for x in candidate_flds if longest_member(dframe[x]) <= max_member_length]
    for each in get_combos(flds_work):
        df_temp = attach_temp_id(dframe[list(each)], field_list=list(each))  # .copy()
        df_temp['tempid_makeup'] = str(each)
        u_scr = uniqueness(df_temp['tempid'])
        summary.append((str(each), u_scr))
        if u_scr >= threshold:
            print 'Uniqueness: {}'.format(u_scr)
            display(df_temp.head())
            if not show_all:
                return list(each)

    summary.sort(key=itemgetter(1), reverse=True)

    if show_all:
        return summary

    assert summary[0][1] >= threshold, 'Does not meet threshold of {}, best found was {}'.format(threshold, summary[0][1])
    return


def attach_unique_id(dframe, threshold=0.5):
    """Return a new dataframe based on input dframe with unique fields attached.

    Arguments:
        dframe {pandas.DataFrame} -- Source dataframe
        threshold {float} -- Specify how unique 0.0 to 1.0 (most unique)

    Returns:
        pandas.DataFrame -- datatframe with most unique fields attached.
    """
    flds = get_sorted_fields(dframe)
    u_flds = get_unique_fields(dframe, candidate_flds=flds['most_unique'], threshold=threshold)
    print '!!!u_flds:', u_flds
    assert u_flds, 'Did not get valid unique fields'
    df_work = attach_temp_id(dframe, field_list=u_flds, id_label='u_id')
    return df_work[['u_id'] + flds['most_unique'] + sorted(flds['non_object'])]


def coveredness(srs_l, srs_r):
    '''returns percentage of srs_l members that can be covered by srs_r members'''
    return float(srs_l.isin(srs_r).sum()) / srs_l.shape[0]


def jaccard_similarity(srs_l, srs_r):
    """Returns the jaccard similarity between two lists"""
    intersection_card = len(set.intersection(*[set(srs_l), set(srs_r)]))
    union_card = len(set.union(*[set(srs_l), set(srs_r)]))
    return intersection_card / float(union_card)


def oneness(srs_l, srs_r):
    '''TODO'''
    print len(dict(zip(srs_l.tolist(), srs_r.tolist())))
    print len(dict(zip(srs_r.tolist(), srs_l.tolist())))


def has_name_match(srs_l, dframe_r):
    '''Returns True if srs_l name found in dframe_r'''
    if srs_l.name in dframe_r.columns:
        return True
    return False


def get_most_coveredness(srs_l, dframe_r, top_limit=3):
    '''Returns top most series in dframe_r which covers srs_l'''
    res = []
    for each in dframe_r:
        if srs_l.dtype == dframe_r[each].dtype:
            cvd = coveredness(srs_l, dframe_r[each])
            if cvd > 0:
                res.append((each, cvd))
    res.sort(key=itemgetter(1), reverse=True)
    return res[:top_limit]


def check_coveredness(dframe_l, dframe_r):
    '''return ratings of fields for coveredness'''
    df_work = pd.DataFrame({'fld': dframe_l.columns.tolist()}, )
    df_work['obj_type'] = df_work.fld.apply(lambda x: dframe_l[x].dtype)
    df_work['obj_kind'] = df_work.fld.apply(lambda x: dframe_l[x].dtype.kind)
    df_work['has_name_match'] = df_work.fld.apply(lambda x: has_name_match(dframe_l[x], dframe_r))
    df_work['coveredness'] = df_work['fld'].apply(lambda x: get_most_coveredness(dframe_l[x], dframe_r))

    df_work['completeness'] = df_work['fld'].apply(lambda x: completeness(dframe_l[x]))
    df_work['uniqueness'] = df_work['fld'].apply(lambda x: uniqueness(dframe_l[x]))
    df_work['longest_member'] = df_work['fld'].apply(lambda x: longest_member(dframe_l[x]))
    return df_work


def suggest_id_pairs(dframe_l, dframe_r, threshold=0.5, incl_all_dtypes=False, incl_all_pairs=False):
    '''Suggest matching series from two dfs'''
    flds_l = dframe_l.columns.tolist()
    flds_r = dframe_r.columns.tolist()
    pairs = []
    for fld in flds_l:
        dtype_l = dframe_l[fld].dtype.kind
        if incl_all_dtypes or dtype_l == 'O':
            for each in flds_r:
                cvd = coveredness(dframe_l[fld], dframe_r[each])
                if cvd >= threshold:
                    cvd_reversed = coveredness(dframe_r[each], dframe_l[fld])
                    complete_l = completeness(dframe_l[fld])
                    complete_r = completeness(dframe_l[fld])
                    uniq_l = uniqueness(dframe_l[fld])
                    uniq_r = uniqueness(dframe_l[fld])
                    pairs.append((fld, each, cvd, cvd_reversed, complete_l, complete_r, uniq_l, uniq_r))
    df_result = pd.DataFrame(pairs, columns=['fld_l', 'fld_r', 'cover_l', 'cover_r', 'complete_l', 'complete_r', 'uniq_l', 'uniq_r'])
    df_result['id_scr'] = df_result.product(axis=1, numeric_only=True)
    df_result.sort_values(['id_scr'], ascending=False, inplace=True)
    df_result.reset_index(drop=True, inplace=True)
    if incl_all_pairs:
        return df_result
    df_result = df_result.groupby(['fld_l']).first().sort_values(['id_scr'], ascending=False).reset_index()
    return df_result


def cum_uniq(dframe, flds=None):
    '''Return list of uniqueness as tempid is created based on flds'''
    flds_work = [f for f in flds if f in dframe.columns]
    u_res = []

    for each in xrange(len(flds_work)):
        df_temp = attach_temp_id(dframe, field_list=flds_work[:each + 1])
        u_res.append(uniqueness(df_temp['tempid']))
    return u_res


def best_id_pair(dframe_l, dframe_r):
    '''Return df showing which IDs are best for matching two dfs'''
    df_work = suggest_id_pairs(dframe_l, dframe_r)
    df_work['cum_uniq_l'] = cum_uniq(dframe_l, flds=df_work['fld_l'].tolist())
    df_work['cum_uniq_r'] = cum_uniq(dframe_r, flds=df_work['fld_r'].tolist())
    df_work['cum_uniq_l_increment'] = df_work['cum_uniq_l'] - df_work['cum_uniq_l'].shift()
    df_work['cum_uniq_r_increment'] = df_work['cum_uniq_r'] - df_work['cum_uniq_r'].shift()
    return df_work[~((df_work['cum_uniq_l_increment'] == 0) & (df_work['cum_uniq_r_increment'] == 0))]


def reconcile(dframe_l, dframe_r, fields_l=None, fields_r=None, gen_diffs=True):
    """Aligns and compares two dataframes

    Arguments:
        dframe_l {pandas.DataFrame} -- left dataframe
        dframe_r {pandas.DataFrame} -- right dataframe

    Keyword Arguments:
        fields_l {list} -- list of columns names to compare from dframe_l (default: {None})
        fields_r {list} -- list of columns names to compare from dframe_r (default: {None})
        gen_diffs {bool} -- whether or not to include a calculation of the difference in results (default: {True})

    Returns:
        dataframe -- shows results of the comparison
    """
    df_best = best_id_pair(dframe_l, dframe_r)
    print 'Diags:'
    display(df_best)
    df_temp_l = attach_temp_id(dframe_l, field_list=df_best['fld_l'].tolist())
    df_temp_r = attach_temp_id(dframe_r, field_list=df_best['fld_r'].tolist())
    return differ(df_temp_l, df_temp_r, left_on='tempid', right_on='tempid', fields_l=fields_l, fields_r=fields_r, gen_diffs=gen_diffs)


def differ(dframe_l, dframe_r, left_on, right_on, fields_l=None, fields_r=None, gen_diffs=False, return_data=True):
    '''TODO'''
    assert len(fields_l) == len(fields_r), 'Comparison lists not of equal length / None.'

    df_l = dframe_l.rename(columns={left_on: 'compid'}).copy()
    df_r = dframe_r.rename(columns={right_on: 'compid'}).copy()
    fields = zip(fields_l, fields_r)

    final_fields = []
    for each in fields:
        if each[0] == each[1]:
            final_fields.append((each[0] + '_l', each[1] + '_r'))
        else:
            final_fields.append(each)

    ordered_fields = list(chain(*final_fields))
    df_out = pd.merge(df_l[['compid'] + fields_l], df_r[['compid'] + fields_r], how='outer',
                      left_on='compid', right_on='compid', suffixes=['_l', '_r'], indicator='found')
    df_out = df_out[['compid', 'found'] + ordered_fields]

    # Do comparison
    vs_fields = []
    for each in final_fields:
        lbl = each[0] + ' vs ' + each[1]
        try:
            df_out[lbl] = np.isclose(
                df_out[each[0]], df_out[each[1]], rtol=0.000000001, atol=0.0001)
            vs_fields.append(lbl)
        except:
            print 'Cannot compare:', lbl
            print each[0], type(each[0]), each[1], type(each[1])

    df_out['vs_pct'] = sum([df_out[each] for each in vs_fields]) / len(fields)

    # Calc diffs
    if gen_diffs:
        for each in final_fields:
            lbl = each[0] + ' - ' + each[1]
            lbl2 = each[0] + ' / ' + each[1]
            try:
                df_out[lbl] = df_out[each[0]] - df_out[each[1]]
                df_out[lbl2] = df_out[each[0]] / df_out[each[1]]
            except:
                print 'Cannot calc diff:', lbl, lbl2
                print each[0], type(each[0]), each[1], type(each[1])

    # Check if need to return data
    if not return_data:
        df_out = df_out.drop(ordered_fields, axis=1)

    return df_out
