# Ashpool - A Comparison Library
Align and compare data from different sources.

## Installation
How to install
```
pip install ashpool
```
## Quick Start
### 1. Compare
Compare series from "Left Dataframe" and "Right Dataframe" if they each have unique identifiers that match.
```
ashpool.differ(df_l, df_r, left_on='unique_id_l', right_on='unique_id_r', fields_l=['data_l'], fields_r['data_r'], show_diff=True)
```
#### Example
df_l:

|team|player_count|mascot|
|--- |--- |--- |
|Gryffindor|20|Lion|
|The Shire|12|Smeagol|
|Enterprise|1014|Tribble|
|Coruscant|2|Womp Rat|
|Gridpoint|8|Molly|

df_r:

|team|player_count|mascot|
|--- |--- |--- |
|Asgard|3|Hammer|
|Coruscant|4|Womp Rat|
|Gridpoint|80|Rose|
|Gryffindor|20|lion|
|The Shire|12|smeagol|

Results comparing numeric data:
```
ashpool.differ(df_l, df_r, left_on='team', right_on='team', fields_l=['player_count'], fields_r=['player_count'], show_diff=True)
```
|compid|found|player_count_l|player_count_r|player_count_l vs player_count_r|player_count_l - player_count_r|pct_pairs_matched|
|--- |--- |--- |--- |--- |--- |--- |
|CORUSCANT|both|2.0000|4.0000|False|-2.0000|0.0000|
|GRIDPOINT|both|8.0000|80.0000|False|-72.0000|0.0000|
|GRYFFINDOR|both|20.0000|20.0000|True|0.0000|1.0000|
|THE_SHIRE|both|12.0000|12.0000|True|0.0000|1.0000|
|ASGARD|right_only|nan|3.0000|False|nan|0.0000|
|ENTERPRISE|left_only|1,014.0000|nan|False|nan|0.0000|

Results comparing string data:
```
ashpool.differ(df_l, df_r, left_on='team', right_on='team', fields_l=['mascot'], fields_r=['mascot'], show_diff=False)
```
|compid|found|mascot_l|mascot_r|mascot_l vs mascot_r leven_dist|mascot_l vs mascot_r|pct_pairs_matched|
|--- |--- |--- |--- |--- |--- |--- |
|Coruscant|both|Womp Rat|Womp Rat|0.0000|True|1.0000|
|Gridpoint|both|Molly|Rose|4.0000|False|0.0000|
|Gryffindor|both|Lion|lion|1.0000|False|0.0000|
|The Shire|both|Smeagol|smeagol|1.0000|False|0.0000|
|Asgard|right_only|NaN|Hammer|nan|False|0.0000|
|Enterprise|left_only|Tribble|NaN|nan|False|0.0000|

Results show whether the row identifier ('compid') was found in the left, right, or both dataframes. If found in both, a comparison can be made. In the examples above, the two dataframes ('df_l' and 'df_r') are first aligned on ['team'] and then their ['player_count'] compared. ['player_count_l vs player_cout_r] gives an indication of whether an exact match was found.

For comparisons of numeric series, you have the option of calculating the differences and/or ratios for each comparison pair.

In the second example, the comparison is conducted on ['mascot'], which contains string data. For comparison of strings, the Levenshtein distance is calculated as an indication of the difference.

Learn more about this function [here](http://ashpool.readthedocs.io/en/latest/#ashpool.differ).


### 2. Create Unique ID for Each Row
Attach unique ID based on data
```
ashpool.attach_unique_id(df)
```
#### Example
Start with this, a dataframe where none of the series can be used as unique keys individually.

|salutation|first_name|surname|
|--- |--- |--- |
|Mr.|Sam|Chow|
|Mr.|Drew|Chow|
|Mr.|Sam|Williams|
|Mr.|Drew|Williams|
|Mrs.|Sam|Chow|
|Mrs.|Drew|Chow|
|Mrs.|Sam|Williams|
|Mrs.|Drew|Williams|

...adds 'u_id' column, which is a unique identifier that is based upon existing data.

|u_id|salutation|first_name|surname|
|--- |--- |--- |--- |
|MR_SAM_CHOW|Mr.|Sam|Chow|
|MR_DREW_CHOW|Mr.|Drew|Chow|
|MR_SAM_WILLIAMS|Mr.|Sam|Williams|
|MR_DREW_WILLIAMS|Mr.|Drew|Williams|
|MRS_SAM_CHOW|Mrs.|Sam|Chow|
|MRS_DREW_CHOW|Mrs.|Drew|Chow|
|MRS_SAM_WILLIAMS|Mrs.|Sam|Williams|
|MRS_DREW_WILLIAMS|Mrs.|Drew|Williams|

Learn more about this function [here](http://ashpool.readthedocs.io/en/latest/#ashpool.attach_unique_id).

If you would like the specify the series to use to create an id, use [ashpool.attach_temp_id()](http://ashpool.readthedocs.io/en/latest/#ashpool.attach_temp_id).


### 3. Automatically Align Data then Compare
Reconcile a series from "Left Dataframe" vs "Right Dataframe"
```
ashpool.reconcile(df_l, df_r, 'series_in_df_l', 'series_in_df_r')
```

#### Example
It will take the following datasets which do not have unique keys...

df_l:

|salutation|first_name|surname|age_reported|
|---|---|---|---|
|Mr.|Sam|Chow|41|
|Mr.|Drew|Chow|23|
|Mr.|Sam|Williams|24|
|Mr.|Drew|Williams|53|
|Mrs.|Sam|Chow|25|
|Mrs.|Drew|Chow|32|
|Mrs.|Sam|Williams|39|
|Mrs.|Drew|Williams|32|

df_r:

|salutation|first_name|surname|age_real|
|--- |--- |--- |--- |
|Mr.|Sam|Chow|40|
|Mr.|Drew|Chow|30|
|Mrs.|Sam|Chow|35|
|Mrs.|Drew|Chow|30|
|Mr.|Sam|Williams|24|
|Mr.|Drew|Williams|43|
|Mrs.|Sam|Williams|32|
|Mrs.|Drew|Williams|32|

...and align and rows and compare the specified series.
```
ashpool.reconcile(df_l, df_r, 'age_reported', 'age_real')
```
Returns:

|index|compid|found|age_reported|age_real|age_reported vs age_real|age_reported - age_real|pct_pairs_matched|
|--- |--- |--- |--- |--- |--- |--- |--- |
|0|DREW_MRS_CHOW|both|32|30|False|2|0.0000|
|1|DREW_MRS_WILLIAMS|both|32|32|True|0|1.0000|
|2|DREW_MR_CHOW|both|23|30|False|-7|0.0000|
|3|DREW_MR_WILLIAMS|both|53|43|False|10|0.0000|
|4|SAM_MRS_CHOW|both|25|35|False|-10|0.0000|
|5|SAM_MRS_WILLIAMS|both|39|32|False|7|0.0000|
|6|SAM_MR_CHOW|both|41|40|False|1|0.0000|
|7|SAM_MR_WILLIAMS|both|24|24|True|0|1.0000|

Learn more about this function [here](http://ashpool.readthedocs.io/en/latest/#ashpool.reconcile).

## Documentation
Docs at [Read the Docs](http://ashpool.readthedocs.io/en/latest/) (http://ashpool.readthedocs.io/en/latest/)

## License
MIT

## Status
[![Build Status](https://travis-ci.org/cktc/ashpool.svg?branch=master)](https://travis-ci.org/cktc/ashpool)
[![Documentation Status](https://readthedocs.org/projects/ashpool/badge/?version=latest)](http://ashpool.readthedocs.io/en/latest/?badge=latest)