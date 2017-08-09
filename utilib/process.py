# Anthony Ho <anthony.ho@energy.ca.gov>
# Last update 8/8/2017
"""
Python module for processing utility data
"""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin, MonthEnd, DateOffset
import calendar
from scipy.stats import linregress
#from censusgeocode import CensusGeocode


#def _get_geocode_single(row):
#    cg = CensusGeocode()
    
#def get_census_()


def assign_bldg_type(df, bldg_type_file):
    bldg_type_mapping = pd.read_csv(bldg_type_file)
    return df.merge(bldg_type_mapping,
                    how='left',
                    on=['PropertyType', 'Secondary Type',
                        'USE_CODE_STD_CTGR_DESC', 'USE_CODE_STD_DESC'])


def get_climate_zones(df, cz_file):
    cz = pd.read_csv(cz_file, dtype={'zip': str, 'cz': str})
    return df.merge(cz, on='zip', how='left')


def merge_building_cis_data(bldg_data, cis, merge_on_range=True):
    if merge_on_range:
        range_addr_map = _expand_range_addr(bldg_data)
        cis = cis.merge(range_addr_map,
                        left_on=['address', 'city', 'zip'],
                        right_on=['indiv_address', 'city', 'zip'],
                        how='left')
        ind = cis['range_address'].notnull()
        cis.loc[ind, 'address'] = cis.loc[ind, 'range_address']
        cis = cis.drop(['indiv_address', 'range_address'], axis=1)
    bldg_cis = pd.merge(bldg_data, cis,
                        how='inner', on=['address', 'city', 'zip'])
    return bldg_cis


def _expand_range_addr(df):
    address = df['address']
    regex = r"^[0-9]+-[0-9]+$"
    ind = address.str.split(pat=' ', n=1).str[0].str.contains(regex)
    df_range = df[ind]
    list_expanded_df = []
    for (i, row) in df_range.iterrows():
        list_expanded_df.append(_expand_range_addr_single(row))
    return pd.concat(list_expanded_df, axis=0, ignore_index=True)


def _expand_range_addr_single(row):
    address = row['address']
    city = row['city']
    zipcodes = row['zip']
    st_num, st_name = address.split(sep=' ', maxsplit=1)
    st_num_start, st_num_end = st_num.split(sep='-', maxsplit=1)
    st_num_start = int(st_num_start)
    st_num_end = int(st_num_end)
    diff = st_num_end - st_num_start
    if (diff % 2 == 0) and (diff > 0):
        list_st_num = [str(i) for i in range(st_num_start, st_num_end + 2, 2)]
        num_addr = len(list_st_num)
        df = pd.DataFrame({'range_address': [address] * num_addr,
                           'indiv_address': list_st_num,
                           'city': [city] * num_addr,
                           'zip': [zipcodes] * num_addr})
        df['indiv_address'] = df['indiv_address'] + ' ' + st_name
        return df
    else:
        return None


def calendarize(df, group_keys=['keyAcctID', 'premiseID'],
                list_fields=['kWh'],
                keep_cols=[]):
    new_df = df.groupby(group_keys).apply(_calendarize_group,
                                          list_fields, keep_cols)
    return new_df.reset_index(drop=True, level=-1).reset_index()


def _calendarize_group(group, list_fields, keep_cols):
    expanded_rows = []
    for row in group.itertuples():
        expanded_rows.append(_expand_row(row, list_fields))
    expanded_group = pd.concat(expanded_rows, axis=0, ignore_index=True)
    calendarized_group = expanded_group.groupby('yr_mo').sum().reset_index()
    for col in reversed(keep_cols):
        calendarized_group.insert(0, col, group.iloc[0][col])
    return calendarized_group


def _expand_row(row, list_fields):
    start_date = row.lastReadDate
    end_date = row.readDate
    n_months = ((end_date.year - start_date.year) * 12 +
                end_date.month - start_date.month + 1)
    list_yr_mo = [(start_date + DateOffset(months=i)).strftime('%Y-%m')
                  for i in range(0, n_months)]
    days_in_months = _get_days_in_months(start_date, end_date,
                                         n_months, list_yr_mo)
    frac = days_in_months / row.readDays
    expanded_row = pd.DataFrame({'yr_mo': list_yr_mo})
    for field in list_fields:
        expanded_row[field] = getattr(row, field) * frac
    return expanded_row


def _get_days_in_months(start_date, end_date, n_months, list_yr_mo):
    if n_months == 1:
        days_in_months = np.array([(end_date - start_date).days])
    else:
        days_in_month_1 = ((start_date + MonthEnd()) - start_date).days
        days_in_month_n = (end_date - (end_date - MonthBegin())).days + 1
        days_in_months = [days_in_month_1]
        for month in list_yr_mo[1:-1]:
            Y, m = list(map(int, month.split("-")))
            days_in_months.append(calendar.monthrange(Y, m)[1])
        days_in_months.append(days_in_month_n)
    return np.array(days_in_months)


def compute_EUI(df, fuel='tot'):
    kWh_to_kBTU = 3.41214
    therms_to_kBTU = 100
    if fuel == 'tot':
        elec_kbtu = df['kWh'] * kWh_to_kBTU
        gas_kbtu = df['Therms'] * therms_to_kBTU
        total_energy = elec_kbtu.add(gas_kbtu, fill_value=0)
    elif fuel == 'elec':
        total_energy = df['kWh'] * kWh_to_kBTU
    elif fuel == 'gas':
        total_energy = df['Therms'] * therms_to_kBTU
    else:
        raise ValueError('{} not supported'.format(fuel))
    field_name = 'EUI_' + fuel
    eui = total_energy.div(df['cis']['building_area'], axis=0)
    eui = pd.concat({field_name: eui}, axis=1)
    return pd.concat([df, eui], axis=1)


def compute_avg_monthly(df, field, year_range=None):
    # Make data into multi-index df for grouping by years and months
    data = df[field].copy()
    data.columns = pd.MultiIndex.from_tuples([tuple(yr_mo.split('-'))
                                              for yr_mo in data.columns])
    # Get only data from the year if specified
    if year_range:
        start_year = int(year_range[0])
        end_year = int(year_range[1])
        list_year = [str(year) for year in range(start_year, end_year + 1)]
        data = data[list_year]
        field_name = field + '_mo_avg_' + str(start_year) + '_' + str(end_year)
    else:
        field_name = field + '_mo_avg'

    avg_monthly = data.groupby(level=1, axis=1).mean()
    avg_monthly = pd.concat({field_name: avg_monthly}, axis=1)
    return pd.concat([df, avg_monthly], axis=1)


def compute_annual_total(df, field, year, as_series=False):
    # Tally up all 12 months' of data given a year
    year = str(year)
    yr_mo = [col for col in df[field].columns if col[0:4] == year]
    # note that it doesn't make sense to allow NA's when computing total
    annual_total = df[field][yr_mo].sum(axis=1, skipna=False)
    # Return a standalone series or insert back into df
    if as_series:
        return annual_total
    else:
        col_name = field + '_' + year
        annual_total = pd.concat({col_name: annual_total}, axis=1)
        annual_total = pd.concat({'summary': annual_total}, axis=1)
        return pd.concat([df, annual_total], axis=1)


def compute_avg_annual_total(df, field, year_range,
                             skipna=True, as_series=False):
    # Compute annual totals for the years within date range
    start_year = int(year_range[0])
    end_year = int(year_range[1])
    annual_totals = [compute_annual_total(df, field, year, as_series=True)
                     for year in range(start_year, end_year + 1)]
    # Average over all annual totals
    annual_avg = pd.concat(annual_totals, axis=1).mean(axis=1, skipna=skipna)
    # Return a standalone series or insert back into df
    if as_series:
        return annual_avg
    else:
        col_name = field + '_avg_' + str(start_year) + '_' + str(end_year)
        annual_avg = pd.concat({col_name: annual_avg}, axis=1)
        annual_avg = pd.concat({'summary': annual_avg}, axis=1)
        return pd.concat([df, annual_avg], axis=1)


def compute_all_annual_totals(df, field):
    # Get all year-month within varoable
    yr_mo = df[field].columns.sort_values()
    # Make sure getting data from full years only
    if yr_mo[0][-2:] == '01':
        start_year = int(yr_mo[0][0:4])
    else:
        start_year = int(yr_mo[0][0:4]) + 1
    if yr_mo[-1][-2:] == '12':
        end_year = int(yr_mo[-1][0:4])
    else:
        end_year = int(yr_mo[-1][0:4]) - 1
    # Compute annual totals for all full years
    for year in range(start_year, end_year + 1):
        df = compute_annual_total(df, field, year, as_series=False)
    return df


def compute_trend(df, field, year_range, min_sample=2, as_series=False):
    # To be completed
    start_year = int(year_range[0])
    end_year = int(year_range[1])
    list_col = [field + '_' + str(year)
                for year in range(start_year, end_year + 1)]
    x = np.array([year for year in range(start_year, end_year + 1)])
    y = df['summary'][list_col]
    # Fit
    coeff = pd.DataFrame(y.apply(_fit_row, axis=1,
                                 x=x, min_sample=min_sample).tolist())
    col_prefix = field + '_fit_' + str(start_year) + '_' + str(end_year)
    coeff = coeff.rename(columns={0: col_prefix + '_slope',
                                  1: col_prefix + '_incpt'})
    # Return a standalone series (of slope) or insert back into df
    if as_series:
        return coeff[col_prefix + '_slope']
    else:
        coeff = pd.concat({'summary': coeff}, axis=1)
        return pd.concat([df, coeff], axis=1)


def _fit_row(y, x, min_sample=2):
    mask = ~np.isnan(y)
    if mask.sum() >= min_sample:
        x_masked = x[mask]
        y_masked = y[mask]
        result = linregress(x_masked, y_masked)
        return result.slope, result.intercept
    else:
        return (np.nan, np.nan)
