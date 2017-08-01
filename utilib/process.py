# Anthony Ho <anthony.ho@energy.ca.gov>
# Last update 7/31/2017
"""
Python module for processing utility data
"""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin, MonthEnd, DateOffset
import calendar
#from censusgeocode import CensusGeocode


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
    col_name = 'EUI_' + fuel
    eui = total_energy.div(df['cis']['Rentable Building Area'], axis=0)
    eui = pd.concat({col_name: eui}, axis=1)
    return pd.concat([df, eui], axis=1)


def get_climate_zones(df, cz_file):
    cz = pd.read_csv(cz_file, dtype={'zip': str, 'cz': str})
    return df.merge(cz, on='zip', how='left')


#def _get_geocode_single(row):
#    cg = CensusGeocode()
    
#def get_census_()


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
