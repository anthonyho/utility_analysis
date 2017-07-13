# Anthony Ho <anthony.ho@energy.ca.gov>
# Last update 7/12/2017
"""
Python module for processing utility data
"""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin, MonthEnd, DateOffset
import calendar


def read_bills(file, bill_type='elec', test=True):
    usecols = ['keyacctid', 'premiseID', 'rate',
               'readDate', 'lastReadDate', 'readDays',
               'kWh', 'kWhOn', 'kWhSemi', 'kWhOff']
    dtype = {'keyacctid': str,
             'premiseID': str,
             'readDate': str,
             'lastReadDate': str,
             'readDays': int,
             'kWh': np.float16,
             'kWhOn': np.float16,
             'kWhSemi': np.float16,
             'kWhOff': np.float16}
    thousands = ','
    encoding = "ISO-8859-1"
    engine = 'c'
    if test:
        nrows = 10000

    bills = pd.read_csv(file,
                        usecols=usecols, dtype=dtype,
                        thousands=thousands, encoding=encoding, engine=engine,
                        nrows=nrows)
    bills['readDate'] = pd.to_datetime(bills['readDate'], format='%m/%d/%Y')
    bills['lastReadDate'] = pd.to_datetime(bills['lastReadDate'], format='%m/%d/%Y')
    return bills


def calendarize(df, group_keys=['keyacctid', 'premiseID'],
                list_fields=['kWh', 'kWhOn']):
    new_df = df.groupby(group_keys).apply(_calendarize_group, list_fields)
    return new_df.reset_index().drop('level_2', axis=1)


def _calendarize_group(group, list_fields=['kWh', 'kWhOn']):
    expanded_rows = []
    for row in group.itertuples():
        expanded_rows.append(_expand_row(row, list_fields))
    expanded_group = pd.concat(expanded_rows, axis=0, ignore_index=True)
    calendarized_group = expanded_group.groupby('months').sum().reset_index()
    return calendarized_group


def _expand_row(row, list_fields):
    start_date = row.lastReadDate
    end_date = row.readDate
    n_months = ((end_date.year - start_date.year) * 12 +
                end_date.month - start_date.month + 1)
    list_months = [(start_date + DateOffset(months=i)).strftime('%Y-%m')
                   for i in range(0, n_months)]
    days_in_months = _get_days_in_months(start_date, end_date,
                                         n_months, list_months)
    frac = days_in_months / row.readDays
    expanded_row = pd.DataFrame({'months': list_months})
    expanded_row['days'] = days_in_months
    expanded_row['frac'] = frac
    for field in list_fields:
        expanded_row[field] = getattr(row, field) * frac
    return expanded_row


def _get_days_in_months(start_date, end_date, n_months, list_months):
    if n_months == 1:
        days_in_months = np.array([(end_date - start_date).days])
    else:
        days_in_month_1 = ((start_date + MonthEnd()) - start_date).days
        days_in_month_n = (end_date - (end_date - MonthBegin())).days + 1
        days_in_months = [days_in_month_1]
        for month in list_months[1:-1]:
            Y, m = list(map(int, month.split("-")))
            days_in_months.append(calendar.monthrange(Y, m)[1])
        days_in_months.append(days_in_month_n)
    return np.array(days_in_months)
