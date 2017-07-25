# Anthony Ho <anthony.ho@energy.ca.gov>
# Last update 7/24/2017
"""
Python module for reading utility data
"""

import numpy as np
import pandas as pd

# To-do's
# 1. allow reading other types of bills (gas and residential)

bool_dict = {'Y': True,
             'N': False,
             'y': True,
             'n': False,
             '1': True,
             '0': False,
             '.': np.nan}


def _modify_fields(usecols, dtype, badcols):
    for col in badcols:
        usecols = [badcols[col] if uc == col else uc for uc in usecols]
        try:
            dtype[badcols[col]] = dtype.pop(col)
        except KeyError:
            pass
    return usecols, dtype


def _drop_fields(usecols, dtype, dropcols):
    for col in dropcols:
        try:
            usecols.remove(col)
        except ValueError:
            pass
        try:
            del dtype[col]
        except KeyError:
            pass
    return usecols, dtype


def _rev_dict(d):
    return {v: k for k, v in d.items()}


def _filter_valid_id(df, col):
    df = df[(df[col].str.isnumeric().replace({np.nan: False})) &
            (df[col] != '0') &
            (df[col] != 0)]
    return df


def read_costar(file, usecols=None, dtype=None, nrows=None):
    # Define default columns to read from the CSV file
    if usecols is None:
        usecols = ['PropertyID',
                   'Building Address', 'City', 'Zip', 'County Name',
                   'Longitude', 'Latitude',
                   'PropertyType', 'Secondary Type', 'Building Status',
                   'Year Built', 'Year Renovated', 'Vacancy %',
                   'Number Of Stories', 'Rentable Building Area',
                   'Energy Star', 'LEED Certified', 'Last Sale Date']
    # Define the default data type of each column
    if dtype is None:
        dtype = {'PropertyID': str,
                 'Building Address': str,
                 'City': str,
                 'Zip': str,
                 'County Name': str,
                 'Longitude': np.float64,
                 'Latitude': np.float64,
                 'PropertyType': str,
                 'Secondary Type': str,
                 'Building Status': str,
                 'Year Built': np.float64,
                 'Year Renovated': np.float64,
                 'Vacancy %': np.float64,
                 'Number Of Stories': np.float64,
                 'Rentable Building Area': np.float64,
                 'Energy Star': str,
                 'LEED Certified': str,
                 'Last Sale Date': str}
    # Miscell options
    encoding = 'iso-8859-1'
    engine = 'c'

    # Reaf file
    data = pd.read_csv(file,
                       usecols=usecols, dtype=dtype,
                       encoding=encoding, engine=engine,
                       nrows=nrows)
    # Drop duplicates
    data = data.drop_duplicates()

    # Standardize address columns spelling for easier merging
    data = data.rename(columns={'Building Address': 'address',
                                'City': 'city',
                                'Zip': 'zip',
                                'County Name': 'county'})

    # Standardize the entries of address, city and county to upper case
    for col in ['address', 'city', 'county']:
        if col in data:
            data[col] = data[col].str.upper()
    if 'zip' in data:
        data['zip'] = data['zip'].str[:5]
    if 'Last Sale Date' in data:
        data['Last Sale Date'] = pd.to_datetime(data['Last Sale Date'],
                                                format='%m/%d/%Y')

    return data


def read_cis(file, iou, usecols=None, dtype=None, nrows=None):
    # Define default columns to read from the CSV file
    if usecols is None:
        usecols = ['iou', 'fuel',
                   'keyAcctID', 'premiseID', 'siteID', 'nrfSiteID', 'meterNum',
                   'serviceAddress', 'serviceCity', 'serviceZip',
                   'geoID', 'geoLat', 'geoLong',
                   'censusBlock', 'censusCounty', 'censusTract',
                   'premNAICS', 'premNaicsBldg', 'corpNAICS', 'corpNaicsBldg',
                   'CSSnaicsBldg',
                   'NetMeter', 'BenchmarkFlag',
                   'acctProg1012Flag', 'acctProg1314Flag', 'acctProg2015Flag']
    # Define the default data type of each column
    if dtype is None:
        dtype = {'iou': str,
                 'fuel': str,
                 'keyAcctID': str,
                 'premiseID': str,
                 'siteID': str,
                 'nrfSiteID': str,
                 'meterNum': str,
                 'serviceAddress': str,
                 'serviceCity': str,
                 'serviceZip': str,
                 'geoID': np.float64,
                 'geoLat': np.float64,
                 'geoLong': np.float64,
                 'censusBlock': np.float64,
                 'censusCounty': np.float64,
                 'censusTract': np.float64,
                 'premNAICS': str,
                 'premNaicsBldg': str,
                 'corpNAICS': str,
                 'corpNaicsBldg': str,
                 'CSSnaicsBldg': str,
                 'NetMeter': str,
                 'BenchmarkFlag': str,
                 'acctProg1012Flag': str,
                 'acctProg1314Flag': str,
                 'acctProg2015Flag': str}
    # Miscell options
    thousands = ','
    encoding = 'ISO-8859-1'
    engine = 'c'

    # Customize the columns that might have been misspelt/missing from each IOU
    if iou == 'pge':
        badcols = {'keyAcctID': 'keyAcctId',
                   'fuel': 'Fuel',
                   'NetMeter': 'NETMETER'}
        dropcols = None
    elif iou == 'sce':
        badcols = None
        dropcols = ['CECClimateZone']
    elif iou == 'sdge':
        badcols = {'keyAcctID': 'keyAcctId',
                   'fuel': 'FUEL'}
        dropcols = ['corpNAICS', 'corpNaicsBldg', 'CECClimateZone']
    else:
        badcols = None
        dropcols = None

    # Modify/drop the misspelt/missing columns
    if badcols:
        usecols, dtype = _modify_fields(usecols, dtype, badcols)
    if dropcols:
        usecols, dtype = _drop_fields(usecols, dtype, dropcols)

    # Read file
    cis = pd.read_csv(file,
                      usecols=usecols, dtype=dtype,
                      thousands=thousands, encoding=encoding, engine=engine,
                      nrows=nrows)
    # Drop duplicates (SCE data has a lot of those)
    cis = cis.drop_duplicates()
    # Replace '.' as python nan
    cis = cis.replace({'.': np.nan})

    # Rename misspelt columns back to the standardized spelling
    if badcols:
        cis = cis.rename(columns=_rev_dict(badcols))
    # Standardize address columns spelling for easier merging
    cis = cis.rename(columns={'serviceAddress': 'address',
                              'serviceCity': 'city',
                              'serviceZip': 'zip'})
    # Standardize the entries of address and city to upper case
    for col in ['address', 'city']:
        if col in cis:
            cis[col] = cis[col].str.upper()

    # Drop rows that have bad keyAcctID
    for col in ['keyAcctID', 'zip']:
        if col in cis:
            cis = _filter_valid_id(cis, col)

    # Map yes/no columns to boolean
    for col in ['NetMeter', 'BenchmarkFlag', 'acctProg1012Flag',
                'acctProg1314Flag', 'acctProg2015Flag']:
        if col in cis:
            cis[col] = cis[col].map(bool_dict)

    return cis


def read_bills(file, bill_type='elec', nrows=None):
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
    encoding = 'ISO-8859-1'
    engine = 'c'

    bills = pd.read_csv(file,
                        usecols=usecols, dtype=dtype,
                        thousands=thousands, encoding=encoding, engine=engine,
                        nrows=nrows)
    bills['readDate'] = pd.to_datetime(bills['readDate'], format='%m/%d/%Y')
    bills['lastReadDate'] = pd.to_datetime(bills['lastReadDate'],
                                           format='%m/%d/%Y')
    return bills
