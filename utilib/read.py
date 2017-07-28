# Anthony Ho <anthony.ho@energy.ca.gov>
# Last update 7/26/2017
"""
Python module for reading utility data
"""

import numpy as np
import pandas as pd


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


def read_costar(file, usecols=None, dtype=None, nrows=None,
                filter_multiple=False, **kwargs):
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

    # Read file
    data = pd.read_csv(file,
                       usecols=usecols, dtype=dtype,
                       encoding=encoding, engine=engine,
                       nrows=nrows, **kwargs)
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
    # Extract only the 5-digit zip codes
    if 'zip' in data:
        data['zip'] = data['zip'].str[:5]
    # Typecast dates
    if 'Last Sale Date' in data:
        data['Last Sale Date'] = pd.to_datetime(data['Last Sale Date'],
                                                format='%m/%d/%Y')

    # Filter buildings that belong to the same address if selected
    if filter_multiple:
        group_keys = ['address', 'city', 'zip']
        num_bldg = data.groupby(group_keys).size()
        index_pf = num_bldg[num_bldg == 1].index
        data = data.set_index(group_keys).loc[index_pf].reset_index()

    return data


def read_cis(file, iou, usecols=None, dtype=None, nrows=None, **kwargs):
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
    elif iou == 'sdge':
        badcols = {'keyAcctID': 'keyAcctId',
                   'fuel': 'FUEL'}
        dropcols = ['corpNAICS', 'corpNaicsBldg']
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
                      nrows=nrows, **kwargs)
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


def read_bills(file, fuel, iou,
               usecols=None, dtype=None, nrows=None, **kwargs):
    # Define default columns to read from the CSV file
    if usecols is None:
        if fuel == 'elec':
            usecols = ['keyAcctID', 'premiseID', 'rate',
                       'readDate', 'lastReadDate', 'readDays',
                       'kWh', 'kWhOn', 'kWhSemi', 'kWhOff', 'billAmnt']
        elif fuel == 'gas':
            usecols = ['keyAcctID', 'premiseID', 'rate',
                       'readDate', 'lastReadDate', 'readDays',
                       'Therms', 'billAmnt']
    # Define the default data type of each column
    if dtype is None:
        if fuel == 'elec':
            dtype = {'keyAcctID': str,
                     'premiseID': str,
                     'rate': str,
                     'readDate': str,
                     'lastReadDate': str,
                     'readDays': int,
                     'kWh': np.float64,
                     'kWhOn': np.float64,
                     'kWhSemi': np.float64,
                     'kWhOff': np.float64,
                     'billAmnt': str}
        elif fuel == 'gas':
            dtype = {'keyAcctID': str,
                     'premiseID': str,
                     'rate': str,
                     'readDate': str,
                     'lastReadDate': str,
                     'readDays': int,
                     'Therms': np.float64,
                     'billAmnt': str}
    # Miscell options
    thousands = ','
    encoding = 'ISO-8859-1'
    engine = 'c'

    # Customize the columns that might have been misspelt/missing from each IOU
    if iou == 'pge':
        badcols = {'keyAcctID': 'keyacctid'}
        dropcols = None
    elif iou == 'sce':
        badcols = {'keyAcctID': 'keyacctid'}
        dropcols = ['premiseID']
    else:
        badcols = None
        dropcols = None

    # Modify/drop the misspelt/missing columns
    if badcols:
        usecols, dtype = _modify_fields(usecols, dtype, badcols)
    if dropcols:
        usecols, dtype = _drop_fields(usecols, dtype, dropcols)

    # Read file
    bills = pd.read_csv(file,
                        usecols=usecols, dtype=dtype,
                        thousands=thousands, encoding=encoding, engine=engine,
                        nrows=nrows, **kwargs)
    # Drop duplicates
    # bills = bills.drop_duplicates()

    # Rename misspelt columns back to the standardized spelling
    if badcols:
        bills = bills.rename(columns=_rev_dict(badcols))

    # Pad keyAcctID
    bills['keyAcctID'] = bills['keyAcctID'].apply(lambda x:
                                                  '{0:010d}'.format(int(x)))

    # Typecast dates
    for col in ['readDate', 'lastReadDate']:
        if col in bills:
            bills[col] = pd.to_datetime(bills[col], format='%m/%d/%Y')

    # Extract dollar amount
    if 'billAmnt' in bills:
        bills['billAmnt'] = bills['billAmnt'].str.replace('$', '')
        bills['billAmnt'] = bills['billAmnt'].str.replace(',', '')
        bills['billAmnt'] = bills['billAmnt'].astype(float)

    return bills
