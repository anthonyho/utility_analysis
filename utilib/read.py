# Anthony Ho <anthony.ho@energy.ca.gov>
# Last update 7/31/2017
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


def pad_digits(x, width):
    if pd.notnull(x):
        return '{0:0{1}d}'.format(int(x), width)
    else:
        return x


def read_dmp_multiple(list_files, **kwargs):
    list_data = []
    for file in list_files:
        list_data.append(read_dmp(file, **kwargs))
    data = pd.concat(list_data,
                     axis=0, join='outer', ignore_index=True)
    data = data.drop_duplicates(subset=['address', 'city', 'zip'])
    return data


def read_dmp(file, usecols=None, dtype=None, nrows=None,
             filter_multiple=False, drop_no_st_num=True, **kwargs):
    # Define default columns to read from the CSV file
    if usecols is None:
        usecols = ['APN',
                   'SITE_ADDR', 'SITE_CITY', 'SITE_ZIP', 'COUNTY',
                   'LONGITUDE', 'LATITUDE', 'SITE_HOUSE_NUMBER',
                   'USE_CODE_STD_CTGR_DESC', 'USE_CODE_STD_DESC',
                   'YR_BLT', 'DATE_TRANSFER',
                   'BUILDING_SQFT', 'LAND_SQFT']
    # Define the default data type of each column
    if dtype is None:
        dtype = {'APN': str,
                 'SITE_ADDR': str,
                 'SITE_CITY': str,
                 'SITE_ZIP': str,
                 'COUNTY': str,
                 'LONGITUDE': np.float64,
                 'LATITUDE': np.float64,
                 'SITE_HOUSE_NUMBER': str,
                 'USE_CODE_STD_CTGR_DESC': str,
                 'USE_CODE_STD_DESC': str,
                 'YR_BLT': np.float64,
                 'DATE_TRANSFER': str,
                 'BUILDING_SQFT': np.float64,
                 'LAND_SQFT': np.float64}
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

    # Standardize columns spelling for easier merging
    data = data.rename(columns={'APN': 'PropertyID',
                                'SITE_ADDR': 'address',
                                'SITE_CITY': 'city',
                                'SITE_ZIP': 'zip',
                                'COUNTY': 'county',
                                'LONGITUDE': 'Longitude',
                                'LATITUDE': 'Latitude',
                                'YR_BLT': 'year_built',
                                'DATE_TRANSFER': 'date_transfer',
                                'BUILDING_SQFT': 'building_area',
                                'LAND_SQFT': 'land_area'})

    # Drop entries that have empty address/city/zip
    for col in ['address', 'city', 'county']:
        if col in data:
            data = data.dropna(subset=[col], axis=0)
    # Standardize the entries of address, city and county to upper case
    for col in ['address', 'city', 'county']:
        if col in data:
            data[col] = data[col].str.upper()
    # Extract only the 5-digit zip codes
    if 'zip' in data:
        data['zip'] = data['zip'].str[:5]
    # Typecast dates
    if 'date_transfer' in data:
        data['date_transfer'] = data['date_transfer'].str.split(' ').str[0]
        data['date_transfer'] = pd.to_datetime(data['date_transfer'],
                                               format='%m/%d/%Y')

    # Get rid of entries that have no street number
    if drop_no_st_num:
        data = data[data['SITE_HOUSE_NUMBER'].notnull()]

    # Filter buildings that belong to the same address if selected
    if filter_multiple:
        group_keys = ['address', 'city', 'zip']
        num_bldg = data.groupby(group_keys).size()
        index_pf = num_bldg[num_bldg == 1].index
        data = data.set_index(group_keys).loc[index_pf].reset_index()

    return data


def read_costar_multiple(list_files, **kwargs):
    list_data = []
    for file in list_files:
        list_data.append(read_costar(file, **kwargs))
    data = pd.concat(list_data,
                     axis=0, join='outer', ignore_index=True)
    data = data.drop_duplicates(subset=['address', 'city', 'zip'])
    return data


def read_costar(file, usecols=None, dtype=None, nrows=None,
                filter_multiple=False, **kwargs):
    # Define default columns to read from the CSV file
    if usecols is None:
        usecols = ['PropertyID',
                   'Building Address', 'City', 'Zip', 'County Name',
                   'Longitude', 'Latitude',
                   'PropertyType', 'Secondary Type', 'Building Status',
                   'Year Built', 'Year Renovated', 'Last Sale Date',
                   'Number Of Stories', 'Rentable Building Area', 'Vacancy %',
                   'Energy Star', 'LEED Certified']
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
                 'Last Sale Date': str,
                 'Number Of Stories': np.float64,
                 'Rentable Building Area': np.float64,
                 'Vacancy %': np.float64,
                 'Energy Star': str,
                 'LEED Certified': str,
                 }
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
                                'County Name': 'county',
                                'Year Built': 'year_built',
                                'Year Renovated': 'year_renovated',
                                'Last Sale Date': 'date_transfer',
                                'Rentable Building Area': 'building_area'})

    # Drop entries that have empty address/city/zip
    for col in ['address', 'city', 'county']:
        if col in data:
            data = data.dropna(subset=[col], axis=0)
    # Standardize the entries of address, city and county to upper case
    for col in ['address', 'city', 'county']:
        if col in data:
            data[col] = data[col].str.upper()
    # Extract only the 5-digit zip codes
    if 'zip' in data:
        data['zip'] = data['zip'].str[:5]
    # Typecast dates
    if 'date_transfer' in data:
        data['date_transfer'] = pd.to_datetime(data['date_transfer'],
                                               format='%m/%d/%Y')

    # Filter buildings that belong to the same address if selected
    if filter_multiple:
        group_keys = ['address', 'city', 'zip']
        num_bldg = data.groupby(group_keys).size()
        index_pf = num_bldg[num_bldg == 1].index
        data = data.set_index(group_keys).loc[index_pf].reset_index()

    return data


def _read_cis_scg(cis_file, addr_file, info_file, nrows=None, **kwargs):
    # Define columns to read from the CSV file and their datatypes
    usecols_cis = ['BA_ID', 'GNN_ID', 'MTR_ID',
                   'SADDR', 'SCITY', 'SZIP']
    dtype_cis = {'BA_ID': str,
                 'GNN_ID': str,
                 'MTR_ID': str,
                 'SADDR': str,
                 'SCITY': str,
                 'SZIP': str}
    usecols_addr = ['BA_ID', 'GEO_X_NB', 'GEO_Y_NB']
    dtype_addr = {'BA_ID': str,
                  'GEO_X_NB': np.float64,
                  'GEO_Y_NB': np.float64}
    usecols_info = ['BA_ID', 'NAICS']
    dtype_info = {'BA_ID': str,
                  'NAICS': str}
    # Miscell options
    thousands = ','
    encoding = 'ISO-8859-1'
    engine = 'c'

    # Read files and merge into single dataframe
    cis = pd.read_csv(cis_file,
                      usecols=usecols_cis, dtype=dtype_cis,
                      thousands=thousands, encoding=encoding, engine=engine,
                      nrows=nrows, **kwargs)
    addr = pd.read_csv(addr_file,
                       usecols=usecols_addr, dtype=dtype_addr,
                       thousands=thousands, encoding=encoding, engine=engine,
                       nrows=nrows, **kwargs)
    info = pd.read_csv(info_file,
                       usecols=usecols_info, dtype=dtype_info,
                       thousands=thousands, encoding=encoding, engine=engine,
                       nrows=nrows, **kwargs)
    cis = cis.merge(addr, how='left', on='BA_ID')
    cis = cis.merge(info, how='left', on='BA_ID')

    # Rename columns to standardized names
    cis = cis.rename(columns={'BA_ID': 'keyAcctID',
                              'GNN_ID': 'premiseID',
                              'MTR_ID': 'meterNum',
                              'SADDR': 'serviceAddress',
                              'SCITY': 'serviceCity',
                              'SZIP': 'serviceZip',
                              'GEO_X_NB': 'geoLat',
                              'GEO_Y_NB': 'geoLong',
                              'NAICS': 'corpNAICS'})

    # Extract only the 5-digit zip codes
    cis['serviceZip'] = cis['serviceZip'].str[:5]

    # Assign labels for IOU and fuel type
    cis['iou'] = 'SCG'
    cis['fuel'] = 'G'

    return cis


def read_cis(file, iou, usecols=None, dtype=None, nrows=None, **kwargs):
    # Define default columns to read from the CSV file
    if usecols is None:
        usecols = ['iou', 'fuel',
                   'keyAcctID', 'premiseID', 'siteID', 'nrfSiteID', 'meterNum',
                   'serviceAddress', 'serviceCity', 'serviceZip',
                   'censusCounty', 'censusTract', 'censusBlock',
                   'geoID', 'geoLat', 'geoLong',
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
                 'censusCounty': str,
                 'censusTract': str,
                 'censusBlock': str,
                 'geoID': str,
                 'geoLat': np.float64,
                 'geoLong': np.float64,
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
    if iou != 'scg':
        cis = pd.read_csv(file,
                          usecols=usecols, dtype=dtype,
                          thousands=thousands, encoding=encoding,
                          engine=engine, nrows=nrows, **kwargs)
    else:
        cis_file = file['cis']
        addr_file = file['addr']
        info_file = file['info']
        cis = _read_cis_scg(cis_file, addr_file, info_file, nrows, **kwargs)

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

    # Pad keyAcctID and premiseID to 10 digits
    for col in ['keyAcctID', 'premiseID']:
        if col in cis:
            cis[col] = cis[col].apply(pad_digits, width=10)

    # Pad geocodes to the correct digits
    if 'censusCounty' in cis:
        cis['censusCounty'] = cis['censusCounty'].apply(pad_digits, width=3)
    if 'censusTract' in cis:
        cis['censusTract'] = cis['censusTract'].apply(pad_digits, width=6)
    if 'censusBlock' in cis:
        cis['censusBlock'] = cis['censusBlock'].apply(pad_digits, width=4)

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

    # Pad keyAcctID to 10 digits
    bills['keyAcctID'] = bills['keyAcctID'].apply(pad_digits, width=10)

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


def read_processed_bills(file, multi_index=True, dtype=None):

    if multi_index:
        header = [0, 1]
    else:
        header = None

    # Define dtypes for all possible (level 0) columns
    dtype = {'cis': str,
             'kWh': np.float64,
             'kWhOn': np.float64,
             'kWhSemi': np.float64,
             'kWhOff': np.float64,
             'kW': np.float64,
             'kWOn': np.float64,
             'kWSemi': np.float64,
             'billAmnt': np.float64,
             'Therms': np.float64,
             'EUI_elec': np.float64,
             'EUI_gas': np.float64,
             'EUI_tot': np.float64,
             'EUI_tot_mo_avg_2009_2015': np.float64,
             'summary': np.float64}
    # Define all possible (level 1) columns under cis to be converted to float
    col_to_float = ['Longitude', 'Latitude',
                    'Year Built', 'Year Renovated',
                    'Vacancy %', 'Number Of Stories', 'Rentable Building Area']
    # Define all possible (level 1) columns under cis to be converted to
    # datetime
    col_to_time = ['Last Sale Date']

    # Read file
    df = pd.read_csv(file, header=header, dtype=dtype)

    # Convert (level 1) columns to float
    for col in col_to_float:
        full_col = ('cis', col)
        if full_col in df:
            df.loc[:, full_col] = df.loc[:, full_col].astype(np.float64)
    # Convert (level 1) columns to datetime
    for col in col_to_time:
        full_col = ('cis', col)
        if full_col in df:
            df.loc[:, full_col] = pd.to_datetime(df.loc[:, full_col],
                                                 format='%Y-%m-%d')

    return df
