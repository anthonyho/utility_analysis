# Anthony Ho <anthony.ho@energy.ca.gov>
# Last update 7/24/2017
"""
Python module for reading utility data
"""

import numpy as np
import pandas as pd

# To-do's
# 1. allow reading other types of bills (gas and residential)


def read_costar(file, nrows=None):
    usecols = ['PropertyID',
               'Building Address', 'City', 'Zip', 'County Name',
               'Longitude', 'Latitude',
               'PropertyType', 'Secondary Type', 'Building Status',
               'Year Built', 'Year Renovated', 'Vacancy %',
               'Number Of Stories', 'Rentable Building Area',
               'Energy Star', 'LEED Certified', 'Last Sale Date']
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
    encoding = 'iso-8859-1'
    engine = 'c'

    data = pd.read_csv(file,
                       usecols=usecols, dtype=dtype,
                       encoding=encoding, engine=engine,
                       nrows=nrows)
    data = data.drop_duplicates()

    data = data.rename(columns={'Building Address': 'address',
                                'City': 'city',
                                'Zip': 'zip',
                                'County Name': 'county'})

    data['address'] = data['address'].str.upper()
    data['city'] = data['city'].str.upper()
    data['zip'] = data['zip'].str[:5]
    data['county'] = data['county'].str.upper()
    data['Last Sale Date'] = pd.to_datetime(data['Last Sale Date'],
                                            format='%m/%d/%Y')

    return data


def _modify_fields(usecols, dtype, badcols):
    for col in badcols:
        usecols = [badcols[col] if uc == col else uc for uc in usecols]
        dtype[badcols[col]] = dtype.pop(col)
    return usecols, dtype


def _drop_fields(usecols, dtype, dropcols):
    for col in dropcols:
        usecols.remove(col)
        del dtype[col]
    return usecols, dtype


def _rev_dict(d):
    return {v: k for k, v in d.items()}


def read_cis(file, iou, nrows=None):

    usecols = ['iou', 'fuel',
               'keyAcctID', 'premiseID', 'siteID', 'nrfSiteID', 'meterNum',
               'serviceAddress', 'serviceCity', 'serviceZip',
               'geoID', 'geoLat', 'geoLong',
               'censusBlock', 'censusCounty', 'censusTract',
               'CECClimateZone', 'CSSnaicsBldg',
               'premNAICS', 'premNaicsBldg', 'corpNAICS', 'corpNaicsBldg',
               'NetMeter', 'BenchmarkFlag',
               'acctProg1012Flag', 'acctProg1314Flag', 'acctProg2015Flag']
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
             'CECClimateZone': str,
             'CSSnaicsBldg': str,
             'premNAICS': str,
             'premNaicsBldg': str,
             'corpNAICS': str,
             'corpNaicsBldg': str,
             'NetMeter': str,
             'BenchmarkFlag': str,
             'acctProg1012Flag': str,
             'acctProg1314Flag': str,
             'acctProg2015Flag': str}
    thousands = ','
    encoding = 'ISO-8859-1'
    engine = 'c'

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
        dropcols = ['corpNAICS', 'corpNaicsBldg']

    if badcols is not None:
        usecols, dtype = _modify_fields(usecols, dtype, badcols)
    if dropcols is not None:
        usecols, dtype = _drop_fields(usecols, dtype, dropcols)

    cis = pd.read_csv(file,
                      usecols=usecols, dtype=dtype,
                      thousands=thousands, encoding=encoding, engine=engine,
                      nrows=nrows)
    cis = cis.drop_duplicates()

    if badcols is not None:
        cis = cis.rename(columns=_rev_dict(badcols))
    cis = cis.rename(columns={'serviceAddress': 'address',
                              'serviceCity': 'city',
                              'serviceZip': 'zip'})

    cis['address'] = cis['address'].str.upper()
    cis['city'] = cis['city'].str.upper()
    cis['NetMeter'] = cis['NetMeter'].map({'Y': True, 'N': False})
    cis['acctProg1012Flag'] = cis['acctProg1012Flag'].map({'1': True,
                                                           '0': False})
    cis['acctProg1314Flag'] = cis['acctProg1314Flag'].map({'1': True,
                                                           '0': False})
    cis['acctProg2015Flag'] = cis['acctProg2015Flag'].map({'1': True,
                                                           '0': False})

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
