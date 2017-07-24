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
    usecols = ['PropertyID', 'PropertyType', 'Secondary Type',
               'Building Address', 'City', 'Zip', 'County Name',
               'Building Status', 'Year Built', 'Year Renovated',
               'Number Of Stories', 'Rentable Building Area',
               'Energy Star', 'LEED Certified', 'Last Sale Date', 'Vacancy %']
    dtype = {'PropertyID': str,
             'PropertyType': str,
             'Secondary Type': str,
             'Building Address': str,
             'City': str,
             'Zip': str,
             'County Name': str,
             'Building Status': str,
             'Year Built': np.float16,
             'Year Renovated': np.float16,
             'Number Of Stories': np.float16,
             'Rentable Building Area': np.float16,
             'Energy Star': str,
             'LEED Certified': str,
             'Last Sale Date': str,
             'Vacancy %': np.float16}
    encoding = 'iso-8859-1'
    engine = 'c'

    data = pd.read_csv(file,
                       usecols=usecols, dtype=dtype,
                       encoding=encoding, engine=engine,
                       nrows=nrows)

    data['Building Address'] = data['Building Address'].str.upper()
    data['City'] = data['City'].str.upper()
    data['Zip'] = data['Zip'].str[:5]
    data['County Name'] = data['County Name'].str.upper()
    data['Last Sale Date'] = pd.to_datetime(data['Last Sale Date'],
                                            format='%m/%d/%Y')

    data = data.rename(columns={'Building Address': 'address',
                                'City': 'city',
                                'Zip': 'zip',
                                'County Name': 'county'})

    return data


def read_cis(file, nrows=None):
    usecols = ['serviceAddress', 'serviceCity', 'serviceZip',  # add more
               'keyAcctId', 'premiseID', 'siteID', 'nrfSiteID', 'meterNum',
               'Fuel',
               'premNAICS', 'premNaicsBldg', 'corpNAICS', 'corpNaicsBldg',
               'CSSnaicsBldg',
               'BenchmarkFlag', 'CECClimateZone',
               'NETMETER',
               'acctProg1012Flag', 'acctProg1314Flag', 'acctProg2015Flag']
    dtype = {'serviceZip': str,
             'premiseID': str,
             'keyAcctId': str}
    encoding = 'ISO-8859-1'
    engine = 'c'

    cis = pd.read_csv(file,
                      usecols=usecols, dtype=dtype,
                      encoding=encoding, engine=engine,
                      nrows=nrows)

    # cis = cis.drop_duplicates()
    # clean up uneven column names
    cis = cis.rename(columns={'serviceAddress': 'address',
                              'serviceCity': 'city',
                              'serviceZip': 'zip'})

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
