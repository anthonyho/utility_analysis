"""
Python module for processing customer-level monthly utility data

Required libraries:
* numpy (included in Anaconda)
* scipy (included in Anaconda)
* pandas (included in Anaconda)
* calendar (included in Python)
* os (included in Python)

Anthony Ho <anthony.ho@energy.ca.gov>
Last update 9/5/2017
"""

import numpy as np
from scipy.stats import linregress
import pandas as pd
from pandas.tseries.offsets import MonthBegin, MonthEnd, DateOffset
import calendar
import os


def get_census_tract(df):
    """
    Function to get the census tracts for all addresses in a dataframe.

    Deprecated - use get_geocodes_batch() and _get_geocodes_single_chunk()
    instead

    Required library:
    ----------------
    * censusgeocode (have to install separately:
                     https://pypi.python.org/pypi/censusgeocode)

    Parameters:
    ----------
    df: Pandas dataframe
        dataframe to get census tracts with. Must contain the fields 'address',
        'city', and 'zip'

    Return:
    -------
    df: Pandas dataframe
        same with df from input, except with 'tract_geocode' as an additional
        column containing the census tract number of each row
    """
    globals()['censusgeocode'] = __import__('censusgeocode')
    df = df.copy()
    df['tract_geocode'] = df.apply(_get_geocode_single, axis=1)
    return df


def _get_geocode_single(row, state='CA'):
    """
    Internal function to get the geocode of a single address. Used as part of
    get_census_tract()

    Deprecated - use get_geocodes_batch() and _get_geocodes_single_chunk()
    instead

    Parameters:
    ----------
    row: Pandas series
        row of the dataframe for getting the geocode. Must contain the fields
        'address', 'city', and 'zip'
    state: str (default: 'CA')
        two–letter state abbreviation for the address

    Return:
    ------
    tract_geocode: int or np.nan
        11 digit census tract number (2-digit state code + 3-digit county code
        + 6-digit census tract code)
    """
    address = row['address']
    city = row['city']
    zipcode = row['zip']
    cg = censusgeocode.CensusGeocode()
    try:
        result = cg.address(address, city=city, state=state, zipcode=zipcode)
        tract_geocode = result[0]['geographies']['Census Tracts'][0]['GEOID']
    except (TypeError, KeyError, IndexError):
        tract_geocode = np.nan
    return tract_geocode


def get_geocodes_batch(df, max_chunk_size=1000,
                       save_chunk=True, chunk_dir='./', restart=None):
    """
    Function to get the census tracts for all addresses in a dataframe using
    the Census Geocoding Services API batch service
    https://geocoding.geo.census.gov/geocoder/geographies/addressbatch?form

    Required library:
    ----------------
    * censusbatchgeocoder (have to install separately:
                           https://github.com/datadesk/python-censusbatchgeocoder)

    Parameters:
    ----------
    df: Pandas dataframe
        dataframe to get census tracts with. Must contain the fields 'address',
        'city', 'zipcode', and 'state'.
    max_chunk_size: int (default: 1000)
        max number of rows in a single chunk.  Limited to 1000 rows due to the
        census geocoder API.
    save_chunk: bool (default: True)
        save geocoded chunk into temporary file if True. Useful for restarting
        if function breaks due to API connection timeout
    chunk_dir: str (default: './')
        path to the directory for saving the temporary chunk files
    restart: int (default: None)
        restart the geocoding process at a specific chunk if geocoding process
        breaks due to API connection timeout

    Return:
    -------
    df_geocodes : Pandas dataframe
        geocode results. See https://github.com/datadesk/python-censusbatchgeocoder
        for details
    """
    globals()['censusbatchgeocoder'] = __import__('censusbatchgeocoder')
    # Break df into chunks
    n_row = len(df)
    n_chunk = n_row // max_chunk_size + 1
    list_chunks = np.array_split(df, n_chunk)
    # Get geocodes
    list_geocodes = []
    # Start from the beginning if no restart specified
    if restart is None:
        for i, chunk in enumerate(list_chunks):
            geocodes = _get_geocodes_single_chunk(chunk, i,
                                                  save_chunk, chunk_dir)
            list_geocodes.append(geocodes)
    # Start from restarting point by loading previosuly saved chunks if restart
    # is specified
    else:
        for i, chunk in enumerate(list_chunks):
            if i < restart:
                try:
                    chunk_path = os.path.join(chunk_dir, str(i) + '.csv')
                    geocodes = pd.read_csv(chunk_path)
                except FileNotFoundError:
                    geocodes = None
                list_geocodes.append(geocodes)
            else:
                geocodes = _get_geocodes_single_chunk(chunk, i,
                                                      save_chunk, chunk_dir)
                list_geocodes.append(geocodes)
    # Combine all goecoded chunks into a single dataframe
    df_geocodes = pd.concat(list_geocodes, axis=0, ignore_index=True)
    return df_geocodes


def _get_geocodes_single_chunk(chunk, chunk_id=None,
                               save_chunk=True, chunk_dir='./'):
    """
    Internal function to get the geocode of a chunk of many addresses in batch
    using the Census Geocoding Services API batch service. Used as part of
    get_geocodes_batch()

    Parameters:
    ----------
    chunk: Pandas dataframe
        small chunk of addresses to match geocodes with. Must contain the
        fields 'address', 'city', 'zipcode', and 'state'. Limited to 1000 rows
        in chunk due to the census geocoder API.
    chunk_id: str or int (default: None)
        name of the chunk for saving into temporary file if save_chunk is True
    save_chunk: bool (default: True)
        save geocoded chunk into temporary file if True. Useful for restarting
        if function breaks due to API connection timeout
    chunk_dir: str (default: './')
        path to the directory for saving the temporary chunk files

    Return:
    ------
    geocodes: Pandas dataframe
        geocode results. See https://github.com/datadesk/python-censusbatchgeocoder
        for details
    """
    chunk_dict = chunk.to_dict('records')
    try:
        geocodes = pd.DataFrame(censusbatchgeocoder.geocode(chunk_dict))
        if save_chunk:
            chunk_path = os.path.join(chunk_dir, str(chunk_id) + '.csv')
            geocodes.to_csv(chunk_path, index=False)
    except ValueError:
        geocodes = None
    return geocodes


def assign_bldg_type(df, bldg_type_file):
    """
    Function to assign standardized building type to a dataframe of addresses
    based on their CoStar or DMP building categories

    Parameters:
    ----------
    df: Pandas dataframe
        dataframe to assign building types with. Must contain fields
        'PropertyType' (from CoStar), 'Secondary Type' (from CoStar),
        'USE_CODE_STD_CTGR_DESC' (from DMP), and 'USE_CODE_STD_DESC' (from DMP)
    bldg_type_file: str
        path to the csv file containing the mapping from CoStar/DMP building
        categories to the standardized building types. Must contain fields
        'PropertyType' (from CoStar), 'Secondary Type' (from CoStar),
        'USE_CODE_STD_CTGR_DESC' (from DMP), and 'USE_CODE_STD_DESC' (from DMP)

    Return:
    -------
    Pandas dataframe
        same as df in input except with additional columns as specified by
        bldg_type_file
    """
    bldg_type_mapping = pd.read_csv(bldg_type_file)
    return df.merge(bldg_type_mapping,
                    how='left',
                    on=['PropertyType', 'Secondary Type',
                        'USE_CODE_STD_CTGR_DESC', 'USE_CODE_STD_DESC'])


def get_climate_zones(df, cz_file):
    """
    Function to assign climate zone to a dataframe of addresses based on their
    zip codes

    Parameters:
    ----------
    df: Pandas dataframe
        dataframe to assign climate zones with. Must contain fields "zip"
    cz_file: str
        path to the csv file containing the zipcode-to-climate zone mapping.
        (http://www.energy.ca.gov/maps/renewable/building_climate_zones.html)
        Must contain fields 'zip' and 'cz'

    Return:
    -------
    Pandas dataframe
        same as df in input except with the additional columns 'cz'
    """
    cz = pd.read_csv(cz_file, dtype={'zip': str, 'cz': str})
    return df.merge(cz, on='zip', how='left')


def merge_building_cis_data(bldg_data, cis, merge_on_range=True):
    """
    Function to merge building data (e.g. CoStar or DMP) with CIS data, with
    the capability to merge addresses with hyphenated ranges in street numbers
    (e.g. commonly seen in building data) to the corresponding single street
    number addresses (e.g. those in CIS)

    Parameters:
    ----------
    bldg_data: Pandas dataframe
        building data dataframe. Must contain the fields 'address', 'city', and
        'zip'. The field 'address' must contain single-street-number addresses
        only
    cis: Pandas dataframe
        CIS dataframe. Must contain the fields 'address', 'city', and 'zip'.
        The field 'address' could contain hyphenated ranges in street numbers.
    merge_on_range: bool (default: True)
        If True, merge addresses with hyphenated ranges in street numbers to
        the corresponding single street number addresses, and keep the address
        with street number range in the final output

    Return:
    -------
    bldg_cis: Pandas dataframe
        merged results from bldg_data and cis
    """
    # If enabled, replace addresses in CIS that match with the range addresses
    # in bldg_data by the matching range addresses
    if merge_on_range:
        range_addr_map = _expand_range_addr(bldg_data)
        cis = cis.merge(range_addr_map,
                        left_on=['address', 'city', 'zip'],
                        right_on=['indiv_address', 'city', 'zip'],
                        how='left')
        ind = cis['range_address'].notnull()
        cis.loc[ind, 'address'] = cis.loc[ind, 'range_address']
        cis = cis.drop(['indiv_address', 'range_address'], axis=1)
    # Merge bldg_data with cis
    bldg_cis = pd.merge(bldg_data, cis,
                        how='inner', on=['address', 'city', 'zip'])
    return bldg_cis


def _expand_range_addr(df):
    """
    Internal function called by merge_building_cis_data(). Given a dataframe
    containing a column 'address', the function looks for addresses that
    contains hyphenated ranges in street numbers and returns a dataframe with
    the "expanded" version of the addresses, i.e. all possible single-numbered
    addresses that fall within the ranges.

    Parameters:
    ----------
    df: Pandas dataframe
        dataframe containing the 'address' column.

    Return:
    -------
    Pandas dataframe
        expanded single-numbered addresses given the range addresses in df
    """
    address = df['address']
    regex = r"^[0-9]+-[0-9]+$"
    ind = address.str.split(pat=' ', n=1).str[0].str.contains(regex)
    df_range = df[ind]
    list_expanded_df = []
    for (i, row) in df_range.iterrows():
        list_expanded_df.append(_expand_range_addr_single(row))
    return pd.concat(list_expanded_df, axis=0, ignore_index=True)


def _expand_range_addr_single(row):
    """
    Internal function called by _expand_range_addr(). Given a address that
    contain hyphenated range in its street number, the function returns a
    dataframe with the "expanded" version of that address, i.e. all possible
    single-numbered addresses that fall within the range.

    Parameters:
    ----------
    row: Pandas series
        row containing the field 'address', 'city', and 'zip'. The field
        'address' must contain hyphenated range in its street number.

    Return:
    -------
    Pandas dataframe
        expanded single-numbered addresses given the range address in row
    """
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
    """
    Function to calendarize billing data into calendar months based on start
    and end dates of bills

    Parameters:
    ----------
    df: Pandas dataframe
        dataframe to be calendarized. Must be in "long-form" with the following
        columns: 'lastReadDate', 'ReadDate', 'readDays'
    group_keys: list of str or str (default: ['keyAcctID', 'premiseID'])
        columns to be used as group keys (e.g. grouping by account/building)
    list_fields: list of str (default: ['kWh'])
        columns for the values to be calendarized (e.g. kWh/Therms/bill_amount)
    keep_cols: list of str (default: [])
        columns to be kept constant during calendarization (e.g. rate)

    Return:
    -------
    Pandas dataframe
        calendarized version of df containing a column called 'yr_mo' which
        indicates the calendar year/month instead of 'lastReadDate' and
        'ReadDate'.
    """
    new_df = df.groupby(group_keys).apply(_calendarize_group,
                                          list_fields, keep_cols)
    return new_df.reset_index(drop=True, level=-1).reset_index()


def _calendarize_group(group, list_fields, keep_cols):
    """
    Internal function for calendarizing a single group. Used with calendarize()

    Parameters:
    ----------
    group: Pandas dataframe
        group to be calendarized. Must be in "long-form" with the following
        columns: 'lastReadDate', 'ReadDate', 'readDays'
    list_fields: list of str
        columns for the values to be calendarized (e.g. kWh/Therms/bill_amount)
    keep_cols: list of str
        columns to be kept constant during calendarization (e.g. rate)

    Return:
    -------
    calendarized_group: Pandas dataframe
        calendarized version of group containing a column called 'yr_mo' which
        indicates the calendar year/month instead of 'lastReadDate' and
        'ReadDate'.
    """
    expanded_rows = []
    for row in group.itertuples():
        expanded_rows.append(_expand_row(row, list_fields))
    expanded_group = pd.concat(expanded_rows, axis=0, ignore_index=True)
    calendarized_group = expanded_group.groupby('yr_mo').sum().reset_index()
    for col in reversed(keep_cols):
        calendarized_group.insert(0, col, group.iloc[0][col])
    return calendarized_group


def _expand_row(row, list_fields):
    """
    Internal function for calendarizing a single bill (a row). Used with
    _calendarize_group()

    Parameters:
    ----------
    row: Pandas series
        bill to be calendarized. Must contain the following columns:
        'lastReadDate', 'ReadDate', 'readDays'
    list_fields: list of str
        columns for the values to be calendarized (e.g. kWh/Therms/bill_amount)

    Return:
    -------
    expanded_row: Pandas dataframe
        dataframe containing rows that correspond to all calendar months
        contained in the bill
    """
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
    """
    Internaty function for computing the number of days in a list of months.
    Used with _expand_row().

    Parameters:
    ----------
    start_date: datetime object
        start date of a bill
    end_date: datetime object
        end date of a bill
    n_months: int
        number of months in list_yr_mo
    list_yr_mo: list
        list of yr_mo in the format of 'YYYY-mm'

    Return:
    -------
    numpy array
        array of number of days in each calendar month
    """
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


def _masked_add(elec, gas):
    """
    Internal function for adding energy consumption of different fuel types as
    described by the following:
    1. when E and G are both present in a given month, return E + G
    2. when only E is present in a given month, return:
     (a) np.nan when the G is present in any other months for the same building
     (b) E when the G is not present in any other months for the same building
    3. when only G is present in a given month, return:
     (a) np.nan when the E is present in any other months for the same building
     (b) G when the E is not present in any other months for the same building

    Parameters:
    ----------
    elec: Pandas series
    gas: Pandas series

    Return:
    -------
    total: Pandas series
        sum of elec and gas consumption (in kBTU) as defined by above
    """
    # Define mask for electric according to rules above
    all_zeros_elec = ~elec.notnull().any(axis=1)
    mask_elec = elec.notnull().replace({True: 1, False: np.nan})
    mask_elec[all_zeros_elec] = 1
    # Define mask for gas according to rules above
    all_zeros_gas = ~gas.notnull().any(axis=1)
    mask_gas = gas.notnull().replace({True: 1, False: np.nan})
    mask_gas[all_zeros_gas] = 1
    # Add and mask
    total = elec.add(gas, fill_value=0)
    total = total.multiply(mask_elec).multiply(mask_gas)
    return total


def compute_EUI(df, fuel='tot'):
    kWh_to_kBTU = 3.41214
    therms_to_kBTU = 100
    if fuel == 'tot':
        elec_kbtu = df['kWh'] * kWh_to_kBTU
        gas_kbtu = df['Therms'] * therms_to_kBTU
        total_energy = _masked_add(elec_kbtu, gas_kbtu)
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
