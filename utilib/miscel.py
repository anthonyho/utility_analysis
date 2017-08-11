# Anthony Ho <anthony.ho@energy.ca.gov>
# Last update 8/11/2017
"""
Python module for miscellaneous functions for analyzing costumer-level energy
consumption data
"""


def add_level(list_col, level='cis'):
    return [(level, col) for col in list_col]


def group_by(df, list_col, level='cis'):
    return df.groupby(add_level(list_col, level=level))
