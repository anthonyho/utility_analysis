#!/usr/bin/env python3
'''
Script to convert sas7bdat files to csv files
Supports reading compressed SAS data

Anthony Ho <anthony.ho@energy.ca.gov>
Last updated 6/28/2017
'''

from sas7bdat import SAS7BDAT
import os
import time


# Define input SAS7BDAT files here
list_input_files = ["../../Residential/CIS/pge_elec_custinfo_2010_2015.sas7bdat",
                    "../../Residential/CIS/pge_gas_custinfo_2010_2015.sas7bdat",
                    "../../Residential/bills/bills_pge_2010e.sas7bdat",
                    "../../Residential/bills/bills_pge_2011e.sas7bdat",
                    "../../Residential/bills/bills_pge_2012e.sas7bdat",
                    "../../Residential/bills/bills_pge_2013e.sas7bdat",
                    "../../Residential/bills/bills_pge_2014e.sas7bdat",
                    "../../Residential/bills/bills_pge_2015e.sas7bdat",
                    "../../Residential/bills/bills_pge_2010g.sas7bdat",
                    "../../Residential/bills/bills_pge_2011g.sas7bdat",
                    "../../Residential/bills/bills_pge_2012g.sas7bdat",
                    "../../Residential/bills/bills_pge_2013g.sas7bdat",
                    "../../Residential/bills/bills_pge_2014g.sas7bdat",
                    "../../Residential/bills/bills_pge_2015g.sas7bdat",
                    "../../Residential/CIS/sce_custinfo_2010_2015.sas7bdat",
                    "../../Residential/CIS/scg_custinfo_2010_2015.sas7bdat",
                    "../../Residential/bills/bills_sce_2010.sas7bdat",
                    "../../Residential/bills/bills_sce_2011.sas7bdat",
                    "../../Residential/bills/bills_sce_2012.sas7bdat",
                    "../../Residential/bills/bills_sce_2013.sas7bdat",
                    "../../Residential/bills/bills_sce_2014.sas7bdat",
                    "../../Residential/bills/bills_sce_2015.sas7bdat",
                    "../../Residential/bills/bills_scg_2010.sas7bdat",
                    "../../Residential/bills/bills_scg_2011.sas7bdat",
                    "../../Residential/bills/bills_scg_2012.sas7bdat",
                    "../../Residential/bills/bills_scg_2013.sas7bdat",
                    "../../Residential/bills/bills_scg_2014.sas7bdat",
                    "../../Residential/bills/bills_scg_2015.sas7bdat",
                    "../../Residential/CIS/sdge_elec_custinfo_2010_2015.sas7bdat",
                    "../../Residential/CIS/sdge_gas_custinfo_2010_2015.sas7bdat",
                    "../../Residential/bills/bills_sdge_2010e.sas7bdat",
                    "../../Residential/bills/bills_sdge_2011e.sas7bdat",
                    "../../Residential/bills/bills_sdge_2012e.sas7bdat",
                    "../../Residential/bills/bills_sdge_2013e.sas7bdat",
                    "../../Residential/bills/bills_sdge_2014e.sas7bdat",
                    "../../Residential/bills/bills_sdge_2015e.sas7bdat",
                    "../../Residential/bills/bills_sdge_2010g.sas7bdat",
                    "../../Residential/bills/bills_sdge_2011g.sas7bdat",
                    "../../Residential/bills/bills_sdge_2012g.sas7bdat",
                    "../../Residential/bills/bills_sdge_2013g.sas7bdat",
                    "../../Residential/bills/bills_sdge_2014g.sas7bdat",
                    "../../Residential/bills/bills_sdge_2015g.sas7bdat"]

# Check if files exist
print("Checking if files exist...")
files_valid = []
for input_file in list_input_files:
    valid = os.path.isfile(input_file)
    files_valid.append(valid)
    if not valid:
        raise IOError(input_file+" does not exist")
if all(files_valid):
    print("All file paths are valid.")

# Convert to csv files
n_files = len(list_input_files)
print("Converting {} sas7bdat files...".format(n_files))
t_start = time.time()
for input_file in list_input_files:
    print("Converting {} ...".format(input_file))
    with SAS7BDAT(input_file) as f:
        df = f.to_data_frame()
        output_file = ".".join(input_file.split(".")[0:-1])+".csv"
        df.to_csv(output_file, index=False)
t_end = time.time()
print("Converted {} sas7bdat files into csv files".format(n_files))
print("Elapsed time = {:.2f} min".format((t_end - t_start) / 60))
