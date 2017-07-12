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
list_input_files = ["../../Commercial/CIS/PGE/nrf_byacct_pge_2016.sas7bdat",
                    "../../Commercial/CIS/PGE/nrf_bysite_pge_2016.sas7bdat",
                    "../../Commercial/CIS/SCE/nrf_byacct_sce_2016.sas7bdat",
                    "../../Commercial/CIS/SCE/nrf_bysite_sce_2016.sas7bdat",
                    "../../Commercial/CIS/SCG/eega_addr24112.sas7bdat",
                    "../../Commercial/CIS/SCG/eega_cust24112.sas7bdat",
                    "../../Commercial/CIS/SCG/eega_info24412.sas7bdat",
                    "../../Commercial/CIS/SCG/scgcis_201503.sas7bdat",
                    "../../Commercial/CIS/SDGE/nrf_byacct_sdge_2016.sas7bdat",
                    "../../Commercial/CIS/SDGE/nrf_bysite_sdge_2016.sas7bdat",
                    "../../Commercial/EE Program Tracking/pge_track_byclaimid_q12.sas7bdat",
                    "../../Commercial/EE Program Tracking/sce_track_byclaimid_q12.sas7bdat",
                    "../../Commercial/EE Program Tracking/scg_track_byclaimid_q12.sas7bdat",
                    "../../Commercial/EE Program Tracking/sdge_track_byclaimid_q12.sas7bdat",
                    "../../Commercial/EE Program Tracking/trk1315_wroadmap.sas7bdat",
                    "../../Commercial/Monthly Bills/pge_elec_bill_data_2010_2016.sas7bdat",
                    "../../Commercial/Monthly Bills/pge_gas_bill_data_2010_2016.sas7bdat",
                    "../../Commercial/Monthly Bills/sce_elec_bill_data_2010_2016.sas7bdat",
                    "../../Commercial/Monthly Bills/scg_gas_bill_data_2013_2016.sas7bdat",
                    "../../Commercial/Monthly Bills/sdge_elec_bill_data_2010_2016.sas7bdat",
                    "../../Commercial/Monthly Bills/sdge_gas_bill_data_2010_2016.sas7bdat"]

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
