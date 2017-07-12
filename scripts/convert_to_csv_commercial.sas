/* Script to convert all sas7bdat files in given directories to csv files
   Anthony Ho <anthony.ho@energy.ca.gov>
   Last updated 7/6/2017 */

* Macro to convert a single data set to csv;
%MACRO convert_to_csv_single(data_lib, data_memname, data_dir);
	PROC EXPORT DATA = &data_lib..&data_memname
				OUTFILE = "&data_dir\&data_memname..csv"
				DBMS = CSV REPLACE;
		PUTNAMES = YES;
	RUN;
%MEND;

* Macro to convert all sas7bdat files in a given directory;
%MACRO convert_to_csv_dir(data_dir);
	LIBNAME data_lib &data_dir;
	DATA _NULL_;
		SET sashelp.vstable(WHERE = (LIBNAME = 'DATA_LIB'));
		CALL EXECUTE(cats('%nrstr(%convert_to_csv_single)('
	                   ,catx(',', libname, lowcase(memname), &data_dir)
	                   ,')'));
	RUN;
%MEND;

* Run through commercial data;
%LET data_dir = "c:\Users\ahho\work\CPUC\Commercial\CIS\PGE";
%convert_to_csv_dir(&data_dir)

%LET data_dir = "c:\Users\ahho\work\CPUC\Commercial\CIS\SCE";
%convert_to_csv_dir(&data_dir)

%LET data_dir = "c:\Users\ahho\work\CPUC\Commercial\CIS\SCG";
%convert_to_csv_dir(&data_dir)

%LET data_dir = "c:\Users\ahho\work\CPUC\Commercial\CIS\SDGE";
%convert_to_csv_dir(&data_dir)

%LET data_dir = "c:\Users\ahho\work\CPUC\Commercial\EE Program Tracking";
%convert_to_csv_dir(&data_dir)

%LET data_dir = "c:\Users\ahho\work\CPUC\Commercial\Monthly Bills";
%convert_to_csv_dir(&data_dir)