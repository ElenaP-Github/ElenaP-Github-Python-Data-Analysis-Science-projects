#Import the data: Create a SQL database using the attached CSV files
# ‘ assignment_data_clinic_group.csv’ contains clinic group data (clinic IDs and ClinicGroups)
#‘ assignment_data_clinics_with_patients.csv’ contains patient/clinic data (patient IDs, clinic title, clinic IDs, patient created at date and patient deleted at date)
# ‘ assignment_data_modules.csv’ contains module data (patient IDs, date of module generation, number of modules)
CREATE DATABASE IF NOT EXISTS all_data;
                                     ###########
# Create a dataset: use SQL to combine data from imported tables. Final table should contain following columns:
# patient ID, clinic ID, clinic title, clinic group, patient created at date, module completion date, number of modules
USE all_data;
SELECT
    pat.patient_id, pat.clinic_id,
    pat.clinic_title,  gr.CLINIC_GROUP,  pat.created_at, 
    modules.COMPLETION_DATE,
    modules.NUMBER_OF_MODULES
FROM
    assignment_data_clinics_with_patients pat
    join
    assignment_data_modules modules ON  modules.patient_id= pat.patient_id 
    join assignment_data_clinic_group_ gr on gr.CLINIC_ID= pat.CLINIC_ID
    #is null
    ORDER BY pat.patient_id asc ;
    #------------------- Data Exploration
### patients database
SELECT
   (select count(pat.patient_id) FROM
    assignment_data_clinics_with_patients pat) as count_rows ,
  ( SELECT
    count(distinct(pat.patient_id)) FROM
    assignment_data_clinics_with_patients pat) as count_distinct_patientID;

SELECT
   (select count(modules.patient_id) FROM
   assignment_data_modules modules) as count_rows_mod ,
  ( SELECT
    count(distinct(modules.patient_id)) FROM
    assignment_data_modules modules) as count_distinct_mod;
### modules: each pateint_id has several rows
### clinic_patient DB : just one row for each patient_id

#-------------  
SELECT 
(select count(Distinct (pat.patient_id)) from assignment_data_clinics_with_patients pat) as count_distinct_patients_PatientsDB
 ,
 (select count(Distinct (modules.patient_id)) from  assignment_data_modules modules) as count_distinct_patients_ModulesDB,
 (select count(Distinct (pat.patient_id)) from assignment_data_clinics_with_patients pat)
 -
 (select count(Distinct (modules.patient_id)) from  assignment_data_modules modules) as Difference_btw_DBs;