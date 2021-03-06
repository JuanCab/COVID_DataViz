##
## This library file contains routines to import COVID datafile created
## by the Collect_Data notebook.  This allows running with local copies
## of the data instead of requiring retrieval of data from the internet
## with every run of the COVID Visualization software.
##

import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from ast import literal_eval

## STRING TO LIST FUNCTIONS
## The following functions convert strings in the saved CSV files into
## lists of various types


def StringToListDate(string):
    # Converts string to list of date objects (and handles non-dates gracefully)
    # Initial Author: Juan
    dates_str_list = string.replace('\'','').strip('][').split(', ')
    dates_list = []
    for date in dates_str_list:
        try:
            newdate = datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            newdate = ""
        dates_list.append(newdate)
    return dates_list


def StringToListFloat(string):
    # Converts string to list of floats (We even treat integers as floats to
    # allow for NaN values)
    # Initial Author: Juan
    flt_list = list(map(float, string.replace('\'','').strip('][').split(', ')))
    return flt_list


def getLocalDataFrame(FIPS, DataFrame):
    # Returns a copied dataframe of just one FIP's entry.
    # Initial Author: Juan
    return DataFrame[DataFrame['FIPS'] == FIPS].copy()


def fixDataFrame(cols2convert, DataFrame):
    # This function will take a dictionayr cols2convert which lists columns and the function to
    # apply to them and then applies those functions to the columns of the DataFrame.
    # Initial author: Juan
    for key in cols2convert:
        DataFrame[key] = DataFrame[key].apply(cols2convert[key])

    return


## JOHN HOPKINS DATA IO
## The following block of routines are designed for reading in John Hopkins
## data on confirmed cases, deaths, and recovered numbers.


def GetCDRDataFrames(countyFile = 'our_data/countylevel_combinedCDR.csv', stateFile = 'our_data/statelevel_combinedCDR.csv'):
    # This function creates the data frames which are used in the functions below.
    # Initial author: Luke
    stateDataFrame = pd.read_csv(stateFile)
    countyDataFrame = pd.read_csv(countyFile)

    return stateDataFrame, countyDataFrame


def GetCountyDeaths(countyDF, countyFIPS):
    # Gets number of confirmed deaths (int)
    # Note: This function requires calling the GetCDRDataFrames first; this uses the second data frame returned
    # Initial author: Luke

    deathsList = countyDF[countyDF['FIPS'] == countyFIPS]['Deaths'].values[0]
    numDeaths = deathsList[len(deathsList) - 1]
    return numDeaths

def GetCountyInfections(countyDF, countyFIPS):
    # Gets number of confirmed infections (int)
    # Note: This function requires calling the GetCDRDataFrames first; this uses the second data frame returned
    # Initial author: Luke

    infectList = countyDF[countyDF['FIPS'] == countyFIPS]['Confirmed'].values[0]
    numInfect = infectList[len(infectList) - 1]
    return numInfect

def GetCountyRecoveries(countyDF, countyFIPS):
    # Gets number of confirmed recoveries (int)
    # Note: This function requires calling the GetCDRDataFrames first; this uses the second data frame returned
    # Initial author: Luke

    recovList = countyDF[countyDF['FIPS'] == countyFIPS]['Recovered'].values[0]
    numRecov = recovList[len(recovList) - 1]
    return numRecov

def GetStateDeaths(stateDF, stateFIPS):
    # Gets number of confirmed deaths (int)
    # Note: This function requires calling the GetCDRDataFrames first; this uses the second data frame returned
    # Initial author: Luke

    deathsList = stateDF[stateDF['FIPS'] == stateFIPS]['Deaths'].values[0]
    numDeaths = deathsList[len(deathsList) - 1]
    return numDeaths

def GetStateInfections(stateDF, stateFIPS):
    # Gets number of confirmed infections (int)
    # Note: This function requires calling the GetCDRDataFrames first; this uses the second data frame returned
    # Initial author: Luke

    infectList = stateDF[stateDF['FIPS'] == stateFIPS]['Confirmed'].values[0]
    numInfect = infectList[len(infectList) - 1]
    return numInfect

def GetStateRecoveries(stateDF, stateFIPS):
    # Gets number of confirmed recoveries (int)
    # Note: This function requires calling the GetCDRDataFrames first; this uses the second data frame returned
    # Initial author: Luke

    recovList = stateDF[stateDF['FIPS'] == stateFIPS]['Recovered'].values[0]
    numRecov = recovList[len(recovList) - 1]
    return numRecov


def CSVtoCDRDataFrames(countyFile = 'our_data/countylevel_combinedCDR.csv', stateFile = 'our_data/statelevel_combinedCDR.csv'):
    # This function creates the data frames from the John Hopkins data files.
    # It also handles the conversion of lists (stored as strings in the CSV file)
    # back into lists.
    # Initial author: Luke

    # List columns needing string to list conversion, use literal_eval to handle them
    # and import
    cols2convert = { 'Dates' : StringToListDate,
                     'Confirmed' : StringToListFloat,
                     'Deaths'  : StringToListFloat,
                     'Recovered' : StringToListFloat,
                     'dConfirmed'  : StringToListFloat,
                     'd2Confirmed'  : StringToListFloat,
                     'dDeaths' : StringToListFloat,
                     'd2Deaths' : StringToListFloat }
    countyDataFrame = pd.read_csv(countyFile)
    fixDataFrame(cols2convert, countyDataFrame)

    # Now add the additional columns to process in state CSV file
    cols2convert.update( { 'Incident_Rate' : StringToListFloat,
                           'People_Tested' : StringToListFloat,
                           'People_Hospitalized' : StringToListFloat,
                           'Mortality_Rate' : StringToListFloat,
                           'Testing_Rate' : StringToListFloat,
                           'Hospitalization_Rate' : StringToListFloat } )
    stateDataFrame = pd.read_csv(stateFile)
    fixDataFrame(cols2convert, stateDataFrame)

    return stateDataFrame, countyDataFrame


def PtoCDRDataFrames(stateFile = 'our_data/statelevel_combinedCDR.p', countyFile = 'our_data/countylevel_combinedCDR.p'):
    # This function creates the data frames from the John Hopkins data files.
    # It reads the pickle files, trying to avoid the time needed to convert strings to lists when reading CSVs.
    # Initial author: Luke

    # Import pickle files and convert appropriate columns
    pickle_file = open(countyFile,'rb')
    countyDataFrame = pickle.load(pickle_file)
    pickle_file.close()

    pickle_file = open(stateFile,'rb')
    stateDataFrame = pickle.load(pickle_file)
    pickle_file.close()

    return stateDataFrame, countyDataFrame


## IMHE DATA IO
## The following block of routines are designed for reading in IMHE
## data on confirmed cases, deaths, and recovered numbers.


def GetIMHEDataFrames(summaryFile = 'our_data/imhe_summary.csv', hospitalFile = 'our_data/imhe_hospitalizations.csv'):
    # This reads the local IMHE data files and returns data frames, which are to be used in the following functions
    summaryDF = pd.read_csv(summaryFile)
    hospitalizationsDF = pd.read_csv(hospitalFile)

    # Fix the column formats

    return summaryDF, hospitalizationsDF


def GetEquipData(fipsNum, summaryDataFrame): # This one's fine
    # Returns data on bed and ventilator usage/availability from imhe_summary.csv as a data frame
    # Note: This function requires calling the GetIMHEData first; this uses the first data frame returned
    # Initial author: Luke

    state = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['state']

    peak_bed_day_mean = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['peak_bed_day_mean']
    peak_bed_day_lower = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['peak_bed_day_lower']
    peak_bed_day_upper = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['peak_bed_day_upper']

    peak_icu_bed_day_mean = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['peak_icu_bed_day_mean']
    peak_icu_bed_day_lower = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['peak_icu_bed_day_lower']
    peak_icu_bed_day_upper = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['peak_icu_bed_day_upper']

    peak_vent_day_mean = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['peak_vent_day_mean']
    peak_vent_day_lower = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['peak_vent_day_lower']
    peak_vent_day_upper = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['peak_vent_day_upper']

    all_bed_capacity = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['all_bed_capacity']
    icu_bed_capacity = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['icu_bed_capacity']
    all_bed_usage = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['all_bed_usage']
    icu_bed_usage = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['icu_bed_usage']

    outDF = pd.DataFrame({'FIPS':fipsNum, 'state':state,
                        'peak_bed_day_mean':peak_bed_day_mean, 'peak_bed_day_lower':peak_bed_day_lower,
                        'peak_bed_day_upper':peak_bed_day_upper, 'peak_icu_bed_day_mean':peak_icu_bed_day_mean,
                        'peak_icu_bed_day_lower':peak_icu_bed_day_lower, 'peak_icu_bed_day_upper':peak_icu_bed_day_upper,
                        'peak_vent_day_mean':peak_vent_day_mean, 'peak_vent_day_lower':peak_vent_day_lower,
                        'peak_vent_day_upper':peak_vent_day_upper, 'all_bed_capacity':all_bed_capacity,
                        'icu_bed_capacity':icu_bed_capacity, 'all_bed_usage':all_bed_usage, 'icu_bed_usage':icu_bed_usage})

    # Columns with dates
    datecols = [ 'peak_bed_day_mean', 'peak_bed_day_lower', 'peak_bed_day_upper',
                 'peak_icu_bed_day_mean', 'peak_icu_bed_day_lower', 'peak_icu_bed_day_upper',
                 'peak_vent_day_mean', 'peak_vent_day_lower', 'peak_vent_day_upper', ]
    # Convert SINGLE date entries to proper pandas.datetime64 format
    for col in datecols:
        outDF[col] = pd.to_datetime(outDF[col], errors='coerce')

    return outDF


def GetNumICUBeds(fipsNum, summaryDataFrame):
    # These functions are for the "at a glance" values for bed availability and usage
    # Initial author: Luke

    icu_beds = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['icu_bed_capacity']
    return icu_beds.values[0] # make a return


def GetNumAllBeds(fipsNum, summaryDataFrame):
    # Initial author: Luke

    icu_beds = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['all_bed_capacity']
    return icu_beds.values[0] # make a return


def GetICUBedUsage(fipsNum, summaryDataFrame):
    # Initial author: Luke

    icu_beds = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['icu_bed_usage']
    return icu_beds.values[0] # make a return


def GetAllBedUsage(fipsNum, summaryDataFrame):
    # Initial author: Luke

    icu_beds = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['all_bed_usage']
    return icu_beds.values[0]


def GetHospitalizationData(fipsNum, hospitalizationsDF): # This one's having issues
    # Returns data from imhe_hospitalizations.csv as a data frame
    # Note: This function requires calling the GetIMHEData first; this uses the second data frame returned
    # Initial author: Luke

    state = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['state']
    dates = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['dates']

    allbed_mean = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['allbed_mean']
    allbed_lower = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['allbed_lower']
    allbed_upper = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['allbed_upper']

    ICUbed_mean = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['ICUbed_mean']
    ICUbed_lower = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['ICUbed_lower']
    ICUbed_upper = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['ICUbed_upper']

    InvVen_mean = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['InvVen_mean']
    InvVen_lower = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['InvVen_lower']
    InvVen_upper = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['InvVen_upper']

    Deaths_mean = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['deaths_mean']
    Deaths_lower = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['deaths_lower']
    Deaths_upper = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['deaths_upper']

    Admis_mean = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['admis_mean']
    Admis_lower = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['admis_lower']
    Admis_upper = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['admis_upper']

    newICU_mean = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['newICU_mean']
    newICU_lower = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['newICU_lower']
    newICU_upper = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['newICU_upper']

    outDF = pd.DataFrame({'FIPS':fipsNum, 'state':state, 'dates':dates,
                    'allbed_mean':allbed_mean, 'allbed_lower':allbed_lower,
                    'allbed_upper':allbed_upper, 'ICUbed_mean':ICUbed_mean,
                    'ICUbed_lower':ICUbed_lower, 'ICUbed_upper':ICUbed_upper,
                    'InvVen_mean':InvVen_mean, 'InvVen_lower':InvVen_lower,
                    'InvVen_upper':InvVen_upper, 'deaths_mean':Deaths_mean,
                    'deaths_lower':Deaths_lower, 'deaths_upper':Deaths_upper,
                    'admis_mean':Admis_mean, 'admis_lower':Admis_lower, 'admis_upper':Admis_lower,
                    'newICU_mean':newICU_mean, 'newICU_lower':newICU_lower, 'newICU_upper':newICU_lower
                    })
    return outDF


def CSVtoIMHEDataFrames(summaryFile = 'our_data/imhe_summary.csv', hospitalFile = 'our_data/imhe_hospitalizations.csv'):
    # This reads the local IMHE data files and returns data frames. It also handles
    # the conversion of lists (stored as strings in the CSV file) back into lists.
    # Original Author: Luke

    # Summary file has no columns needing strings converted to other variables,
    # however, many of the dates (stored as strings) have NaN in them.
    summaryDF = pd.read_csv(summaryFile)
    # Columns with dates
    datecols = [ 'peak_bed_day_mean', 'peak_bed_day_lower', 'peak_bed_day_upper',
                 'peak_icu_bed_day_mean', 'peak_icu_bed_day_lower', 'peak_icu_bed_day_upper',
                 'peak_vent_day_mean', 'peak_vent_day_lower', 'peak_vent_day_upper',
                 'travel_limit_start_date', 'travel_limit_end_date',
                 'stay_home_start_date', 'stay_home_end_date',
                 'educational_fac_start_date', 'educational_fac_end_date',
                 'any_gathering_restrict_start_date', 'any_gathering_restrict_end_date',
                 'any_business_start_date', 'any_business_end_date',
                 'all_non-ess_business_start_date', 'all_non-ess_business_end_date' ]
    # Convert SINGLE date entries
    for col in datecols:
        summaryDF[col] = pd.to_datetime(summaryDF[col], errors='coerce')

    # Hospitalization CSV has a lot of time series data stored as strings, need
    # to convert those in input

    hospitalizationsDF = pd.read_csv(hospitalFile)
    cols2convert = { 'dates' : StringToListDate,
                     'allbed_mean' : StringToListFloat,
                     'allbed_lower' : StringToListFloat,
                     'allbed_upper' : StringToListFloat,
                     'ICUbed_mean' : StringToListFloat,
                     'ICUbed_lower' : StringToListFloat,
                     'ICUbed_upper' : StringToListFloat,
                     'InvVen_mean' : StringToListFloat,
                     'InvVen_lower' : StringToListFloat,
                     'InvVen_upper' : StringToListFloat,
                     'deaths_mean' : StringToListFloat,
                     'deaths_lower' : StringToListFloat,
                     'deaths_upper' : StringToListFloat,
                     'admis_mean' : StringToListFloat,
                     'admis_lower' : StringToListFloat,
                     'admis_upper' : StringToListFloat,
                     'newICU_mean' : StringToListFloat,
                     'newICU_lower' : StringToListFloat,
                     'newICU_upper' : StringToListFloat,
                     'totdea_mean' : StringToListFloat,
                     'totdea_lower' : StringToListFloat,
                     'totdea_upper' : StringToListFloat,
                     'deaths_mean_smoothed' : StringToListFloat,
                     'deaths_lower_smoothed' : StringToListFloat,
                     'deaths_upper_smoothed' : StringToListFloat,
                     'totdea_mean_smoothed' : StringToListFloat,
                     'totdea_lower_smoothed' : StringToListFloat,
                     'totdea_upper_smoothed' : StringToListFloat,
                     'total_tests' : StringToListFloat,
                     'confirmed_infections' : StringToListFloat,
                     'est_infections_mean' : StringToListFloat,
                     'est_infections_lower' : StringToListFloat,
                     'est_infections_upper' : StringToListFloat }
    hospitalizationsDF = pd.read_csv(hospitalFile)
    fixDataFrame(cols2convert, hospitalizationsDF)

    return summaryDF, hospitalizationsDF


def PtoIMHEDataFrames(summaryFile = 'our_data/imhe_summary.p', hospitalFile = 'our_data/imhe_hospitalizations.p'):
    # This reads the local IMHE data files and returns data frames. It loads from pickle files, avoiding the entire
    # need to fix the lists stored as strings in the CSV files.
    # Original Author: Luke

    # Import pickle files and convert appropriate columns
    pickle_file = open(summaryFile,'rb')
    summaryDF = pickle.load(pickle_file)
    pickle_file.close()

    pickle_file = open(hospitalFile,'rb')
    hospitalizationsDF = pickle.load(pickle_file)
    pickle_file.close()

    return summaryDF, hospitalizationsDF


## APPLE AND GOOGLE MOBILITY DATA IO
## The following blocks of routines were designed for reading in Apple and
## Google mobility data.


def initAaplMobilityDataframes(countyFile = 'our_data/aapl_mobility_cnty.csv',stateFile = 'our_data/aapl_mobility_state.csv'):
    #reads the necessary paths and converts paths into dataframes
    #used inconjunction with applMobility_DataFunc()
    # Initial Author: Dio

    aaplMobilityCountyFrame = pd.read_csv(countyFile)
    aaplMobilityStateFrame = pd.read_csv(stateFile)
    #retuns dataFrames to be used
    return aaplMobilityCountyFrame,aaplMobilityStateFrame


def initgoogMobilityDataframes(countyFile = 'our_data/goog_mobility_cnty.csv',stateFile = 'our_data/goog_mobility_state.csv'):
    #used inconjunction with goog_MobilityFunc()
    # Initial Author: Dio
    googMobilityCountyFrame = pd.read_csv(countyFile)
    googMobilityStateFrame = pd.read_csv(stateFile)
    return googMobilityCountyFrame,googMobilityStateFrame


def CSVtoAAPLMobilityDataFrames(countyFile = 'our_data/aapl_mobility_cnty.csv', stateFile = 'our_data/aapl_mobility_state.csv'):
    # Retrieves the Apple mobility data from the CSV files, handling to conversion
    # of strings to lists in the dataframe.
    # Initial Author: Dio

    # List columns to convert from strings to lists
    cols2convert = { 'dates' : StringToListDate,
                     'driving_mobility' : StringToListFloat }
    # Import CSV and convert appropriate columns
    aaplMobilityCountyFrame = pd.read_csv(countyFile)
    fixDataFrame(cols2convert, aaplMobilityCountyFrame)
    aaplMobilityStateFrame = pd.read_csv(stateFile)
    fixDataFrame(cols2convert, aaplMobilityStateFrame)

    return aaplMobilityCountyFrame, aaplMobilityStateFrame


def CSVtoGOOGMobilityDataFrames(countyFile = 'our_data/goog_mobility_cnty.csv', stateFile = 'our_data/goog_mobility_state.csv'):
    # Retrieves the Google mobility data from the CSV files, handling to conversion
    # of strings to lists in the dataframe.
    # Initial Author: Dio

    # List columns to convert
    cols2convert = { 'dates' : StringToListDate,
                     'retail_and_recreation_percent_change_from_baseline' : StringToListFloat,
                     'grocery_and_pharmacy_percent_change_from_baseline' : StringToListFloat,
                     'parks_percent_change_from_baseline' : StringToListFloat,
                     'transit_stations_percent_change_from_baseline' : StringToListFloat,
                     'workplaces_percent_change_from_baseline' : StringToListFloat,
                     'residential_percent_change_from_baseline' : StringToListFloat }

    # Dictionary of renamed columns
    newnames = { 'retail_and_recreation_percent_change_from_baseline': 'retail_recreation_Percent',
                 'grocery_and_pharmacy_percent_change_from_baseline' : 'grocery_pharm_Percent',
                 'parks_percent_change_from_baseline' : 'parks_Percent',
                 'transit_stations_percent_change_from_baseline' : 'transit_stations_Percent',
                 'workplaces_percent_change_from_baseline' : 'residential_Percent',
                 'residential_percent_change_from_baseline' : 'workplace_Percent' }

    # Fix the columns that are strings that should be lists in the dataframe
    googMobilityCountyFrame = pd.read_csv(countyFile)
    fixDataFrame(cols2convert, googMobilityCountyFrame)
    googMobilityCountyFrame.rename(columns=newnames, errors="raise", inplace=True)
    googMobilityStateFrame = pd.read_csv(stateFile)
    fixDataFrame(cols2convert, googMobilityStateFrame)
    googMobilityStateFrame.rename(columns=newnames, errors="raise", inplace=True)
    return googMobilityCountyFrame, googMobilityStateFrame


def PtoAAPLMobilityDataFrames(countyFile = 'our_data/aapl_mobility_cnty.p', stateFile = 'our_data/aapl_mobility_state.p'):
    # Retrieves the Apple mobility data from the pickle files so dataframe shouldn't need fixing.
    # Initial Author: Dio

    # Import pickle files and convert appropriate columns
    pickle_file = open(countyFile,'rb')
    aaplMobilityCountyFrame = pickle.load(pickle_file)
    pickle_file.close()

    pickle_file = open(stateFile,'rb')
    aaplMobilityStateFrame = pickle.load(pickle_file)
    pickle_file.close()

    return aaplMobilityCountyFrame, aaplMobilityStateFrame


def PtoGOOGMobilityDataFrames(countyFile = 'our_data/goog_mobility_cnty.p', stateFile = 'our_data/goog_mobility_state.p'):
    # Retrieves the Google mobility data from the pickle files so dataframe shouldn't need fixing.
    # Initial Author: Dio

    # Import pickle files and convert appropriate columns
    pickle_file = open(countyFile,'rb')
    googMobilityCountyFrame = pickle.load(pickle_file)
    pickle_file.close()

    pickle_file = open(stateFile,'rb')
    googMobilityStateFrame = pickle.load(pickle_file)
    pickle_file.close()

    # Dictionary of renamed columns
    newnames = { 'retail_and_recreation_percent_change_from_baseline': 'retail_recreation_Percent',
                 'grocery_and_pharmacy_percent_change_from_baseline' : 'grocery_pharm_Percent',
                 'parks_percent_change_from_baseline' : 'parks_Percent',
                 'transit_stations_percent_change_from_baseline' : 'transit_stations_Percent',
                 'workplaces_percent_change_from_baseline' : 'workplace_Percent',
                 'residential_percent_change_from_baseline' : 'residential_Percent' }

    # Fix the columns that are strings that should be lists in the dataframe
    googMobilityCountyFrame.rename(columns=newnames, errors="raise", inplace=True)
    googMobilityStateFrame.rename(columns=newnames, errors="raise", inplace=True)

    return googMobilityCountyFrame, googMobilityStateFrame


def PtoRtDataFrames(stateFile = 'our_data/Rt_live.p'):
    # Retrieves the Rt live data from the pickle files so dataframe shouldn't need fixing.
    # Initial Author: Juan

    # Import pickle files and convert appropriate columns
    pickle_file = open(stateFile,'rb')
    RtLiveDataframe = pickle.load(pickle_file)
    pickle_file.close()

    return RtLiveDataframe
