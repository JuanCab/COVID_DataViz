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


def StringToListFloat(flt):
    # Converts string to list of floats (We even treat integers as floats to
    # allow for NaN values)
    # Initial Author: Juan
    flt_list = list(map(float, flt.replace('\'','').strip('][').split(', ')))
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


def GetCDRState(stateFIPS, stateDataFrame):
    # Gets number of confirmed cases, deaths and recoveries at the state level
    # Note: This function requires calling the GetCDRDataFrames first; this uses the first data frame returned
    # Initial author: Luke
    state = stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['State']
    dates = StringToListDate(stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['Dates'].values[0])

    confirmed = StringToListFloat(stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['Confirmed'].values[0])
    deaths = StringToListFloat(stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['Deaths'].values[0])
    recovered = StringToListFloat(stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['Recovered'].values[0])

    dConfirmed = StringToListFloat(stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['dConfirmed'].values[0])
    d2Confirmed = StringToListFloat(stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['d2Confirmed'].values[0])
    dDeaths = StringToListFloat(stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['dDeaths'].values[0])
    d2Deaths = StringToListFloat(stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['d2Deaths'].values[0])

    outDF = pd.DataFrame({'FIPS':stateFIPS, 'state':state, 'Dates':[dates],
                    'Confirmed':[confirmed], 'Deaths':[deaths],
                    'Recovered':[recovered], 'dConfirmed':[dConfirmed],
                    'd2Confirmed':[d2Confirmed], 'dDeaths':[dDeaths],
                    'd2Deaths':[d2Deaths]})
    return outDF


def GetCDRCounty(countyFIPS, countyDataFrame):
    # Gets number of confirmed cases, deaths and recoveries at the state level
    # Note: This function requires calling the GetCDRDataFrames first; this uses the second data frame returned
    # Initial author: Luke
    county = countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['County']
    dates = StringToListDate(countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['Dates'].values[0])

    confirmed = StringToListFloat(countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['Confirmed'].values[0])
    deaths = StringToListFloat(countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['Deaths'].values[0])
    recovered = StringToListFloat(countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['Recovered'].values[0])

    dConfirmed = StringToListFloat(countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['dConfirmed'].values[0])
    d2Confirmed = StringToListFloat(countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['d2Confirmed'].values[0])
    dDeaths = StringToListFloat(countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['dDeaths'].values[0])
    d2Deaths = StringToListFloat(countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['d2Deaths'].values[0])

    outDF = pd.DataFrame({'FIPS':countyFIPS, 'County':county, 'Dates':[dates],
                    'Confirmed':[confirmed], 'Deaths':[deaths],
                    'Recovered':[recovered], 'dConfirmed':[dConfirmed],
                    'd2Confirmed':[d2Confirmed], 'dDeaths':[dDeaths],
                    'd2Deaths':[d2Deaths]})
    return outDF


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
    return summaryDF, hospitalizationsDF


def GetEquipData(fipsNum, summaryDataFrame): # This one's fine
    # Returns data on bed and ventilator usage/availability from imhe_summary.csv as a data frame
    # Note: This function requires calling the GetIMHEData first; this uses the first data frame returned
    # Initial author: Luke

    #imhe_local_data = pd.read_csv(fileName)

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

    return outDF


def GetNumICUBeds(fipsNum, summaryDataFrame):
    # These functions are for the "at a glance" values for bed availability and usage
    # Initial author: Luke

    #imhe_local_data = pd.read_csv(fileName)
    mn_icu_beds = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['icu_bed_capacity']
    return mn_icu_beds.values[0] # make a return


def GetNumAllBeds(fipsNum, summaryDataFrame):
    # Initial author: Luke

    #imhe_local_data = pd.read_csv(fileName)
    mn_icu_beds = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['all_bed_capacity']
    return mn_icu_beds.values[0] # make a return


def GetICUBedUsage(fipsNum, summaryDataFrame):
    # Initial author: Luke

    #imhe_local_data = pd.read_csv(fileName)
    mn_icu_beds = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['icu_bed_usage']
    return mn_icu_beds.values[0] # make a return


def GetAllBedUsage(fipsNum, summaryDataFrame):
    # Initial author: Luke

    #imhe_local_data = pd.read_csv(fileName)
    mn_icu_beds = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['all_bed_usage']
    return mn_icu_beds.values[0]


def GetHospitalizationData(fipsNum, hospitalizationsDF): # This one's having issues
    # Returns data from imhe_hospitalizations.csv as a data frame
    # Note: This function requires calling the GetIMHEData first; this uses the second data frame returned
    # Initial author: Luke

    state = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['state']
    dates = StringToListDate(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['dates'].values[0])

    # Will the StringToList function be called here or later on...?
    allbed_mean = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['allbed_mean'].values[0])
    allbed_lower = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['allbed_lower'].values[0])
    allbed_upper = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['allbed_upper'].values[0])

    ICUbed_mean = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['ICUbed_mean'].values[0])
    ICUbed_lower = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['ICUbed_lower'].values[0])
    ICUbed_upper = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['ICUbed_upper'].values[0])

    InvVen_mean = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['InvVen_mean'].values[0])
    InvVen_lower = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['InvVen_lower'].values[0])
    InvVen_upper = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['InvVen_upper'].values[0])

    Deaths_mean = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['deaths_mean'].values[0])
    Deaths_lower = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['deaths_lower'].values[0])
    Deaths_upper = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['deaths_upper'].values[0])

    Admis_mean = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['admis_mean'].values[0])
    Admis_lower = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['admis_lower'].values[0])
    Admis_upper = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['admis_upper'].values[0])

    newICU_mean = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['newICU_mean'].values[0])
    newICU_lower = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['newICU_lower'].values[0])
    newICU_upper = StringToListFloat(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['newICU_upper'].values[0])

    outDF = pd.DataFrame({'FIPS':fipsNum, 'state':state, 'dates':[dates],
                    'allbed_mean':[allbed_mean], 'allbed_lower':[allbed_lower],
                    'allbed_upper':[allbed_upper], 'ICUbed_mean':[ICUbed_mean],
                    'ICUbed_lower':[ICUbed_lower], 'ICUbed_upper':[ICUbed_upper],
                    'InvVen_mean':[InvVen_mean], 'InvVen_lower':[InvVen_lower],
                    'InvVen_upper':[InvVen_upper], 'deaths_mean':[Deaths_mean],
                    'deaths_lower':[Deaths_lower], 'deaths_upper':[Deaths_upper],
                    'admis_mean':[Admis_mean], 'admis_lower':[Admis_lower], 'admis_upper':[Admis_lower],
                    'newICU_mean':[newICU_mean], 'newICU_lower':[newICU_lower], 'newICU_upper':[newICU_lower]
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


def getAaplCountyMobility(countyFIPS, countyMobilityDataframe):
    # function reads dataFrame retrives FIPS,state,data and driving mobility Information
    # Initial Author: Dio

    #dates = StringToListDate(hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['dates'].values[0])
    county = countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['county']
    dates = StringToListDate(countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['dates'].values[0])
    driving_mobility = StringToListFloat(countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['driving_mobility'].values[0])
    #creates data frame for the output
    outputFrame = pd.DataFrame({'FIPS':countyFIPS, 'state':county,
                    'dates':[dates], 'driving_mobility':[driving_mobility],})
    return outputFrame


def getAaplStateMobility(stateFIPS, stateMobilityDataframe):
    # function reads dataFrame retrives FIPS,state,data and driving mobility Information
    # Initial Author: Dio

    states = stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['state']
    dates = StringToListDate(stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['dates'].values[0])
    driving_mobility = StringToListFloat(stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['driving_mobility'].values[0])

    #creates data frame for the output
    outputFrame = pd.DataFrame({'FIPS':stateFIPS, 'state':states,'dates':[dates], 'driving_mobility':[driving_mobility],})
    #returns output for function
    return outputFrame


def getGoogleCountyMobility(countyFIPS, countyMobilityDataframe):
    # function dataFrame retries FIPS,county,dates and driving mobility information
    # Initial Author: Dio

    #dataframe is being used from another function
    county = countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['county']
    dates = StringToListDate(countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['dates'].values[0])
    #All of the percentages are changes from the baseline
    retail_recreation_Percent = StringToListFloat(countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS] ['retail_and_recreation_percent_change_from_baseline'].values[0])
    grocery_pharm_Percent = StringToListFloat(countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['grocery_and_pharmacy_percent_change_from_baseline'].values[0])
    parks_Percent = StringToListFloat(countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['parks_percent_change_from_baseline'].values[0])
    transit_stations_Percent = StringToListFloat(countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['transit_stations_percent_change_from_baseline'].values[0])
    workplace_Percent = StringToListFloat(countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['workplaces_percent_change_from_baseline'].values[0])
    residential_Percent = StringToListFloat(countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['residential_percent_change_from_baseline'].values[0])

    outputFrame = pd.DataFrame({'FIPS':countyFIPS, 'county':county,
                    'dates':[dates], 'retail_recreation_Percent':[retail_recreation_Percent],
                               'grocery_pharm_Percent':[grocery_pharm_Percent],'parks_Percent':[parks_Percent],
                               'transit_stations_Percent':[transit_stations_Percent],'residential_Percent':[residential_Percent],
                               'workplace_Percent':[workplace_Percent]})
    return outputFrame


def getGoogleStateMobility(stateFIPS, stateMobilityDataframe):
    # function dataframe retries FIPS,county,dates and driving mobility information
    # Initial Author: Dio

    #dataframe is being used from another function
    State = stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['state']
    dates = StringToListDate(stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['dates'].values[0])
    #All of the percentages are changes from the baseline
    retail_recreation_Percent = StringToListFloat(stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['retail_and_recreation_percent_change_from_baseline'].values[0])
    grocery_pharm_Percent = StringToListFloat(stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['grocery_and_pharmacy_percent_change_from_baseline'].values[0])
    parks_Percent = StringToListFloat(stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['parks_percent_change_from_baseline'].values[0])
    transit_stations_Percent = StringToListFloat(stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['transit_stations_percent_change_from_baseline'].values[0])
    workplace_Percent = StringToListFloat(stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['workplaces_percent_change_from_baseline'].values[0])
    residential_Percent = StringToListFloat(stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['residential_percent_change_from_baseline'].values[0])

    outputFrame = pd.DataFrame({'FIPS':stateFIPS, 'state':State,
                    'dates':[dates], 'retail_recreation_Percent':[retail_recreation_Percent],
                               'grocery_pharm_Percent':[grocery_pharm_Percent],'parks_Percent':[parks_Percent],
                               'transit_stations_Percent':[transit_stations_Percent],'residential_Percent':[residential_Percent],
                               'workplace_Percent':[workplace_Percent]})
    return outputFrame


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
                 'workplaces_percent_change_from_baseline' : 'residential_Percent',
                 'residential_percent_change_from_baseline' : 'workplace_Percent' }

    # Fix the columns that are strings that should be lists in the dataframe
    googMobilityCountyFrame.rename(columns=newnames, errors="raise", inplace=True)
    googMobilityStateFrame.rename(columns=newnames, errors="raise", inplace=True)

    return googMobilityCountyFrame, googMobilityStateFrame
