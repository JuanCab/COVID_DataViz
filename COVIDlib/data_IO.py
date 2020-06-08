##
## This library file contains routines to import COVID datafile created
## by the Collect_Data notebook.  This allows running with local copies
## of the data instead of requiring retrieval of data from the internet
## with every run of the COVID Visualization software.
##

import numpy as np
import pandas as pd
from datetime import datetime 


## STRING TO LIST FUNCTIONS
## The following functions convert strings in the saved CSV files into
## lists of various types


def StringToListDate(string):
    # Converts string to list of date objects
    # Initial Author: Juan
    dates_str_list = string.replace('\'','').strip('][').split(', ')
    dates_list = [datetime.strptime(date, '%Y-%m-%d').date() for date in dates_str_list]
    return dates_list


def StringToListInt(integer):
    # Converts string to list of ints
    # Initial author: Juan
    int_list = list(map(int, integer.replace('\'','').strip('][').split(', '))) 
    return int_list


def StringToListFloat(flt):
    # Converts string to list of floats
    # Initial Author: Juan
    flt_list = list(map(float, flt.replace('\'','').strip('][').split(', ')))
    return flt_list


## JOHN HOPKINS DATA IO
## The following block of routines are designed for reading in John Hopkins
## data on confirmed cases, deaths, and recovered numbers.


def GetCDRDataFrames(stateFile = 'our_data/statelevel_combinedCDR.csv', countyFile = 'our_data/countylevel_combinedCDR.csv'):
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

    confirmed = stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['Confirmed']
    deaths = stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['Deaths']
    recovered = stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['Recovered']

    dConfirmed = stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['dConfirmed']
    d2Confirmed = stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['d2Confirmed']
    dDeaths = stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['dDeaths'] 
    d2Deaths = stateDataFrame[stateDataFrame['FIPS'] == stateFIPS]['d2Deaths']

    outDF = pd.DataFrame({'FIPS':stateFIPS, 'state':state,
                    'Confirmed':confirmed, 'Deaths':deaths,
                    'Recovered':recovered, 'dConfirmed':dConfirmed,
                    'd2Confirmed':d2Confirmed, 'dDeaths':dDeaths,
                    'd2Deaths':d2Deaths})
    return outDF


def GetCDRCounty(countyFIPS, countyDataFrame):
    # Gets number of confirmed cases, deaths and recoveries at the state level
    # Note: This function requires calling the GetCDRDataFrames first; this uses the second data frame returned
    # Initial author: Luke
    county = countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['County']

    confirmed = countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['Confirmed']
    deaths = countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['Deaths']
    recovered = countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['Recovered']

    dConfirmed = countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['dConfirmed']
    d2Confirmed = countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['d2Confirmed']
    dDeaths = countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['dDeaths'] 
    d2Deaths = countyDataFrame[countyDataFrame['FIPS'] == countyFIPS]['d2Deaths']

    outDF = pd.DataFrame({'FIPS':countyFIPS, 'County':county,
                    'Confirmed':confirmed, 'Deaths':deaths,
                    'Recovered':recovered, 'dConfirmed':dConfirmed,
                    'd2Confirmed':d2Confirmed, 'dDeaths':dDeaths,
                    'd2Deaths':d2Deaths})
    return outDF


## IMHE DATA IO
## The following block of routines are designed for reading in IMHE
## data on confirmed cases, deaths, and recovered numbers.


def GetIMHEDataFrames(summaryFile = 'our_data/imhe_summary.csv', hospitalFile = 'our_data/imhe_hospitalizations.csv'):
    # This reads the local IMHE data files and returns data frames, which are to be used in the following functions
    summaryDF = pd.read_csv(summaryFile)
    hospitalizationsDF = pd.read_csv(hospitalFile)
    return summaryDF, hospitalizationsDF


def GetEquipData(fipsNum, summaryDataFrame):
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

    #print(peak_bed_day_mean, peak_bed_day_lower, peak_bed_day_upper)
    #print(peak_icu_bed_day_mean, peak_icu_bed_day_lower, peak_icu_bed_day_upper)
    #print(peak_vent_day_mean, peak_vent_day_lower, peak_vent_day_upper)
    #print(all_bed_capacity, icu_bed_capacity, all_bed_usage, icu_bed_usage)

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
    return mn_icu_beds # make a return


def GetNumAllBeds(fipsNum, summaryDataFrame):
    # Initial author: Luke

    #imhe_local_data = pd.read_csv(fileName)
    mn_icu_beds = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['all_bed_capacity']
    return mn_icu_beds # make a return


def GetICUBedUsage(fipsNum, summaryDataFrame):
    # Initial author: Luke

    #imhe_local_data = pd.read_csv(fileName)
    mn_icu_beds = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['icu_bed_usage']
    return mn_icu_beds # make a return


def GetAllBedUsage(fipsNum, summaryDataFrame):
    # Initial author: Luke

    #imhe_local_data = pd.read_csv(fileName)
    mn_icu_beds = summaryDataFrame[summaryDataFrame['FIPS'] == fipsNum]['all_bed_usage']
    return mn_icu_beds


def GetHospitalizationData(fipsNum, hospitalizationsDF):
    # Returns data from imhe_hospitalizations.csv as a data frame
    # Note: This function requires calling the GetIMHEData first; this uses the second data frame returned
    # Initial author: Luke

    #imhe_local_data = pd.read_csv(fileName)

    state = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['state']

    # Will the StringToList function be called here or later on...?
    allbed_mean = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['allbed_mean']
    allbed_lower = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['allbed_lower']
    allbed_upper = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['allbed_upper']

    ICUbed_mean = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['ICUbed_mean']
    ICUbed_lower = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['ICUbed_lower']
    ICUbed_upper = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['ICUbed_upper']

    InvVen_mean = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['InvVen_mean']
    InvVen_lower = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['InvVen_lower']
    InvVen_upper = hospitalizationsDF[hospitalizationsDF['FIPS'] == fipsNum]['InvVen_upper']

    outDF = pd.DataFrame({'FIPS':fipsNum, 'state':state,
                    'allbed_mean':allbed_mean, 'allbed_lower':allbed_lower,
                    'allbed_upper':allbed_upper, 'ICUbed_mean':ICUbed_mean,
                    'ICUbed_lower':ICUbed_lower, 'ICUbed_upper':ICUbed_upper,
                    'InvVen_mean':InvVen_mean, 'InvVen_lower':InvVen_lower,
                    'InvVen_upper':InvVen_upper})
    return outDF


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

    county = countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['county']   
    dates = countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['dates']
    driving_mobility = countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['driving_mobility']
    #creates data frame for the output
    outputFrame = pd.DataFrame({'FIPS':countyFIPS, 'state':county,
                    'dates':dates, 'driving_mobility':driving_mobility,})
    return outputFrame


def getAaplStateMobility(stateFIPS, stateMobilityDataframe):
    # function reads dataFrame retrives FIPS,state,data and driving mobility Information
    # Initial Author: Dio

    states = stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['state']
    dates = stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['dates']
    driving_mobility = stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['driving_mobility']

    #creates data frame for the output
    outputFrame = pd.DataFrame({'FIPS':stateFIPS, 'state':states,'dates':dates, 'driving_mobility':driving_mobility,})
    #returns output for function
    return outputFrame


def getGoogleCountyMobility(countyFIPS, countyMobilityDataframe):
    # function dataFrame retries FIPS,county,dates and driving mobility information
    # Initial Author: Dio

    #dataframe is being used from another function
    county = countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['county']   
    dates = countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['dates']
    #All of the percentages are changes from the baseline
    retail_recreation_Percent = countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['retail_and_recreation_percent_change_from_baseline']
    grocery_pharm_Percent = countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['grocery_and_pharmacy_percent_change_from_baseline']
    parks_Percent = countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['parks_percent_change_from_baseline']
    transit_stations_percent = countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['transit_stations_percent_change_from_baseline']
    residential_percent = countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['workplaces_percent_change_from_baseline']
    workplace_percent = countyMobilityDataframe[countyMobilityDataframe['FIPS'] == countyFIPS]['residential_percent_change_from_baseline']

    outputFrame = pd.DataFrame({'FIPS':countyFIPS, 'county':county,
                    'dates':dates, 'retail_recreation_Percent':retail_recreation_Percent,
                               'grocery_pharm_Percent':grocery_pharm_Percent,'parks_Percent':parks_Percent,
                               'transit_stations_percent':transit_stations_percent,'residential_percent':residential_percent,
                               'workplace_percent':workplace_percent})
    return outputFrame


def getGoogleStateMobility(stateFIPS, stateMobilityDataframe):
    # function dataframe retries FIPS,county,dates and driving mobility information
    # Initial Author: Dio

    #dataframe is being used from another function
    State = stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['state']   
    dates = stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['dates']
    #All of the percentages are changes from the baseline
    retail_recreation_Percent = stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['retail_and_recreation_percent_change_from_baseline']
    grocery_pharm_Percent = stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['grocery_and_pharmacy_percent_change_from_baseline']
    parks_Percent = stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['parks_percent_change_from_baseline']
    transit_stations_percent = stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['transit_stations_percent_change_from_baseline']
    residential_percent = stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['workplaces_percent_change_from_baseline']
    workplace_percent = stateMobilityDataframe[stateMobilityDataframe['FIPS'] == stateFIPS]['residential_percent_change_from_baseline']

    outputFrame = pd.DataFrame({'FIPS':stateFIPS, 'State':State,
                    'dates':dates, 'retail_recreation_Percent':retail_recreation_Percent,
                               'grocery_pharm_Percent':grocery_pharm_Percent,'parks_Percent':parks_Percent,
                               'transit_stations_percent':transit_stations_percent,'residential_percent':residential_percent,
                               'workplace_percent':workplace_percent})
    return outputFrame
