##
## This is a set of functions for presenting COVID data to the user via a Dashboard
##

import math
import numpy as np
import matplotlib.pyplot as plt
import datetime
from IPython.core.display import display, HTML
import COVIDlib.data_IO as COVID_IO

##
## Define variables to be accessed within Dashboard functions
##

# Define size of legend text
legendsize = 10

# Define titles/labels in English for John Hopkins and Mobility Data
var_descript = {'FIPS' : 'Federal Information Processing Standards State/County Number',
                'county' : 'County Name',
                'state' : 'State Name',
                'Lat': 'Latitude',
                'Long_' : 'Longitude',
                'dates' : 'Dates',
                'Confirmed' : 'Total Confirmed COVID Infections',
                'Deaths' : 'Total Confirmed and Probable COVID Deaths',
                'Recovered' : 'Total Confirmed and Probable COVID Recoveries',
                'Active' : 'Total Confirmed and Probable Active COVID Cases',
                'Incident_Rate' : 'Confirmed COVID Cases',  # As provided by John Hopkins
                'dTested' : 'New COVID Tests Administered (#/day)',
                'dTestedWk' : 'New COVID Tests Administered (#/week)',
                'People_Tested' : 'Total People tested for COVID',
                'Testing_Rate' : 'Total Tested for COVID (per 100,000 persons)',
                'People_Hospitalized' : 'Total People Hospitalized for COVID',
                'Mortality_Rate' : 'Mortality Rate',
                'Hospitalization_Rate' : 'Hospitalization Rate',
                'dConfirmed' : 'New COVID Infections (#/day)',
                'd2Confirmed' : 'Change in New COVID Infections',
                'dConfirmedWk' : 'New COVID Infections (#/week)',
                'd2ConfirmedWk' : 'Change in New COVID Infections',
                'dDeaths' : 'New COVID Deaths (#/day)',
                'd2Deaths' : 'Change in New COVID Deaths',
                'dDeathsWk' : 'New COVID Deaths (#/week)',
                'd2DeathsWk' : 'Change in New COVID Deaths',
                'dConfirmedRate' : 'New COVID Infections (#/day) (per 100,000 persons)',
                'd2ConfirmedRate' : 'Change in New COVID Infections (per 100,000 persons)',
                'dDeathsRate': 'New COVID Deaths (#/day) (per 100,000 persons)',
                'd2DeathsRate' : 'Change in New COVID Deaths (per 100,000 persons)',
                'dConfirmedWkRate' : 'New COVID Infections (#/week) (per 100,000 persons)',
                'd2ConfirmedWkRate' : 'Change in New COVID Infections (per 100,000 persons)',
                'dDeathsWkRate': 'New COVID Deaths (#/week) (per 100,000 persons)',
                'd2DeathsWkRate' : 'Change in New COVID Deaths (per 100,000 persons)',
                'PopEst2019' : 'Estimated Population (July 1, 2019)',
                'PopChg2019' : 'Estimated Population Increase (2018-19)',
                'ConfirmedRate' : 'Total Confirmed COVID Infections (per 100,000 persons)', # As computed by us
                'DeathRate' : 'Total Confirmed COVID Deaths (per 100,000 people)', # As computed by us
                'driving_mobility' : 'Apple Maps Directions Requests',
                'driving_mobility_Percent' : 'Apple Maps Directions Requests',
                'retail_recreation_Percent': 'Google-tracked Retail & Recreation Activity',
                'grocery_pharm_Percent': 'Google-tracked Grocery & Pharmacy Activity',
                'parks_Percent': 'Google-tracked Park Activity',
                'transit_stations_Percent' : 'Google-tracked Transit Station Activity',
                'residential_Percent' : 'Google-tracked Residential Activity',
                'workplace_Percent' : 'Google-tracked Workplace Activity' }

var_ylabel = {'FIPS' : 'FIPS Number',
                'county' : '',
                'state' : '',
                'Lat': 'Latitude (degrees)',
                'Long_' : 'Longitude (degrees)',
                'Dates' : 'Dates',
                'Confirmed' : 'Confirmed Infections',
                'Deaths' : 'Confirmed & Probable Deaths',
                'Recovered' : 'Confirmed & Probable Recoveries',
                'Active' : 'Confirmed & Probable Active Cases',
                'Incident_Rate' : 'Confirmed Cases (per 100,000 persons)',  # As provided by John Hopkins
                'dTested' : 'New Tests (#/day)',
                'dTestedWk' : 'New Tests (#/week)',
                'People_Tested' : 'Total Tested',
                'Testing_Rate' : 'Total Tested (per 100,000 people)',
                'People_Hospitalized' : 'Total hospitalized for COVID',
                'Mortality_Rate' : 'Deaths/Confirmed Infection (%)',
                'Hospitalization_Rate' : 'Hospitalized/Confirmed Infection (%)',
                'dConfirmed' : 'New Infections per day',
                'd2Confirmed' : 'New Infections/Day vs. Previous',
                'dConfirmedWk' : 'New Infections in past week',
                'd2ConfirmedWk' : 'New Infections/Week vs. Previous',
                'dDeaths' : 'New Deaths per day',
                'd2Deaths' : 'Change in New Deaths (Deaths/day per day)',
                'dDeathsWk' : 'New Deaths in past week',
                'd2DeathsWk' : 'Change in New Deaths (Deaths/week per week)',
                'dConfirmedRate' : 'New Infections/Day (per 100,000 people)',
                'dDeathsRate': 'New Deaths/Day (per 100,000 people)',
                'd2ConfirmedRate' : 'New Infections/Day vs. Previous',
                'd2DeathsRate' : 'New Deaths/Day vs. Previous',
                'dConfirmedWkRate' : 'New Infections/Week (per 100,000 people)',
                'dDeathsWkRate': 'New Deaths/Week (per 100,000 people)',
                'd2ConfirmedWkRate' : 'New Infections/Week vs. Previous',
                'd2DeathsWkRate' : 'New Deaths/Week vs. Previous',
                'PopEst2019' : 'Estimated Population',
                'PopChg2019' : 'Estimated Population Increase',
                'ConfirmedRate' : 'COVID Infections (per 100,000 persons)', # As computed by us
                'DeathRate' : 'COVID Deaths (per 100,000 people)', # As computed by us
                'driving_mobility' : 'Relative volume (vs Jan 13, 2020)',
                'driving_mobility_Percent' : 'Percent Change (vs Jan 13, 2020)',
                'retail_recreation_Percent': 'Percent Change (vs Jan 3 - Feb 6, 2020)',
                'grocery_pharm_Percent': 'Percent Change (vs Jan 3 - Feb 6, 2020)',
                'parks_Percent': 'Percent Change (vs Jan 3 - Feb 6, 2020)',
                'transit_stations_Percent' : 'Percent Change (vs Jan 3 - Feb 6, 2020)',
                'residential_Percent' : 'Percent Change (vs Jan 3 - Feb 6, 2020)',
                'workplace_Percent' : 'Percent Change (vs Jan 3 - Feb 6, 2020)' }

# Titles for plots of IMHE Hospital Data 
var_descript_Hosp = {'FIPS' : 'Federal Information Processing Standards State/County Number',
                'county' : 'County Name',
                'state' : 'State Name',
                'dates' : 'Dates',
                'allbed_mean': 'IMHE Predicted Number of Hospital Beds Needed',
                'ICUbed_mean' : 'IMHE Predicted Number of ICU beds Needed',
                'InvVen_mean' : 'IMHE Predicted Number of Ventilators Needed',
                'deaths_mean' : 'IMHE Predicted Number of Deaths',
                'admis_mean' : 'IMHE Predicted New Hospital Admissions per Day',
                'newICU_mean' : 'IMHE Predicted New ICU Patients per Day',
                'totdea_mean' : 'IMHE Predicted Cumilative COVID Deaths',
                'deaths_mean_smoothed': 'IMHE Predicted Daily COVID Deaths (Smoothed)',
                'totdea_mean_smoothed' : 'IMHE Predicted Cumilative COVID Deaths (Smoothed)',
                'total_tests_data_type' : 'Test Type',
                'total_tests' : 'IMHE Predicted Total Tests',
                # 'confirmed_infections' : 'Observed Confirmed COVID Infections', # Removed as redundant data
                'est_infections_mean' : 'IMHE Predicted COVID Infections' }

# dictionary tracking variables with ranges and those without
var_ranges_Hosp = {'FIPS' : False,
                'county' : False,
                'state' : False,
                'dates' : False,
                'allbed_mean': True,
                'ICUbed_mean' : True,
                'InvVen_mean' : True,
                'deaths_mean' : True,
                'admis_mean' : True,
                'newICU_mean' : True,
                'totdea_mean': True,
                'deaths_mean_smoothed': True,
                'totdea_mean_smoothed' : True,
                'total_tests_data_type' : False,
                'total_tests' : False,
                'confirmed_infections' : False,
                'est_infections_mean' : True,}

#Y label for plots in Hospital Data
var_ylabel_Hosp ={'FIPS' : 'Federal Information Processing Standards State/County Number',
                'county' : 'County Name',
                'state' : 'State Name',
                'dates' : 'Dates',
                'allbed_mean': 'Number of Beds',
                'ICUbed_mean' : 'Number of ICU beds',
                'InvVen_mean' : 'Number of Venitlators',
                'deaths_mean' : 'Number of Deaths',
                'admis_mean' : 'Number of Hospital Admissions',
                'newICU_mean' : 'Number of New ICU Patients',
                'totdea_mean' : 'Cumilative COVID Deaths',
                'deaths_mean_smoothed': 'Daily COVID Deaths',
                'totdea_mean_smoothed' : 'Cumilative COVID Deaths',
                'total_tests_data_type' : 'Test Type',
                'total_tests' : 'Total Tests',
                'confirmed_infections' : 'Confirmed COVID Infections',
                'est_infections_mean' : 'Predicted COVID Infections' }


def build_fipsdict(county_df, state_df):
    # This function converts the John Hopkins state and county dataframes into a dictionary
    # that can be used to look up FIPS numbers by location name.

    # Start by converting states dataframe into a dictionary
    FIPSdf = state_df[['FIPS','state']].copy()
    FIPSdict = FIPSdf.set_index('state').T.to_dict('records')[0]

    # Pull FIPS, county, and state columns from county dataframe, then
    # construct placenames and use that to append additional dictionary
    # entries for all the counties.
    FIPSdf = county_df[['FIPS','county','state']].copy()
    FIPSdf['placename'] = FIPSdf.agg('{0[county]} ({0[state]})'.format, axis=1)
    FIPSdf.drop(columns=['county', 'state'], inplace=True)
    FIPSdict.update(FIPSdf.set_index('placename').T.to_dict('records')[0])

    return FIPSdict


def GetFIPS(FIPSdict, state, county=None):
    # Return the FIPS for this county/state combination automatically from FIPSdict
    if (county == None):
        return FIPSdict[state]
    else:
        placename = f'{county} ({state})'
        return FIPSdict[placename]


def BuildStatesList():
    #
    # This is a dopey way to just define this list in the library instead of the main block of
    # code.
    #
    AllStates = [ 'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
                'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida',
                'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas',
                'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
                'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 
                'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 
                'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 
                'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 
                'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 
                'Wyoming' ]
    return AllStates


def BuildCountiesList(dataframe, AllStates):
    #
    # Construct list of counties in each state from a stateframe containing the full list
    #
    CountiesDict = {}
    for state in AllStates:
        CountiesDict[state] =  sorted(dataframe[dataframe['state'] == state]['county'].tolist())

    return CountiesDict


def BuildJHVarDict():
    # Load the John Hopkins time series variables information dictionary into memory
    #  descript is dropmenuworthy description of variable
    #  valdescript is used when printing actual values after the variable
    #  format is the formatting string to use when displaying variable
    #  stateonly notes if the variable is avaiable only for states
    #  df is an indicator of which dataframe to pull the data from ('JH' for John Hopkins)
    var_dict = {
        'Confirmed': {'descript': 'Total Confirmed COVID Infections', 'valdescript': 'COVID Infections', 'format' : ',d', 'stateonly': False, 'df': 'JH'},
        'ConfirmedRate': {'descript': 'Total Confirmed COVID Infections (per 100,000 persons)', 'valdescript': 'COVID Infections (per 100,000)', 'format' : '.2f', 'stateonly': False, 'df': 'JH'},
        'dConfirmed': {'descript': 'New COVID Infections (#/day)', 'valdescript': 'New COVID Infections/Day', 'format' : 'd', 'stateonly': False, 'df': 'JH'},
        'dConfirmedRate': {'descript': 'New COVID Infections (#/day per 100,000 people)', 'valdescript': 'New COVID Infections/Day (per 100,000)', 'format' : '.2f', 'stateonly': False, 'df': 'JH'},
        'd2Confirmed': {'descript': 'Change in New COVID Infections', 'valdescript': 'New COVID Infections/Day vs. Previous', 'format' : '+d', 'stateonly': False, 'df': 'JH'},
        'd2ConfirmedRate': {'descript': 'Change in New COVID Infections (per 100,000 people)', 'valdescript': 'New COVID Infections/Day vs. Previous', 'format' : '+.2f', 'stateonly': False, 'df': 'JH'},
        'dConfirmedWk': {'descript': 'New COVID Infections (#/week)', 'valdescript': 'New COVID Infections/Week', 'format' : 'd', 'stateonly': False, 'df': 'JH'},
        'dConfirmedWkRate': {'descript': 'New COVID Infections (#/week per 100,000 people)', 'valdescript': 'New COVID Infections/Week (per 100,000)', 'format' : '.2f', 'stateonly': False, 'df': 'JH'},
        'd2ConfirmedWk': {'descript': 'Change in New COVID Infections/week', 'valdescript': 'New COVID Infections/Week vs. Previous', 'format' : '+d', 'stateonly': False, 'df': 'JH'},
        'd2ConfirmedWkRate': {'descript': 'Change in New COVID Infections/week (per 100,000 people)', 'valdescript': 'New COVID Infections/Week vs. Previous', 'format' : '+.2f', 'stateonly': False, 'df': 'JH'},
        'Deaths': {'descript': 'Total Confirmed and Probable COVID Deaths', 'valdescript': 'COVID Deaths', 'format' : ',d', 'stateonly': False, 'df': 'JH'},
        'DeathRate': {'descript': 'Total Confirmed COVID Deaths (per 100,000 people)', 'valdescript': 'COVID Deaths (per 100,000)', 'format' : '.2f', 'stateonly': False, 'df': 'JH'},
        'dDeaths': {'descript': 'New COVID Deaths (#/day)', 'valdescript': 'New COVID Deaths/Day', 'format' : 'd', 'stateonly': False, 'df': 'JH'},
        'dDeathsRate': {'descript': 'New COVID Deaths (#/day per 100,000 people)', 'valdescript': 'New COVID Deaths/Day (per 100,000)', 'format' : '.2f', 'stateonly': False, 'df': 'JH'},
        'dDeathsWk': {'descript': 'New COVID Deaths (#/week)', 'valdescript': 'New COVID Deaths/Week', 'format' : 'd', 'stateonly': False, 'df': 'JH'},
        'dDeathsWkRate': {'descript': 'New COVID Deaths (#/week per 100,000 people)', 'valdescript': 'New COVID Deaths/Week (per 100,000)', 'format' : '.2f', 'stateonly': False, 'df': 'JH'},
        'd2Deaths': {'descript': 'Change in New COVID Deaths', 'valdescript': 'New COVID Deaths/Day vs. Previous Day', 'format' : 'd', 'stateonly': False, 'df': 'JH'},
        'd2DeathsRate': {'descript': 'Change in New COVID Deaths (per 100,000 people)', 'valdescript': 'New COVID Deaths/Day vs. Previous Day (per 100,000)', 'format' : '.2f', 'stateonly': False, 'df': 'JH'},
        'd2DeathsWkRate': {'descript': 'Change in New COVID Deaths/week (per 100,000 people)', 'valdescript': 'New COVID Deaths/Week vs. Previous', 'format' : '.2f', 'stateonly': False, 'df': 'JH'},
        'Recovered': {'descript': 'Total Confirmed and Probable COVID Recoveries [State only]', 'valdescript': 'COVID Recoveries', 'format' : ',d', 'stateonly': True, 'df': 'JH'},
        'Active': {'descript': 'Total Confirmed and Probable Active COVID Cases [State only]', 'valdescript': 'Active COVID Cases', 'format' : ',d', 'stateonly': True, 'df': 'JH'},
        'dTested': {'descript': 'New Tests Administered (#/day) [State only]', 'valdescript': 'People tested/Day', 'format' : ',d', 'stateonly': True, 'df': 'JH'},
        'dTestedWk': {'descript': 'New Tests Administered (#/week) [State only]', 'valdescript': 'People tested/Week', 'format' : 'd', 'stateonly': True, 'df': 'JH'},
        'People_Tested': {'descript': 'Total Number of COVID Test [State only]', 'valdescript': 'Number of COVID Tests', 'format' : ',d', 'stateonly': True, 'df': 'JH'},
        'Testing_Rate': {'descript': 'Total Tested for COVID (per 100,000 persons) [State only]', 'valdescript': 'People tested for COVID (per 100,000)', 'format' : 'd', 'stateonly': True, 'df': 'JH'},
        'Mortality_Rate': {'descript': 'Mortality Rate [State only]', 'valdescript': '% Deaths per Confirmed Infection', 'format' : '.1f', 'stateonly': True, 'df': 'JH'},
        'People_Hospitalized': {'descript': 'Total People Hospitalized for COVID [State only]', 'valdescript': 'People Hospitalized', 'format' : ',d', 'stateonly': True, 'df': 'JH'},
        'Hospitalization_Rate': {'descript': 'Hospitalization Rate (Hospitalized/Infected) [State only]', 'valdescript': 'Percent Hospitalized', 'format' : '.2f', 'stateonly': True, 'df': 'JH'}
        }
    return var_dict


def BuildMobilityVarDict():
    # Load the Mobility time series variables information dictionary into memory
    #  descript is dropmenuworthy description of variable
    #  valdescript is used when printing actual values after the variable
    #  format is the formatting string to use when displaying variable
    #  stateonly notes if the variable is avaiable only for states
    #  df is an indicator of which dataframe to pull the data from ('apple' for Apple Mobility, 'google' for Google Mobility)
    var_dict = {
        'driving_mobility_Percent': {'descript': 'Apple Maps Directions Requests', 'valdescript': '% Directions Requests (vs Jan 13, 2020)', 'format': '+.1f', 'stateonly': False, 'df': 'apple'},
        'retail_recreation_Percent': {'descript': 'Google-tracked Retail & Recreation Activity', 'valdescript': '% Retail & Recreation Activity (vs Jan 3 - Feb 6, 2020)', 'format': '+.1f', 'stateonly': False, 'df': 'google'},
        'grocery_pharm_Percent': {'descript': 'Google-tracked Grocery & Pharmacy Activity', 'valdescript': '% Grocery & Pharmacy Activity (vs Jan 3 - Feb 6, 2020)', 'format': '+.1f', 'stateonly': False, 'apple': False, 'df': 'google'},
        'parks_Percent': {'descript': 'Google-tracked Park Activity', 'valdescript': '% Park Activity (vs Jan 3 - Feb 6, 2020)', 'format': '+.1f', 'stateonly': False, 'apple': False, 'df': 'google'},
        'transit_stations_Percent': {'descript': 'Google-tracked Transit Station Activity', 'valdescript': '% Transit Station Activity (vs Jan 3 - Feb 6, 2020)', 'format': '+.1f', 'stateonly': False, 'apple': False, 'df': 'google'},
        'residential_Percent': {'descript': 'Google-tracked Residential Activity', 'valdescript': '% Residential Activity (vs Jan 3 - Feb 6, 2020)', 'format': '+.1f', 'stateonly': False, 'apple': False, 'df': 'google'},
        'workplace_Percent': {'descript': 'Google-tracked Workplace Activity', 'valdescript': '% Workplace Activity (vs Jan 3 - Feb 6, 2020)', 'format': '+.1f', 'stateonly': False, 'apple': False, 'df': 'google'}
        }
    return var_dict


def BuildIMHEHospitalizationVarDict():
    # Load the IMHE Hospitalizations time series variables information into memory
    var_dict = var_descript_Hosp.copy()

    # Delete unneeded keys
    del var_dict['FIPS']
    del var_dict['county']
    del var_dict['state']
    del var_dict['dates']
    del var_dict['total_tests_data_type']
    return var_dict


def html_status(dataframe, fips, hospital_summary_df=None, Rt_df=None, BedsStatus=True, Predictions=True, Display=True):
    ## Print an HTML statement of current status (Confirmed, Deaths, Recovered)
    ## based on Johns Hopkins dataframes (county or State)

    ## Check if FIPS input is reasonable
    if (type(fips) == int):
        fips = [fips]
    elif (type(fips) != list):
        raise ValueError('Input fips must be integer or list of integers')

    # Loop through the FIPS values
    html_out = "" # Start with no HTML
    for FIPS in fips:
        # If this is DC county FIPS, skip is (you should use the state one)
        if (FIPS == 11001):
            continue

        # Retrieve local data
        local_df = COVID_IO.getLocalDataFrame(FIPS, dataframe)

        # Determine name to use
        if (FIPS > 100):
            namestr = f"{local_df['county'].values[0]} ({local_df['state'].values[0]})"
        else:
            # This is a state
            namestr = local_df['state'].values[0]

        # For each list item, remember its a nested list, so pull the list out [0] and then reference the
        # last item in the nested list using index -1.
        last_day = local_df['dates'].to_list()[0][-1].strftime("%B %d, %Y")
        last_infect_tot = local_df['Confirmed'].to_list()[0][-1]
        last_recovered_tot = local_df['Recovered'].to_list()[0][-1]
        last_active_tot = local_df['Active'].to_list()[0][-1]
        last_death_tot = local_df['Deaths'].to_list()[0][-1]
        dead_percent = (last_death_tot/last_infect_tot)*100
        recovered_percent = (last_recovered_tot/last_infect_tot)*100
        active_percent = (last_active_tot/last_infect_tot)*100
        last_infectrate = local_df['ConfirmedRate'].to_list()[0][-1]
        last_deathrate = local_df['DeathRate'].to_list()[0][-1]
        last_infect_change = local_df['dConfirmed'].to_list()[0][-1]
        last_death_change = local_df['dDeaths'].to_list()[0][-1]
        last_infect_change2 = local_df['d2Confirmed'].to_list()[0][-1]
        last_death_change2 = local_df['d2Deaths'].to_list()[0][-1]
        last_infect_change_rate = local_df['dConfirmedRate'].to_list()[0][-1]
        last_death_change_rate = local_df['dDeathsRate'].to_list()[0][-1]
        last_infect_change2_rate = local_df['d2ConfirmedRate'].to_list()[0][-1]
        last_death_change2_rate = local_df['d2DeathsRate'].to_list()[0][-1]

        # Build HTML report
        scale_title= "140%" # Scale for title text
        scale_enhance = "120%" # Scale for important text
        scale_enhance2 = "105%" # Scale for important but less important text

        html_out += "<p style='margin: 1em 0 0 0;'>"
        html_out += f"<b style='font-size: {scale_title};'>{namestr} as of {last_day}</b><br/>"
        html_out += "<div style='margin: 0 0 0 2em; line-height: 1.2em;'>"
        html_out += f"<b style='font-size: {scale_enhance};'>{last_infect_tot:,.0f} Total Cases</b> ({last_infectrate} per 100,000 people)<br/>"

        # Since entire US recovered value is not accurate (since it involves summing up
        # known state values, and some states are not reporting this), force it to not
        # display...
        if (FIPS == 0):
            last_recovered_tot = 0

        # Don't try to print recovered/active stats if they are bogus
        if ((last_active_tot < 0)|(last_recovered_tot == 0)|(math.isnan(last_recovered_tot))):
            html_out += f"<b style='font-size: {scale_enhance2};'>{last_death_tot:,.0f} Dead ({dead_percent:.1f}%)</b><br/>"
        else:
            html_out += f"<b style='color:#ff0000;font-size: {scale_enhance2}'>{last_active_tot:,.0f} Active ({active_percent:.1f}%)</b> <b>/</b> <b style='color:rgb(0,128,20);font-size: {scale_enhance2};'>{last_recovered_tot:,.0f} Recovered ({recovered_percent:.1f}%)</b> <b>/</b> <b style='font-size: {scale_enhance2};'>{last_death_tot:,.0f} Dead ({dead_percent:.1f}%)</b><br/>"

        # # Add Rt information if passed Rt dataframe
        # if ((FIPS>0)&(FIPS < 100)&(Rt_df is not None)):
        #         status = str(html_status_Rt(Rt_df, FIPS, Display=False))
        #         html_out += f"<span style='font-size: {scale_enhance2}'>{status}</span>"

        # Present last day stats
        html_out += "<b>New Cases [change from previous day]:</b><br/>"
        html_out += f"<li><b>{last_infect_change:,.0f} [{last_infect_change2:+,.0f}] new infections</b> ({last_infect_change_rate:,.2f} [{last_infect_change2_rate:+,.2f}] per 100,000 people).</li>"
        html_out += f"<li><b>{last_death_change:,.0f} [{last_death_change2:+,.0f}] new deaths</b> ({last_death_change_rate:,.2f} [{last_death_change2_rate:+,.2f}] per 100,000 people).</li>"
        # If a hospitalization summary dataframe is provided, process it and produce HTML report
        if ((FIPS>0)&(FIPS < 100)&(hospital_summary_df is not None)&(BedsStatus)):
            html_out += str(html_status_beds(hospital_summary_df, FIPS, Display=False))
        if ((FIPS>0)&(FIPS < 100)&(hospital_summary_df is not None)&(Predictions)):
                html_out += str(html_IHME_Predictions(hospital_summary_df, FIPS, Display=False))
        html_out += "</div></p>"

    if (Display):
        # Display HTML to screen
        display(HTML(html_out))
        return
    else:
        return html_out

    return


# def html_status_Rt(dataframe, fips, Display=True):
#     ## Print the current estimate R_t reproduction rate for this state

#     ## Check if FIPS input is reasonable
#     if (type(fips) == int):
#         fips = [fips]
#     elif (type(fips) != list):
#         raise ValueError('Input fips must be integer or list of integers')

#     # Loop through the FIPS values for states
#     html_out = ""  # Start with blank string for HTML

#     # Deal with accidentally passing in US or county FIPS values to list
#     fips = [FIPS for FIPS in fips if (FIPS >0)&(FIPS<100)]
#     for FIPS in fips:
#         # Get state name
#         local_df = COVID_IO.getLocalDataFrame(FIPS, dataframe)
#         namestr = local_df['state'].values[0]
#         last_day = local_df['dates'].to_list()[0][-1].strftime("%B %d, %Y")
#         # current_Rt = local_df['Rt_mean'].to_list()[0][-1]
#         # current_lower = local_df['Rt_lower_80'].to_list()[0][-1]
#         # current_upper = local_df['Rt_upper_80'].to_list()[0][-1]

#         # Highligh if R_t above 1
#         if (current_Rt>1):
#             clr = "#ff0000"
#         else:
#             clr = "#000000"

#         # Print HTML report
#         html_out += f"<b style='color:{clr};'>R<sub>t</sub>={current_Rt:.2f}</b> (80% chance between {current_lower:.2f} & {current_upper:.2f}) on {last_day}<br/>"
#         if (Display):
#             display(HTML(html_out))
#             return
#         else:
#             return html_out


def html_status_beds(dataframe, fips, Display=True):
    ## Print an HTML statement of current status (Confirmed, Deaths, Recovered)
    ## based on Johns Hopkins dataframes (county or State)

    ## Check if FIPS input is reasonable
    if (type(fips) == int):
        fips = [fips]
    elif (type(fips) != list):
        raise ValueError('Input fips must be integer or list of integers')

    # Loop through the FIPS values for states
    html_out = ""  # Start with blank string for HTML
    # Deal with accidentally passing in US or county FIPS values to list
    fips = [FIPS for FIPS in fips if (FIPS >0)&(FIPS<100)]
    for FIPS in fips:
        # Get state name
        local_df = COVID_IO.getLocalDataFrame(FIPS, dataframe)
        namestr = local_df['state'].values[0]

        # Getting total of beds and number of beds used
        num_icu_beds_used = int(COVID_IO.GetICUBedUsage(FIPS, dataframe))
        num_icu_beds_total = int(COVID_IO.GetNumICUBeds(FIPS, dataframe))
        num_reg_beds_used = int(COVID_IO.GetAllBedUsage(FIPS, dataframe))
        num_reg_beds_total = int(COVID_IO.GetNumAllBeds(FIPS, dataframe))

        # Calculating percent of beds used
        percent_icu_used = float(num_icu_beds_used / num_icu_beds_total) * 100
        percent_reg_used = float(num_reg_beds_used / num_reg_beds_total) * 100

        # Print HTML report
        if (Display):
            html_out += f"<b>Status of Hospital Beds in {namestr}</b>"
            html_out += f"<table style='padding: 5px;'>"
            html_out += f"<tr><td style='text-align: left;vertical-align: top;'><b style='font-size: 140%;'>{num_reg_beds_used} of {num_reg_beds_total} Hospital Beds ({percent_reg_used:,.2f}%) Being Used.</b></td>"
            html_out += f"<tr><td style='text-align: left;vertical-align: top;'><b style='font-size: 140%;'>{num_icu_beds_used} of {num_icu_beds_total} ICU Beds ({percent_icu_used:,.2f}%) Being Used.</b></td>"
            html_out += "</td></tr></table>"
            display(HTML(html_out))
            return
        else:
            # Build HTML to returm for inclusion in other report
            html_out += "<div style='text-indent: 1em;'>"
            html_out += f"<b>Status of Hospital Beds in {namestr}</b>"
            html_out += f"<li>{num_reg_beds_used} of {num_reg_beds_total} Hospital Beds ({percent_reg_used:,.2f}%) in use.</li>"
            html_out += f"<li>{num_icu_beds_used} of {num_icu_beds_total} ICU Beds ({percent_icu_used:,.2f}%) in use.</li>"
            html_out += str(html_IHME_Predictions(dataframe, FIPS, Display=False))
            html_out += "</div>"
            return html_out


def html_IHME_Predictions(dataframe, fips, Display = True):
    # function prints out IHME predictions or if called from html_status_beds using display=False 
    # it just returns HTML.

    ## Check if FIPS input is reasonable
    if (type(fips) == int):
        fips = [fips]
    elif (type(fips) != list):
        raise ValueError('Input fips must be integer or list of integers')

    empList = []
    # Loop through the FIPS values for states
    html_out = ""  # Start with blank string for HTML
    # Deal with accidentally passing in US or county FIPS values to list
    fips = [FIPS for FIPS in fips if (FIPS >0)&(FIPS<100)]
    for FIPS in fips:
        # Get state name
        local_df = COVID_IO.getLocalDataFrame(FIPS, dataframe)
        namestr = local_df['state'].values[0]

        # gets peak mean day  predictions for icu,beds and vents
        peakBedDay = local_df['peak_bed_day_mean'].to_list()[0].strftime("%B %d, %Y")
        PeakIcuDay = local_df['peak_icu_bed_day_mean'].to_list()[0].strftime("%B %d, %Y")
        peakVentDay = local_df['peak_vent_day_mean'].to_list()[0].strftime("%B %d, %Y")
        # Will have to add more if needed

        # Is this in the past or in the future
        today = datetime.datetime.now().date()
        if (local_df['peak_bed_day_mean'].to_list()[0].date() > today):
            bed_verb = "to peak"
        else:
            bed_verb = "to have peaked"
        if (local_df['peak_icu_bed_day_mean'].to_list()[0].date() > today):
            icu_verb = "to peak"
        else:
            icu_verb = "to have peaked"
        if (local_df['peak_vent_day_mean'].to_list()[0].date() > today):
            vent_verb = "to peak"
        else:
            vent_verb = "to have peaked"

        # Print HTML report
        if (Display):
            html_out += f"<h3>Resource Use Predictions for the state of {namestr} by IMHE</h3>"
            html_out += f"<table style='padding: 5px;'>"
            html_out += f"<tr><td style='text-align: left;vertical-align: top;'><b style='font-size: 140%;'>Hospital bed use predicted {bed_verb} {peakBedDay}</b><br/></td>"
            html_out += f"<tr><td style='text-align: left;vertical-align: top;'><b style='font-size: 140%;'>ICU bed use predicted {icu_verb} {PeakIcuDay}</b><br/></td>"
            html_out += f"<tr><td style='text-align: left;vertical-align: top;'><b style='font-size: 140%;'>Ventilator use predicted {vent_verb} {peakVentDay}</b><br/></td>"
            html_out += "</td></tr></table>"
            display(HTML(html_out))
        else:
            # Build HTML to add to list from html_bed_status()
            html_out += f"<b>Predicted Resource Use for {namestr} by IMHE</b>"
            html_out += f"<li><b>Hospital bed use</b> predicted {bed_verb} <b>{peakBedDay}</b>.</li>"
            html_out += f"<li><b>ICU bed use</b> predicted {icu_verb} <b>{PeakIcuDay}</b></li>"
            html_out += f"<li><b>Ventilator use</b> predicted {vent_verb} <b>{peakVentDay}</b></li>"
            return html_out


def running_average(ts_array, n_days):
    # Compute the running average of the last n_days for ts_array (assumed to be
    # a contiguous set of data with a one day cadence).  It assumes the input is
    # a time-series numpy array.

    ##
    ## Perform the running average as a sum of n_days+1 shifted time-series
    ##

    # Create array to store running average padded to deal with building a sum using
    # time-shifted data
    npts = ts_array.size
    running_avg_raw = np.zeros(npts+n_days+1, float)

    for m in range(0, n_days+1):
        running_avg_raw[m : m+npts] += ts_array[:]
    running_avg_raw[:] /= n_days+1   # Divide by n_days+1 to make actual running average

    # Grab the slice of running_avg_raw that actually corresponds to real dates.
    # WARNING: The first n_days entries here are going to be biased low, so inserting
    # np.nan there instead
    running_avg_raw[0:n_days] = np.nan

    return running_avg_raw[0:npts]


def ts_plot(dataframe, colname, fips, connectdots=False, ylog=False, running_avg=0, last90=False, fig=None, ax=None):
    ## Plot up a time series of colname data from dataframe, plotting each fips provided in the list.

    ## Start by defaulting to a single figure and plotting it if no fig, ax values
    ## are provided
    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise ValueError('Must provide both "fig" and "ax" if you provide one of them')
    elif fig is None and ax is None:
        fig, ax = plt.subplots(1, 1)

    ## Check if FIPS input is reasonable
    if (type(fips) == int):
        fips = [fips]
    elif (type(fips) != list):
        raise ValueError('Input fips must be integer or list of integers')

    # Label the plot
    ax.tick_params(axis='x', rotation=30) # Rotate date labels
    xlabel = ax.set_xlabel("Date")
    ylabel = ax.set_ylabel(var_ylabel[colname])
    title = ax.set_title(var_descript[colname])

    # Loop through the FIPS values
    for FIPS in fips:
        # Get dataframe
        this_frame = COVID_IO.getLocalDataFrame(FIPS, dataframe)

        # Determine legend label to use
        if (FIPS > 100):
            labelstr = f"{this_frame['county'].values[0]} ({this_frame['state'].values[0]})"
        else:
            # This is a state
            labelstr = this_frame['state'].values[0]

        # retrieve the data (nan values are automatically excluded)
        dates = np.array(this_frame['dates'].to_list()[0])
        var = np.array(this_frame[colname].to_list()[0])

        # If the running average is set to a value greater than zero, compute and plot the
        # running average instead of raw data
        if (running_avg > 0):
            var = running_average(var, running_avg)
            labelstr = labelstr+f" [{running_avg:d} day running avg]"

        if (connectdots):
            ls='-'
        else:
            ls ='None'

        # Plot the data for this FIPS record
        p = ax.plot(dates, var, marker='o', markersize=3, linestyle=ls, label=labelstr)

    # Adjust y axis to be logarithmic if requested
    if (ylog):
        ax.set_yscale('log')

    # Adjust the x-axis limits to only cover last 90 days
    if (last90):
        xlims = ax.get_xlim()
        ax.set_xlim(xlims[1]-90,xlims[1])

    # Add legend
    legend = ax.legend(prop={'size': legendsize})


def ts_plot_Hos(dataframe, colname, fips, sum_dataframe=None, connectdots=False, ylog=False, last180=False, fig=None, ax=None):
    ## Plot up a time series of colname data from dataframe, plotting each fips provided in the list.

    # Started off as a copy of ts_plot that Juan wrote it just has different variables for y label & description
    # Then removed running averages and added error bars.

    ## Start by defaulting to a single figure and plotting it if no fig, ax values
    ## are provided
    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise ValueError('Must provide both "fig" and "ax" if you provide one of them')
    elif fig is None and ax is None:
        fig, ax = plt.subplots(1, 1)

    ## Check if FIPS input is reasonable
    if (type(fips) == int):
        fips = [fips]
    elif (type(fips) != list):
        raise ValueError('Input fips must be integer or list of integers')

    # Label the plot
    ax.tick_params(axis='x', rotation=30) # Rotate date labels
    xlabel = ax.set_xlabel("Date")
    ylabel = ax.set_ylabel(var_ylabel_Hosp[colname])
    title = ax.set_title(var_descript_Hosp[colname])

    # Loop through the FIPS values
    for FIPS in fips:
        # Get dataframe
        this_frame = COVID_IO.getLocalDataFrame(FIPS, dataframe)

        # Determine legend label to use (add bed numbers if appropriate and summary_df available)
        labelstr = this_frame['state'].values[0]
        if ((colname == 'allbed_mean')|(colname == 'ICUbed_mean')):
            if sum_dataframe is not None:
                this_summary_frame = COVID_IO.getLocalDataFrame(FIPS, sum_dataframe)
                if (colname == 'allbed_mean'):
                    labelstr = labelstr + f" ({int(this_summary_frame['all_bed_capacity'].to_list()[0])} total beds)"
                else: # ICU beds
                    labelstr = labelstr + f" ({int(this_summary_frame['icu_bed_capacity'].to_list()[0])} total ICU beds)"

        # retrieve the data (nan values are automatically excluded)
        dates = np.array(this_frame['dates'].to_list()[0])
        var = np.array(this_frame[colname].to_list()[0])

        if (connectdots):
            ls='-'
        else:
            ls ='None'

        # Plot the data for this FIPS record
        p = ax.plot(dates, var, marker='o', markersize=3, linestyle=ls, label=labelstr)

        # Track the color just used so error range matches
        colour = p[0].get_color()

        # Add error bars (in continuous form) if appropriate
        #print(f"Checking {colname} for ranges {var_ranges_Hosp[colname]}")
        if (var_ranges_Hosp[colname]):
            lower_colname = colname.replace("mean", "lower")
            upper_colname = colname.replace("mean", "upper")
            #print(f"Looking at {lower_colname} and {upper_colname}")
            var_lower = np.array(this_frame[lower_colname].to_list()[0])
            var_upper = np.array(this_frame[upper_colname].to_list()[0])
            #print(var_lower)
            prange = ax.fill_between(dates, var_lower, var_upper, color=colour, alpha=0.2)


    # Adjust y axis to be logarithmic if requested
    if (ylog):
        ax.set_yscale('log')

    # Adjust the x-axis limits to only cover last 180 days of model
    if (last180):
        xlims = ax.get_xlim()
        ax.set_xlim(xlims[1]-180,xlims[1])

    # Add legend
    legend = ax.legend(prop={'size': legendsize})


# def ts_plot_Rt(dataframe, fips, sum_dataframe=None, connectdots=False, ylog=False, last90=False, fig=None, ax=None):
#     ## Plot up a time series of Rt from dataframe, plotting each fips provided in the list.

#     ## Start by defaulting to a single figure and plotting it if no fig, ax values
#     ## are provided
#     if (fig is None and ax is not None) or (fig is not None and ax is None):
#         raise ValueError('Must provide both "fig" and "ax" if you provide one of them')
#     elif fig is None and ax is None:
#         fig, ax = plt.subplots(1, 1)

#     ## Check if FIPS input is reasonable
#     if (type(fips) == int):
#         fips = [fips]
#     elif (type(fips) != list):
#         raise ValueError('Input fips must be integer or list of integers')

#     # Label the plot
#     ax.tick_params(axis='x', rotation=30) # Rotate date labels
#     xlabel = ax.set_xlabel("Date")
#     ylabel = ax.set_ylabel("$R_t$")
#     title = ax.set_title("Effective Reproduction Rate")

#     # Loop through the FIPS values
#     for FIPS in fips:
#         # Get dataframe
#         this_frame = COVID_IO.getLocalDataFrame(FIPS, dataframe)

#         # Determine legend label to use (add bed numbers if appropriate and summary_df available)
#         labelstr = this_frame['state'].values[0]

#         # retrieve the data (nan values are automatically excluded)
#         dates = np.array(this_frame['dates'].to_list()[0])
#         var = np.array(this_frame['Rt_mean'].to_list()[0])

#         if (connectdots):
#             ls='-'
#         else:
#             ls ='None'

#         # Plot the data for this FIPS record
#         p = ax.plot(dates, var, marker='o', markersize=3, linestyle=ls, label=labelstr)

#         # Track the color just used so error range matches
#         colour = p[0].get_color()

#         # Add error bars (in continuous form)
#         lower_colname = "Rt_lower_80"
#         upper_colname = "Rt_upper_80"
#         var_lower = np.array(this_frame[lower_colname].to_list()[0])
#         var_upper = np.array(this_frame[upper_colname].to_list()[0])
#         prange = ax.fill_between(dates, var_lower, var_upper, color=colour, alpha=0.2)


#     # Adjust y axis to be logarithmic if requested
#     if (ylog):
#         ax.set_yscale('log')

#     # Adjust the x-axis limits to only cover last 90 days
#     if (last90):
#         xlims = ax.get_xlim()
#         ax.set_xlim(xlims[1]-90,xlims[1])

#     # Add legend
#     legend = ax.legend(prop={'size': legendsize})


def ts_barplot(dataframe, colname, fips, ylog=False, running_avg=0, fig=None, ax=None):
    ## Plot up bar graph of a time series of colname data from dataframe and a SINGLE fips

    ## Start by defaulting to a single figure and plotting it if no fig, ax values
    ## are provided
    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise ValueError('Must provide both "fig" and "ax" if you provide one of them')
    elif fig is None and ax is None:
        fig, ax = plt.subplots(1, 1)

    ## Check if FIPS input is reasonable
    if (type(fips) != int):
        raise ValueError('Input fips must be integer')

    # Get dataframe
    this_frame = COVID_IO.getLocalDataFrame(fips, dataframe)

    # Determine legend label to use
    if (fips > 100):
        labelstr = f"{this_frame['county'].values[0]} ({this_frame['state'].values[0]})"
    else:
        # This is a state
        labelstr = this_frame['state'].values[0]

    # Label the plot
    ax.tick_params(axis='x', rotation=30) # Rotate date labels
    xlabel = ax.set_xlabel("Date")
    ylabel = ax.set_ylabel(var_ylabel[colname])
    titlestr = var_descript[colname]+f" for {labelstr}"
    if (running_avg > 0):
        titlestr = titlestr + f" (with {running_avg} day running average)"
    title = ax.set_title(titlestr)

    # plot the bar graph of the data
    dates = np.array(this_frame['dates'].to_list()[0])
    var = np.array(this_frame[colname].to_list()[0])
    ax.bar(dates, var, width=0.5, color='lightcoral')

    # If the running average is set to a value greater than zero, ALSO plot the running
    # average in addition to the bar graph of the time series.
    if (running_avg > 0):
        var = running_average(var, running_avg)
        ax.plot(dates, var, marker='.', markersize=1, color='blue', linestyle='-', linewidth=2)

    # Adjust y axis to be logarithmic if requested
    if (ylog):
        ax.set_yscale('log')


##
## Functions for providing credit to data sources
##
def creditForApplMob(Display = True):
    html_out = f"<a href=\"https://www.apple.com/covid19/mobility\">Apple Mobility Data</a> provided by Apple Inc. (“Apple”) and its licensors."
    if(Display):
        display(HTML("<em style='font-size:8; line-height: 1em; text-align: justify;'>"+html_out+"</em>"))
    else:
        return html_out


def creditForIHME(Display = True):
    html_out = f"<a href=\"http://www.healthdata.org/covid/data-downloads\">Current and Projected Epidemiological/Resource data</a> provided by Institute for Health Metrics and Evaluation (IHME) at University of Washington, 2020."
    if(Display):
        display(HTML("<em style='font-size:8; line-height: 1em; text-align: justify;'>"+html_out+"</em>"))
    else:
        return html_out


def creditForGoogMob(Display = True):
    html_out = f"<a href=\"https://www.google.com/covid19/mobility\">Google Mobility Data</a> provided by Google LLC \"Google COVID-19 Community Mobility Reports\""
    if(Display):
        display(HTML("<em style='font-size:8; line-height: 1em; text-align: justify;'>"+html_out+"</em>"))
    else:
        return html_out


def creditForJH(Display = True):
    html_out = f"<a href=\"https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases\">Epidemiological data</a> provided by the <a href=\"https://github.com/CSSEGISandData/COVID-19\">COVID-19 Data Repository</a> of the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University.  Population data from <a href=\"https://www.census.gov/topics/population.html\">U.S. Census Bureau, Population Division</a> (Release Date: March 2020)"
    if(Display):
        display(HTML("<em style='font-size:8; line-height: 1em; text-align: justify;'>"+html_out+"</em>"))
    else:
        return html_out

def creditForRt(Display = True):
    html_out = f"R<sub>t</sub> model data provided by the <a href=\"http://rt.live/\">R<sub>t</sub> COVID-19</a> Model of Kevin Systrom and Thomas Vladeck."
    if(Display):
        display(HTML("<em style='font-size:8; line-height: 1em; text-align: justify;'>"+html_out+"</em>"))
    else:
        return html_out