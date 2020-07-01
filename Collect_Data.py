# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Collecting and Condensing COVID Data
#
# This Jupyter notebook reads in the data from a variety of online sources that we need for the COVID Data Vizualization Project.  A lot of effort is made to 'condense' the time series data into lists that can be retrieved much more quickly after the fact.  Also, the John Hopkins data went through a few changes in format, so some effort was made to put everything in a uniform format.
#
# Finally, all the data has been cross-identified with FIPS codes (a standard, although deprecated, geographic identifier) for the state or countys, which should made cross-matching data from multiple sources much easier later.

# %%
# This forces a reload of any external library file if it changes.  
# Useful when developing external libraries since otherwise Jupyter 
# will not re-import any library without restarting the python kernel.

# %load_ext autoreload
# %autoreload 2

# %%
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import git
import pickle
import warnings

# Import COVID data retrieval routines from external python library
import COVIDlib.collectors as COVIDdata

# %%
# Mark the start of processing
start = time.perf_counter()

## Define variables of interest below
data_dir = 'our_data/'    # Data directory for files we created

## Define FIPS corresponding to various local areas
ClayFIPS = 27027
CassFIPS = 38017
MNFIPS = 27
NDFIPS = 38

## Check if data directory exists, if not, create it
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# %% [markdown]
# ##  John Hopkins Cases Data (FIPS Present)
#     - https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases
#
# This dataset is part of COVID-19 Pandemic Novel Corona Virus (COVID-19)
# epidemiological data since 22 January 2020. The JHU CCSE maintains the data on the 2019 Novel
# Coronavirus COVID-19 (2019-nCoV) Data Repository on Github
# (https://github.com/CSSEGISandData/COVID-19). 
# I have also folded in US Census Bureau population informationf for these counties/states.
#
# ### Notes about the dataframes:
# - County level daily confirmed/deaths/recovered data files changes format
#   several times before April 23, 2020, so I didn't include that data. 
# - State level daily data files contain some additional data that county level
#   files do not contain, notably Incident_Rate, People_Tested,
#   People_Hospitalized, Mortality_Rate, Testing_Rate, and Hospitalization_Rate. 
#   However, it only exists starting April 12, 2020.
# - Time-series data files contain more limited data (only confirmed cases and
#   deaths) and are essentially redundant data compared to the daily files, so
#   combining the daily files makes sense.
#
# **Suggested Citations**: the COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University.  Population data from U.S. Census Bureau, Population Division (Release Date: March 2020)

# %%
## 
## Retrieve the US Census data and then John Hopkins data
##

# Retrieve needed Census Bureau data first
print("\n- Retrieving US Census and John Hopkins Data")
(cnty_pop_df, state_pop_df) = COVIDdata.retrieve_census_population_data()

# Retrieve John Hopkins data
(ts_us_confirmed_df, ts_us_dead_df, combined_cnty_df, combined_state_df) = COVIDdata.retrieve_John_Hopkins_data(cnty_pop_df, state_pop_df)

#
# Save the county and state-level processed daily dataframes into CSV files
#
#
# Save the same data to pickle files
#
combined_datafile = data_dir + "countylevel_combinedCDR.p"
print("   - JH county data exported to ", combined_datafile)
with open(combined_datafile, 'wb') as pickle_file:
    pickle.dump(combined_cnty_df, pickle_file)
    pickle_file.close()

combined_datafile = data_dir + "statelevel_combinedCDR.p"
print("   - JH state data exported to ", combined_datafile)
with open(combined_datafile, 'wb') as pickle_file:
    pickle.dump(combined_state_df, pickle_file)
    pickle_file.close()


# %%
# Convert datetime lists into strings
combined_cnty_df['Dates'] = combined_cnty_df['Dates'].apply(COVIDdata.dates2strings)
combined_state_df['Dates'] = combined_state_df['Dates'].apply(COVIDdata.dates2strings)

combined_datafile = data_dir + "countylevel_combinedCDR.csv"
print("   - JH county data also exported to ", combined_datafile)
combined_cnty_df.to_csv(combined_datafile, index=False)

combined_datafile = data_dir + "statelevel_combinedCDR.csv"
print("   - JH state level also exported to ", combined_datafile)
combined_state_df.to_csv(combined_datafile, index=False)

# %% [markdown]
# ## Google Mobility Data (FIPS cross-identification performed)
#
# This data is described at https://www.google.com/covid19/mobility/ and can be downloaded in a single monolithic CSV file at https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv
#
# > The data shows how visitors to (or time spent in) categorized places change compared to our baseline days. A baseline day represents a normal value for that day of the week. The baseline day is the median value from the 5‑week period Jan 3 – Feb 6, 2020.
#
# > For each region-category, the baseline isn’t a single value—it’s 7 individual values. The same number of visitors on 2 different days of the week, result in different percentage changes. So, we recommend the following:
# 1. Don’t infer that larger changes mean more visitors or smaller changes mean less visitors.
# 2. Avoid comparing day-to-day changes. Especially weekends with weekdays. (https://support.google.com/covid19-mobility/answer/9824897?hl=en&ref_topic=9822927)
#
# > Note, *Parks* typically means official national parks and not the general outdoors found in rural areas.
#
# Also, I'll note that aggregated national data appears to be available by setting `sub_region_1` **and** `sub_region_2` to `NaN` and state-level data by setting only `sub_region_2` to `NaN`.
#
# **Note to Developers:** Check the `date` column in the reduced data to see if it is a real match or just a marker for a non-match.  Furthermore be away Google has a lot of blank (`NaN`) entries in a lot of columns and variable numbers of entries for each county/state.
#
#
# **Suggested Citation**: Google LLC "Google COVID-19 Community Mobility Reports". https://www.google.com/covid19/mobility/ Accessed: `<Date>.`

# %%
# Retrieve the Google Mobility dataframes

print("\n- Retrieving Google Mobility Data")

# Issue the google retrieval function while suppressing the FutureWarning
# as detailed here (https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur)
# this is an annoying issue of competing open source teams disagreeing about 
# appropriate behavior.
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    (goog_mobility_cnty_df, goog_mobility_states_df) = COVIDdata.retrieve_goog_mobility_data(cnty_pop_df, state_pop_df)

#
# Save the same data to pickle files
#
print("   - Exporting Google mobility data")

goog_mobility_cnty_fname = data_dir + "goog_mobility_cnty.p"
print("   - Google county mobility data exported to ", goog_mobility_cnty_fname)
with open(goog_mobility_cnty_fname, 'wb') as pickle_file:
    pickle.dump(goog_mobility_cnty_df, pickle_file)
    pickle_file.close()

goog_mobility_states_fname = data_dir + "goog_mobility_state.p"
print("   - Google state mobility data exported to ", goog_mobility_states_fname)
with open(goog_mobility_states_fname, 'wb') as pickle_file:
    pickle.dump(goog_mobility_states_df, pickle_file)
    pickle_file.close()


# %%
# Convert datetime lists into strings
goog_mobility_cnty_df['dates'] = goog_mobility_cnty_df['dates'].apply(COVIDdata.dates2strings)
goog_mobility_states_df['dates'] = goog_mobility_states_df['dates'].apply(COVIDdata.dates2strings)

# Export the google mobility data to CSV files
goog_mobility_cnty_fname = data_dir + "goog_mobility_cnty.csv"
print("   - Google county mobility data also exported to ", goog_mobility_cnty_fname)
goog_mobility_cnty_df.to_csv(goog_mobility_cnty_fname, index=False)

goog_mobility_states_fname = data_dir + "goog_mobility_state.csv"
print("   - Google state mobility data also exported to ", goog_mobility_states_fname)
goog_mobility_states_df.to_csv(goog_mobility_states_fname, index=False)

# %% [markdown]
# ## Apple Mobility Data  (FIPS cross-identification performed)
#
# This data is described at https://www.apple.com/covid19/mobility.
#
# **About this Data (copied from Apple's site)**: The CSV file on this site show a relative volume of directions requests per country/region, sub-region or city compared to a baseline volume on January 13th, 2020. We define our day as midnight-to-midnight, Pacific time. Cities are defined as the greater metropolitan area and their geographic boundaries remain constant across the data set. In many countries/regions, sub-regions, and cities, relative volume has increased since January 13th, consistent with normal, seasonal usage of Apple Maps. Day of week effects are important to normalize as you use this data. Data that is sent from users’ devices to the Maps service is associated with random, rotating identifiers so Apple doesn’t have a profile of individual movements and searches. Apple Maps has no demographic information about our users, so we can’t make any statements about the representativeness of usage against the overall population.
#
# Apple tracks three kinds of Apple Maps routing requests: Driving, Walking, Transit.  But the only data available at the state and county level is the Driving data.
#
# **Developer Notes**: While Apple has fewer blank (`NaN`) entries when a county was included (versus Google data), Apple's mobility data only exists for 2090 out of 3142 counties in the US. Counties with no published data are in this dataframe as `NaN` for both dates and driving mobility information.
#

# %%
print("\n- Retrieving Apple Mobility Data")

# Retrieve the Apple Mobility dataframes
(aapl_mobility_cnty_df, aapl_mobility_states_df) = COVIDdata.retrieve_aapl_mobility_data(cnty_pop_df, state_pop_df)

## Notice only driving information is available at the county level here
#print("APPLE MOBILITY DATA IN aapl_mobility_cnty_df() FOR CLAY COUNTY")
#aapl_mobility_clay = aapl_mobility_cnty_df[(aapl_mobility_cnty_df['county'] == 'Clay County') & (aapl_mobility_cnty_df['state'] == 'Minnesota')]
#print(aapl_mobility_clay)
    
# Export the Apple mobility data to CSV files
print("   - Exporting Apple mobility data")
    

#
# Save the same data to pickle files
#
aapl_mobility_cnty_fname = data_dir + "aapl_mobility_cnty.p"
print("   - Apple county mobility data exported to ", aapl_mobility_cnty_fname)
with open(aapl_mobility_cnty_fname, 'wb') as pickle_file:
    pickle.dump(aapl_mobility_cnty_df, pickle_file)
    pickle_file.close()

aapl_mobility_states_fname = data_dir + "aapl_mobility_state.p"
print("   - Apple state mobility data exported to ", aapl_mobility_states_fname)
with open(aapl_mobility_states_fname, 'wb') as pickle_file:
    pickle.dump(aapl_mobility_states_df, pickle_file)
    pickle_file.close()


# %%
# Convert datetime lists into strings
aapl_mobility_cnty_df['dates'] = aapl_mobility_cnty_df['dates'].apply(COVIDdata.dates2strings)
aapl_mobility_states_df['dates'] = aapl_mobility_states_df['dates'].apply(COVIDdata.dates2strings)
    
aapl_mobility_cnty_fname = data_dir + "aapl_mobility_cnty.csv"
print("   - Apple county mobility data also exported to ", aapl_mobility_cnty_fname)
aapl_mobility_cnty_df.to_csv(aapl_mobility_cnty_fname, index=False)

aapl_mobility_states_fname = data_dir + "aapl_mobility_state.csv"
print("   - Apple state mobility data also exported to ", aapl_mobility_states_fname)
aapl_mobility_states_df.to_csv(aapl_mobility_states_fname, index=False)

# %% [markdown]
# ## Institute for Health Metrics and Evaluation (IMHE) Data on Local Resources  (FIPS cross-identification performed)
#
# There is Institute for Health Metrics and Evaluation data on local resources at http://www.healthdata.org/covid/data-downloads although data only has state level resolution. 
#
# **Suggested Citation**: Institute for Health Metrics and Evaluation (IHME). COVID-19 Hospital Needs and Death Projections. Seattle, United States of America: Institute for Health Metrics and Evaluation (IHME), University of Washington, 2020.
#
# **Note to Developers:** IMHE data has a few blank (`NaN`) entries for dates, presumably reflecting unknown values.  Also, some of the dates are from 2019, which suggests no known values.

# %%
# Retrieve IMHE data
print("\n- Retrieving IMHE Data")
(imhe_summary, imhe_hospitalizations) = COVIDdata.retrieve_imhe_data(cnty_pop_df, state_pop_df)

## Summary data includes numbers or dates for the following for each state
#             'peak_bed_day_mean', 'peak_bed_day_lower', 'peak_bed_day_upper': Mean/Lower/Upper Uncertainty peak bed use date
# 'peak_icu_bed_day_mean', 'peak_icu_bed_day_lower', 'peak_icu_bed_day_upper': Mean/Lower/Upper Uncertainty ICU bed use date
#          'peak_vent_day_mean', 'peak_vent_day_lower', 'peak_vent_day_upper': Mean/Lower/Upper Uncertainty Ventilator use date
#    'all_bed_capacity', 'icu_bed_capacity', 'all_bed_usage', 'icu_bed_usage': Number of beds/ICU beds/avg beds used/avg ICU beds used
#                          'travel_limit_start_date', 'travel_limit_end_date': Severe travel restrictions start/end dates
#                                'stay_home_start_date', 'stay_home_end_date': Stay at home order start/end dates
#                    'educational_fac_start_date', 'educational_fac_end_date': Educational facilities closure start/end dates
#      'any_gathering_restrict_start_date', 'any_gathering_restrict_end_date': Any gathering restrictions start/end dates
#                          'any_business_start_date', 'any_business_end_date': Any business closures start/end dates
#          'all_non-ess_business_start_date', 'all_non-ess_business_end_date': Non-essential businesses ordered to close start/end dates
#
# 'NaN' present for dates means it isn't known.

## Hospitalization data is time series date for the following projections by the IMHE:
#                             'allbed_mean', 'allbed_lower','allbed_upper': Predicted COVID beds needed with upper/lower bounds
#                            'ICUbed_mean', 'ICUbed_lower', 'ICUbed_upper': Predicted COVID ICU beds needed with upper/lower bounds
#                            'InvVen_mean', 'InvVen_lower', 'InvVen_upper': Predicted COVID ventilators needed with upper/lower bounds
#                            'deaths_mean', 'deaths_lower', 'deaths_upper': Predicted COVID daily deaths with upper/lower bounds
#                               'admis_mean', 'admis_lower', 'admis_upper': Predicted hospital admissions with upper/lower bounds
#                            'newICU_mean', 'newICU_lower', 'newICU_upper': Predicted new ICU admissions per day with upper/lower bounds
#                            'totdea_mean', 'totdea_lower', 'totdea_upper': Predicted COVID cumilative deaths with upper/lower bounds
# 'deaths_mean_smoothed', 'deaths_lower_smoothed', 'deaths_upper_smoothed': Smoothed version of predicted COVID daily deaths
# 'totdea_mean_smoothed', 'totdea_lower_smoothed', 'totdea_upper_smoothed': Smoothed version of cumilative COVID deaths
#                                   'total_tests_data_type', 'total_tests': observed/predicted tests and total number of tests
#                                                   'confirmed_infections': Observed confirmed infections only
#    'est_infections_mean', 'est_infections_lower', 'est_infections_upper': Predicted estimated infections with upper/lower bounds
#
# 'NaN' present for dates means it isn't known.

#
# Save the same data to pickle files
#
imhe_summary_fname = data_dir + "imhe_summary.p"
print("   - IMHE summary data exported to ", imhe_summary_fname)
with open(imhe_summary_fname, 'wb') as pickle_file:
    pickle.dump(imhe_summary, pickle_file)
    pickle_file.close()

imhe_hospitalizations_fname = data_dir + "imhe_hospitalizations.p"
print("   - IMHE hospitalization data exported to ", imhe_summary_fname)
with open(imhe_hospitalizations_fname, 'wb') as pickle_file:
    pickle.dump(imhe_hospitalizations, pickle_file)
    pickle_file.close()

# %%
# Convert datetime lists into strings
imhe_hospitalizations['dates'] = imhe_hospitalizations['dates'].apply(COVIDdata.dates2strings)

# Write out CSV files to disk
imhe_summary_fname = data_dir + "imhe_summary.csv"
print("   - IMHE summary data also exported to ", imhe_summary_fname)
imhe_summary.to_csv(imhe_summary_fname, index=False)

imhe_hospitalizations_fname = data_dir + "imhe_hospitalizations.csv"
print("   - IMHE hospitalization data also exported to ", imhe_summary_fname)
imhe_hospitalizations.to_csv(imhe_hospitalizations_fname, index=False)


# %%
# Present summary data for local area
#print("\nIMHE SUMMARY DATA IN imhe_summary() FOR MN and ND")
#imhe_summary_local = imhe_summary[(imhe_summary.FIPS == MNFIPS) | (imhe_summary.FIPS == NDFIPS) ]
#print(imhe_summary_local)

# Present hospitalizations data for local area
#print("\nIMHE SUMMARY DATA IN imhe_hospitalizations() FOR MN and ND")
#imhe_hospitalizations_local = imhe_hospitalizations[(imhe_hospitalizations.FIPS == MNFIPS) | (imhe_hospitalizations.FIPS == NDFIPS) ]
#print(imhe_hospitalizations_local)

# %% [markdown]
# # Estimated Effective Reproduction Rate $R_t$
#
# Kevin Systrom and Mike Krieger (co-founders of Instagram) and Tom Vladeck (owner of Gradient Metrics) put together a unaffiliated project to tracked modelled $R_t$ values for each state at http://rt.live/
#
# "$R_t$ represents the effective reproduction rate of the virus calculated for each locale. It lets us estimate how many secondary infections are likely to occur from a single infection in a specific area. Values over 1.0 mean we should expect more cases in that area, values under 1.0 mean we should expect fewer."

# %%
# Retrieve the Rt Live data

print("\n- Retrieving Rt Live Effective Reproduction Rate Data")

# Retrieve the Rt live dataframe
Rt_live_df = COVIDdata.retrieve_Rt_live_data(state_pop_df)

# %%
# Export the Rt live data
print("   - Exporting Rt live data")
    
#
# Save the data to pickle and CSV files
#

Rt_live_fname = data_dir + "Rt_live.p"
print("   - Rt live data exported to ", Rt_live_fname)
with open(Rt_live_fname, 'wb') as pickle_file:
    pickle.dump(Rt_live_df, pickle_file)
    pickle_file.close()

# Convert datetime lists into strings
Rt_live_df['dates'] = Rt_live_df['dates'].apply(COVIDdata.dates2strings)

# Write out CSV files to disk
Rt_live_fname = data_dir + "Rt_live.csv"
print("   - Rt live dataalso exported to ", Rt_live_fname)
Rt_live_df.to_csv(Rt_live_fname, index=False)

# %% [markdown]
# ## NY Times Data on Probable Deaths/Cases (FIPS Present, NOT CURRENTLY USED FOR PROJECT)
#
# The NY Times has assembled data on COVID in a GitHub repository at https://github.com/nytimes/covid-19-data.  I have not examined that data yet, but it may well be interesting.
#
# Note their statement requiring credit:
#
# > In light of the current public health emergency, The New York Times Company is
# providing this database under the following free-of-cost, perpetual,
# non-exclusive license. Anyone may copy, distribute, and display the database, or
# any part thereof, and make derivative works based on it, provided  (a) any such
# use is for non-commercial purposes only and (b) credit is given to The New York
# Times in any public display of the database, in any publication derived in part
# or in full from the database, and in any other public use of the data contained
# in or derived from the database.
#
# Data is available at county, state, and national levels for live numbers (current cases/deaths as well as probable cases/deaths, updated daily).  That said, at least locally I don't think Probable cases are really making a difference.
#

# %%
# ##
# ## Retrieve the NYT datafiles to see what is there that might be of interest
# ##
# print("\n- Retrieving NYT COVID Data")

# # NYT COVID data GIT repo
# NYT_gitrepo = "https://github.com/nytimes/covid-19-data.git"

# # Local repo location for the NYT Datafiles
# NYTdata_dir = "NYT_Data/"

# # Check if the local git repository directory exists, if not, create it and clone the repo to it
# if not os.path.exists(NYTdata_dir):
#     os.makedirs(NYTdata_dir)
#     git.Repo.clone_from(NYT_gitrepo, NYTdata_dir)
    
# g = git.cmd.Git(NYTdata_dir)
# # We should check status to see everything is good eventually, 
# # for now, I am using this to hide the status message from GitPython module
# status = g.pull()  

# # Grab the live data files
# live_county_csv = NYTdata_dir+"live/us-counties.csv"
# live_state_csv = NYTdata_dir+"live/us-states.csv"
# live_us_csv = NYTdata_dir+"live/us.csv"

# # Create pandas dataframes containing the daily data from the CSV files (contains number of confirmed/deaths/recovered on that date)
# live_county_df = pd.read_csv(live_county_csv)   # County totals
# live_state_df = pd.read_csv(live_state_csv)    # State totals
# live_us_df = pd.read_csv(live_us_csv)       # National totals

# # Print county data to screen
# #print("LOCAL COUNTY DATA IN live_county_df() DATAFRAME")
# #print(live_county_df[ (live_county_df['fips'] == ClayFIPS) | (live_county_df['fips'] == CassFIPS) ])

# # Print state level data to screen
# #print("\nLOCAL STATE DATA IN live_state_df() DATAFRAME")
# #print(live_state_df[ (live_state_df['fips'] == MNFIPS) | (live_state_df['fips'] == NDFIPS) ])

# # Print national data
# #print("\nNATIONAL DATA IN live_us_df() DATAFRAME")
# #print(live_us_df)

# %%
# Mark the start of processing
end = time.perf_counter()

print(f"\n\nEntire process of executing this script took {end-start:0.2f} sec.\n")

# %%
