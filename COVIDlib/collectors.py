##
## This is a set of functions for collecting the COVID data from the internet.
##

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import git
import requests
from io import BytesIO
from zipfile import ZipFile
from datetime import date, timedelta, datetime

##
## Define various variables and functions we will use below
##

# Define the postal codes for states
code2name = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'DC': 'District of Columbia',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming'
}

def csvfiles(path):
    contents = os.listdir(path);
    csvs = [ file for file in contents if file.endswith(".csv") ]

    for file in sorted(csvs, key=date2num):
        yield file


def isodate2num(s):
    # Takes input filename of CSV file and returns it sorted by date implies assuming
    # filenames with MM-DD-YYYY.csv format, returning YYYY+MM+DD for sorting purposes.
    M = int(s[0:2])
    D = int(s[3:5])
    Y = int(s[6:10])
    return (f"{Y:04d}-{M:02d}-{D:02d}")


def date2num(s):
    # Takes input filename of CSV file and returns it sorted by date implies assuming
    # filenames with MM-DD-YYYY.csv format, returning YYYY+MM+DD for sorting purposes.
    M = int(s[0:2])
    D = int(s[3:5])
    Y = int(s[6:10])
    return (f"{Y:04d}{M:02d}{D:02d}")


def iso2days(iso):
    # To make computing the derivative easy, compute days since January 1, 2020
    ref = datetime.fromisoformat("2020-01-01")
    return (datetime.fromisoformat(iso) - ref)/timedelta(days=1)


def days_since(date):
    # To make computing the derivative easy, compute days since January 1, 2020
    ref = datetime.fromisoformat("2020-01-01").date()
    return (date - ref)/timedelta(days=1)


def dates2strings(this_list):
    # Converts a list of datetime.date objects into a list of string dates
    try:
        return [day.strftime('%Y-%m-%d') for day in this_list]
    except:
        return this_list


def datetime_range(start=None, end=None):
    # Returns a list of dates between two dates INCLUSIVE
    # Grabbed this from https://stackoverflow.com/questions/7274267/print-all-day-dates-between-two-dates/7274316
    # and modified it for my needs
    span = end - start
    for i in range(span.days + 1):
        yield start + timedelta(days=i)


def derivative(x, y):
    """
    Compute forward difference estimate for the derivative of y with respect
    to x.  The x and y arrays are 2-D with the rows being different data sets
    and the columns being the x/y values in each data set.

    The input and must be the same size.

    Note that we set the first known derivative values into the zeroth column to NaN.
    """
    # Compute the numerator (y[i+1] - y[i]) for all rows in the entire array at once
    dy = y[:, 1:] - y[:, 0:-1]
    # Compute the denominator (x[i+1] - x[i]) for all rows in the for entire array at once
    dx = x[:, 1:] - x[:, 0:-1]
    # Compute the derivatives for all points in the array at once
    dydx = dy / dx
    # Get first column to horizontal stack with numpy array
    first_col = dydx[:,0][..., None] # The [..., None] bit keeps (n, 1) shape (a la https://stackoverflow.com/questions/15815854/how-to-add-column-to-numpy-array)
    dydx = np.hstack((first_col, dydx))
    dydx[:,0] = np.NaN
    return dydx


def derivative_ndays(x, y, ndays):
    """
    Compute forward difference estimate via the same method above, but use a ndays-day baseline 
    for computing the derivative at any point relative to a point ndays before.

    Note that we set the first ndays of these derivative values to NaN.
    """
    # Compute the numerator (y[i+ndays] - y[i]) for all rows in the entire array at once
    dy = y[:, ndays:] - y[:, 0:-ndays]
    # Compute the denominator (x[i+ndays] - x[i]) for all rows in the for entire array at once
    dx = x[:, ndays:] - x[:, 0:-ndays]
    # Compute the derivatives for all points in the array at once
    dydx = dy / dx
    # Get first 7 columns to horizontal stack with numpy array and set them to NaN
    first_cols = dydx[:,0:ndays]
    dydx = np.hstack((first_cols, dydx))
    dydx[:,0:ndays] = np.NaN
    return dydx


def reduce_local_dataframe(raw_df, fips_df):
    # Reduce the raw dataframe of all COVID data to just the relevant entries for the US
    # Match to the list of known FIPS values from Census data

    # Perform a left join to the FIPS list
    reduced_df = fips_df.join(raw_df.set_index('FIPS'), on='FIPS', how='left').copy()

    # Here's the fund part, turns out the John Hopkins data has duplicate lines for a few FIPS
    # entries, so what I did was assume the HIGHER values of COVID confirmed, deaths, and recovered
    # were the accurate ones and just kept those.  This meant checking all the duplicates and then
    # purging them, re-writing the best values into the row of data.

    update_confirmed = {}
    update_deaths = {}
    update_recovered = {}
    # Identify duplicate FIPS entries (assumed mangled data)
    duplicates = reduced_df[reduced_df.duplicated(['FIPS'], keep='first')]
    # Loop through all data for each duplicate FIPS entry
    for index, row in duplicates.iterrows():
        max_confirmed = 0
        max_deaths = 0
        max_recovered = 0
        checkFIPS = row['FIPS']
        # Track the maximum value of confirmed/deaths/recovered for each FIPS
        for dup_ind, dup_row in reduced_df[reduced_df['FIPS'] == row['FIPS']].iterrows():
            if max_confirmed<dup_row['Confirmed']:
                max_confirmed = dup_row['Confirmed']
            if max_deaths<dup_row['Deaths']:
                max_deaths = dup_row['Deaths']
            if max_recovered<dup_row['Recovered']:
                max_recovered = dup_row['Recovered']
        update_confirmed.update( {checkFIPS : max_confirmed} )
        update_deaths.update( {checkFIPS : max_deaths} )
        update_recovered.update( {checkFIPS : max_recovered} )

    # Drop duplicates from pandas dataframe
    reduced_df.drop_duplicates(subset='FIPS', keep='first', inplace=True)

    # Fix values in duplicate lines
    for key in update_confirmed:
        reduced_df.loc[(reduced_df['FIPS'] == key), ['Confirmed', 'Deaths', 'Recovered']] = [update_confirmed[key], update_deaths[key], update_recovered[key]]

    # Return the fixed dataframe
    return reduced_df


def retrieve_census_population_data():
    ###
    ### US Census Data on Populations of States/Counties (FIPS Present)
    ###
    #
    # This data from the US Census Bureau estimates the population in July 2019.  Description of the file format is at https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2019/co-est2019-alldata.pdf
    #
    # - **County Level Data**: https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv
    # - **State Level Data**: https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv
    #
    # **Suggested Citation**:  U.S. Census Bureau, Population Division (Release Date: March 2020)

    # Create pandas dataframes containing the selected population data for each state/county
    census_county_csv = "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv"
    county_columns_of_interest = {'STATE', 'COUNTY', 'STNAME', 'CTYNAME', 'NPOPCHG_2019', 'POPESTIMATE2019'}
    census_county_df = pd.read_csv(census_county_csv,usecols=county_columns_of_interest, encoding='latin-1')

    # Separate state level data from county level data (by creating separate copies in memory)
    county_data_df = census_county_df[census_county_df['COUNTY'] != 0].copy()
    state_data_df = census_county_df[census_county_df['COUNTY'] == 0].copy()

    ##
    ## Manipulate the state-level population data (actually grabbed from county file, since its there anyway)
    ##

    # Add FIPS column for state data then DROP county data and move FIPS to first column before exporting
    state_data_df['FIPS'] = state_data_df['STATE']
    state_data_df.drop(columns=['STATE','COUNTY','CTYNAME'], inplace=True)
    state_data_df = state_data_df.reindex(columns=(['FIPS'] + list([col for col in state_data_df.columns if col != 'FIPS']) ))

    # Compute percent change in population in 2018-19
    state_data_df['PPOPCHG_2019'] = 100*(state_data_df['NPOPCHG_2019']/state_data_df['POPESTIMATE2019'])

    ##
    ## Manipulate the county-level population data and save a reduced datafile
    ##

    # In county data create FIPS column, remove redundant columns, and then move FIPS columns to first column
    county_data_df['FIPS'] = county_data_df['STATE']*1000 + county_data_df['COUNTY']
    county_data_df.drop(columns=['STATE','COUNTY'], inplace=True)
    county_data_df = county_data_df.reindex(columns=(['FIPS'] + list([col for col in county_data_df.columns if col != 'FIPS']) ))

    # Compute percent change in population in 2018-19
    county_data_df['PPOPCHG_2019'] = 100*(county_data_df['NPOPCHG_2019']/county_data_df['POPESTIMATE2019'])

    # We may want to do a daily extrapolation of population since POPESTIMATE2019 is est. population on July 1, 2019
    # and NPOPCHG_2019 is the estimated change between July 1, 2018 and July 1, 2019.  Realistically, this is probably
    # overkill since the increased deaths from Coronavirus are not taken into account in such an extrapolation.
    return(county_data_df, state_data_df)


def retrieve_John_Hopkins_data(county_data_df, state_data_df, JHdata_dir = "JH_Data/"):
    ###
    ###  Novel Coronavirus (COVID-19) Cases Data (FIPS Present)
    ###
    # - https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases
    #
    # This dataset is part of COVID-19 Pandemic Novel Corona Virus (COVID-19)
    # epidemiological data since 22 January 2020. The data is compiled by the Johns
    # Hopkins University Center for Systems Science and Engineering (JHU CCSE) from
    # various sources including the World Health Organization (WHO), DXY.cn, BNO
    # News, National Health Commission of the People’s Republic of China (NHC),
    # China CDC (CCDC),Hong Kong Department of Health, Macau Government, Taiwan
    # CDC, US CDC, Government of Canada, Australia Government Department of Health,
    # European Centre for Disease Prevention and Control (ECDC), Ministry of Health
    # Singapore (MOH), and others. JHU CCSE maintains the data on the 2019 Novel
    # Coronavirus COVID-19 (2019-nCoV) Data Repository on Github
    # (https://github.com/CSSEGISandData/COVID-19).
    #
    # ### Notes about the data files:
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
    # ### Notes about this process:
    # - By processing the US Census Bureau's population data first, I can get a list
    #   of legitimate 'FIPS' values to use, so that I can perform 'left' joins to a
    #   list of FIPS addresses instead of costly 'outer' joins.
    # - It turns out there are DUPLICATE FIPS entries for some of the John Hopkins
    #   data.  So I went through those and picked out the ones with the higher values
    #   for confirmed/deaths/recovered assuming they were entered later.
    # - Once that was done, I constructed a SINGLE Pandas dataframe holding all the
    #   time-series data and wrote it to a CSV file. Accessing this one file will be a
    #   lot faster than looping through all the datafiles each time we load up the
    #   data.
    #
    # **Suggested Citation**: the COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University.

    # Define local variables
    ClayFIPS = 27027
    MNFIPS = 27

    # Check if the local git repository directory exists, if not, create it and clone the repo to it
    JH_repo = "https://github.com/CSSEGISandData/COVID-19.git"
    if not os.path.exists(JHdata_dir):
        os.makedirs(JHdata_dir)
        git.Repo.clone_from(JH_repo, JHdata_dir)


    # Git pull to sync up the data set to the current version on GitHub
    g = git.cmd.Git(JHdata_dir)
    status = g.pull()  # We should check status to see everything is good eventually, for now, I am using this to hide the status message from GitPython module

    # Daily tabulation of all confirmed/deaths/recovered data is in the following directories
    daily_cnty_dir = JHdata_dir+"csse_covid_19_data/csse_covid_19_daily_reports/" # For each admin unit (in the US, that's county) for each day.
    daily_state_dir = JHdata_dir+"csse_covid_19_data/csse_covid_19_daily_reports_us/" # For each state (somewhat redundant, but avoids recomputation I suppose)

    # Individual time series data for confirmed cases and deaths in the US counties and states
    ts_us_confirmed_csv = JHdata_dir+"csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
    ts_us_dead_csv = JHdata_dir+"csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"

    ##
    ## Load the time series datafiles to experiment with them.  These only contain Deaths and Confirmed cases,
    ## so I suspect we won't keep them, since I build the same data from the daily files above.
    ##

    # Create pandas dataframes containing time-series data (We could reconstruct this by looping through all the daily data, since this is missing number of recovered)
    ts_us_dead_df = pd.read_csv(ts_us_dead_csv)            # Deaths in time series
    ts_us_confirmed_df = pd.read_csv(ts_us_confirmed_csv)  # Confirmed in time series

    ##
    ## Now process all the daily datafiles provided by John Hopkins, these have more through data
    ## and make the time-series files somewhat redundant.
    ##

    # Grab complete list of all csvs to then...
    world_csvs = list(csvfiles(daily_cnty_dir))
    us_csvs = list(csvfiles(daily_state_dir))

    # .. grab the most recent CSV file to open the data.
    daily_world_csv = daily_cnty_dir+world_csvs[-1]
    daily_us_csv = daily_state_dir+us_csvs[-1]

    # Create pandas dataframes containing the daily data from the CSV files
    # (contains number of confirmed/deaths/recovered on that date)
    daily_world_df = pd.read_csv(daily_world_csv)   # County/Admin totals
    daily_us_df = pd.read_csv(daily_us_csv)         # State totals

    ##
    ## Build combined county-level datafiles
    ##

    # Build a dataframe containing legitimate FIPS values using county level data
    fips_df = county_data_df.copy()
    fips_df.drop(columns=['STNAME', 'CTYNAME', 'POPESTIMATE2019', 'NPOPCHG_2019', 'PPOPCHG_2019'], inplace=True)

    # Build a dataframe containing legitimate FIPS values using state-level data (doing this now because I will try
    # to extract early state-level data from the county-level files)
    state_fips_df = state_data_df.copy()
    state_fips_df.drop(columns=['STNAME', 'POPESTIMATE2019', 'NPOPCHG_2019', 'PPOPCHG_2019'], inplace=True)

    # Build a state name to FIPS dictionary
    # Start by converting states dataframe into a dictionary
    FIPSdf = state_data_df[['FIPS','STNAME']].copy()
    FIPSd = FIPSdf.set_index('STNAME').T.to_dict('records')[0]

    # Create blank dataframes to store state-level time series data for later (most of this will be NaN early on)
    state_confirmed_df = state_fips_df.copy()
    state_deaths_df = state_fips_df.copy()
    state_recovered_df = state_fips_df.copy()
    state_incident_rate_df = state_fips_df.copy()
    state_tested_df = state_fips_df.copy()
    state_hospitalized_df = state_fips_df.copy()
    state_mortality_df = state_fips_df.copy()
    state_testing_rate_df = state_fips_df.copy()
    state_hospitalization_rate_df = state_fips_df.copy()

    # Scan through the more complete daily files of county level data and construct a single datafile for our use (restricting
    # to US only).  It turns out the format of these local level files changes with the date.  The files before March 22, 2020
    # were in several different forms and the placenames were NOT consistent.  Thus, to make things managable, I am ignoring
    # that early local level data...
    sorted_csvs = []
    dates_list = []
    state_sorted_csvs = []
    state_dates_list = []
    for file in csvfiles(daily_cnty_dir):
        # County level date only starts on March 22, before then it is a mis-mosh of place names in the Province_State field
        # So only keep that data
        this_isodate = isodate2num(file)
        this_date = date2num(file)
        if (int(this_date) >= 20200322):
            # Append to list of accessed csv files
            sorted_csvs.append(daily_cnty_dir+file)

            # Grab the data from the CSV file
            raw_df = pd.read_csv(sorted_csvs[-1])

            # Rename columns in early forms to late forms of column names for consistency
            raw_df.rename(columns={ 'Province/State': 'Province_State',
                                'Country/Region':'Country_Region',
                                'Last Update':'Last_Update' }, inplace = True)

            # Match to the list of known FIPS values from Census data, also removing duplicate rows
            reduced_df = reduce_local_dataframe(raw_df, fips_df)

            # Provide progress report
            idx = len(sorted_csvs)
            # print(f'Processing Date #{idx}: {this_isodate}')

            if (idx == 1):
                # Create combined dataframe sorted by FIPS
                combined_cols = ['FIPS', 'Admin2', 'Province_State', 'Lat', 'Long_']
                combined_cnty_df = reduced_df[combined_cols].copy()

                # Create blank dataframes to store time series data
                confirmed_df = fips_df.copy()
                deaths_df = fips_df.copy()
                recovered_df = fips_df.copy()

            ## Create dataframes for temporarily storing time series date
            # Append date to list of dates
            # dates_list.append(this_isodate)
            dates_list.append(datetime.strptime(this_isodate, '%Y-%m-%d').date())

            # Store Confirmed by merging reduced list and renaming column
            this_col = "X"+f"{idx:03d}"
            confirmed_df = pd.merge(confirmed_df,reduced_df[['FIPS','Confirmed']],on='FIPS', how='left', copy=True)
            confirmed_df.rename(columns={'Confirmed': this_col}, errors="raise", inplace=True)

            # Store Deaths by merging reduced list and renaming column
            deaths_df = pd.merge(deaths_df,reduced_df[['FIPS','Deaths']],on='FIPS', how='left', copy=True)
            deaths_df.rename(columns={'Deaths': this_col}, errors="raise", inplace=True)

            # Store Recovered by merging reduced list and renaming column
            recovered_df = pd.merge(recovered_df,reduced_df[['FIPS','Recovered']],on='FIPS', how='left', copy=True)
            recovered_df.rename(columns={'Recovered': this_col}, errors="raise", inplace=True)

            # Collect state-level counts for dates before April 12 (Since this if statement runs after
            # the else statement below, it should append to those dataframes)
            if (int(this_date) < 20200412):
                # Increment state date count
                state_idx += 1
                state_dates_list.append(datetime.strptime(this_isodate, '%Y-%m-%d').date())

                # Add the columns to the storage dataframes
                this_col = "X"+f"{state_idx:03d}"
                # print (f"Assigning {this_col}")
                state_confirmed_df[this_col] = 0
                state_deaths_df[this_col] =0
                state_recovered_df[this_col] = 0
                state_incident_rate_df[this_col] = np.nan
                state_tested_df[this_col] = np.nan
                state_hospitalized_df[this_col] = np.nan
                state_mortality_df[this_col] = np.nan
                state_testing_rate_df[this_col] = np.nan
                state_hospitalization_rate_df[this_col] = np.nan

                # Loop through all FIPS codes
                for key in FIPSd:
                    # Query for this data
                    matchstr = f"Province_State == '{key}'"
                    confirmed = reduced_df.query(matchstr)['Confirmed'].sum()
                    deaths = reduced_df.query(matchstr)['Deaths'].sum()
                    recovered = reduced_df.query(matchstr)['Recovered'].sum()

                    # Store in the dataframe
                    pd.set_option('mode.chained_assignment', None) # Turn off warning, since NOT an issue here
                    ind = state_confirmed_df.FIPS[state_confirmed_df['FIPS'] == FIPSd[key]].index
                    state_confirmed_df[this_col][ind] = confirmed
                    ind = state_deaths_df.FIPS[state_deaths_df['FIPS'] == FIPSd[key]].index
                    state_deaths_df[this_col][ind] = deaths
                    ind = state_recovered_df.FIPS[state_recovered_df['FIPS'] == FIPSd[key]].index
                    state_recovered_df[this_col][ind] = recovered
                    pd.set_option('mode.chained_assignment', 'warn') # Turn warning back on

        else:  # This is early data and can be processed for later use in state-level datafiles
            # Append to list of accessed csv files
            state_sorted_csvs.append(daily_cnty_dir+file)

            # Grab the data from the CSV file
            raw_df = pd.read_csv(state_sorted_csvs[-1])

            # Rename columns in early forms to late forms of column names for consistency
            raw_df.rename(columns={ 'Province/State': 'Province_State',
                                'Country/Region':'Country_Region',
                                'Last Update':'Last_Update' }, inplace = True)

            state_dates_list.append(datetime.strptime(this_isodate, '%Y-%m-%d').date())

            state_idx = len(state_sorted_csvs)

            # Pull out only the US data and present here for now
            us_df = raw_df[raw_df['Country_Region'] == 'US']
            # print(this_isodate)

            # Add the columns to the storage dataframes
            this_col = "X"+f"{state_idx:03d}"
            # print (f"Assigning {this_col}")
            state_confirmed_df[this_col] = 0
            state_deaths_df[this_col] =0
            state_recovered_df[this_col] = 0
            state_incident_rate_df[this_col] = np.nan
            state_tested_df[this_col] = np.nan
            state_hospitalized_df[this_col] = np.nan
            state_mortality_df[this_col] = np.nan
            state_testing_rate_df[this_col] = np.nan
            state_hospitalization_rate_df[this_col] = np.nan

            # Before March 10, postal codes used and "(from Diamond Princess)" labels the
            # Province_State (which are also separating counties).  Remove labels before
            # postal codes and remove all (from Diamong Princess) cases and then sum by state.
            # Use this to build initial dataframes.
            if (int(this_date) < 20200310):
                # Process the Province_State entry to be proper state names (might be duplicates)
                us_df = us_df[~us_df.Province_State.str.contains("Princess")] # Drop all entries with Diamond Princess
                # Reduce all Province_State entries to postal codes and then replace with full state names
                us_df['Province_State'] = us_df['Province_State'].str.replace('.*, ','')
                us_df['Province_State'] = us_df['Province_State'].map(code2name).fillna(us_df['Province_State'])
                # NOTE: There may be multiple rows with the same name which is why I need to SUM below

            # Loop through all FIPS codes
            for key in FIPSd:
                # Query for this data
                matchstr = f"Province_State == '{key}'"
                confirmed = us_df.query(matchstr)['Confirmed'].sum()
                deaths = us_df.query(matchstr)['Deaths'].sum()
                recovered = us_df.query(matchstr)['Recovered'].sum()

                # Store in the dataframe
                pd.set_option('mode.chained_assignment', None) # Turn off warning, since NOT an issue here
                ind = state_confirmed_df.FIPS[state_confirmed_df['FIPS'] == FIPSd[key]].index
                state_confirmed_df[this_col][ind] = confirmed
                ind = state_deaths_df.FIPS[state_deaths_df['FIPS'] == FIPSd[key]].index
                state_deaths_df[this_col][ind] = deaths
                ind = state_recovered_df.FIPS[state_recovered_df['FIPS'] == FIPSd[key]].index
                state_recovered_df[this_col][ind] = recovered
                pd.set_option('mode.chained_assignment', 'warn') # Turn warning back on

            # NOTE: Still need to sum up county information for March 22 through April 12, but that comes from the 
            # "if" part of the decision tree above, by collapsing the county data.

    # Final cleanup (convert to integers and remove NaN for the confirmed and deaths [don't touch recovered yet])
    confirmed_df = confirmed_df.replace(np.nan,0).astype('int')
    deaths_df= deaths_df.replace(np.nan,0).astype('int')

    # Add lists of dates to the combined dataframe as a single 'Dates' column
    combined_cnty_df['Dates'] = [dates_list]*len(combined_cnty_df)
    # Add time-series list of confirmed to the combined dataframe as a single 'Confirmed' column
    confirmed_listOlists = confirmed_df[ confirmed_df.columns[confirmed_df.columns!='FIPS'] ].values.tolist()
    combined_cnty_df['Confirmed'] = confirmed_listOlists
    # Add time-series list of deaths to the combined dataframe as a single 'Deaths' column
    deaths_listOlists = deaths_df[ deaths_df.columns[deaths_df.columns!='FIPS'] ].values.tolist()
    combined_cnty_df['Deaths'] = deaths_listOlists
    # Add time-series list of recovered to the combined dataframe as a single 'Recovered' column
    recovered_listOlists = recovered_df[ recovered_df.columns[recovered_df.columns!='FIPS'] ].values.tolist()
    combined_cnty_df['Recovered'] = recovered_listOlists

    # Convert the list of dates into numpy array of days since Jan. 1, 2020 for each observation
    dates = combined_cnty_df[combined_cnty_df['FIPS'] == ClayFIPS]['Dates'].tolist()[0]
    dates_list = []
    for dat in dates:
        dates_list.append( days_since(dat) )
    dates_arr = np.array([dates_list]*len(combined_cnty_df))

    # Convert confirmed/deaths/recovered into arrays
    confirmed_arr = np.array(confirmed_listOlists)
    deaths_arr = np.array(deaths_listOlists)

    # At this point I have arrays where the rows are individiual FIPS (counties) and the columns are
    # (depending on the array) the days since 1/1/2020, number of confirmed cases, number of deaths,
    # and number of recovered.

    # Compute the derivatives (using forward derivative approach)
    dconfirmed_arr = derivative(dates_arr, confirmed_arr)
    ddeaths_arr = derivative(dates_arr, deaths_arr)
    dconfirmed7_arr = derivative_ndays(dates_arr, confirmed_arr, 7)
    ddeaths7_arr = derivative_ndays(dates_arr, deaths_arr, 7)

    # Compute the second derivatives (a bit hinky to use forward derivative again, but...)
    d2confirmed_arr = derivative(dates_arr, dconfirmed_arr)
    d2deaths_arr = derivative(dates_arr, ddeaths_arr)
    d2confirmed7_arr = derivative_ndays(dates_arr, dconfirmed7_arr, 7)
    d2deaths7_arr = derivative_ndays(dates_arr, ddeaths7_arr, 7)

    # Convert numpy arrays to lists of lists for storage in combined dataframe
    combined_cnty_df['dConfirmed'] = dconfirmed_arr.tolist()
    combined_cnty_df['d2Confirmed'] = d2confirmed_arr.tolist()
    combined_cnty_df['dDeaths'] = ddeaths_arr.tolist()
    combined_cnty_df['d2Deaths'] = d2deaths_arr.tolist()
    combined_cnty_df['dConfirmedWk'] = dconfirmed7_arr.tolist()
    combined_cnty_df['d2ConfirmedWk'] = d2confirmed7_arr.tolist()
    combined_cnty_df['dDeathsWk'] = ddeaths7_arr.tolist()
    combined_cnty_df['d2DeathsWk'] = d2deaths7_arr.tolist()

    # Add population data to same array
    combined_cnty_df = pd.merge(combined_cnty_df,county_data_df[['FIPS','POPESTIMATE2019', 'NPOPCHG_2019']], on='FIPS', how='left', copy=True)

    # Rename some columns before export
    combined_cnty_df.rename(columns={ 'Admin2': 'County',
                                    'Province_State': 'State',
                                    'POPESTIMATE2019' : 'PopEst2019',
                                    'NPOPCHG_2019' : 'PopChg2019'}, inplace = True)

    # Clear variables
    del sorted_csvs, dates_list
    del fips_df, raw_df, confirmed_df, deaths_df, recovered_df
    del confirmed_listOlists, deaths_listOlists, recovered_listOlists
    del dates_arr, confirmed_arr, deaths_arr
    del dconfirmed_arr, ddeaths_arr, d2confirmed_arr, d2deaths_arr
    del dconfirmed7_arr, ddeaths7_arr, d2confirmed7_arr, d2deaths7_arr

    ##
    ## Build combined state-level datafiles
    ##

    # Build a dataframe containing legitimate FIPS values using state-level data
    fips_df = state_data_df.copy()
    fips_df.drop(columns=['STNAME', 'POPESTIMATE2019', 'NPOPCHG_2019', 'PPOPCHG_2019'], inplace=True)

    # Scan through the more complete daily files of state level data and construct a single datafile for our use (restricting
    # to US only).  These files are all the same format, but only start on April 12, 2020.  For April 18/19 they accidentally
    # included data from other nations.  So this will need to be purged
    sorted_csvs = []
    for file in csvfiles(daily_state_dir):
        # Set up date information in memory
        this_isodate = isodate2num(file)

        # Append to list of accessed csv files
        sorted_csvs.append(daily_state_dir+file)

        # Grab the data from the CSV file
        raw_df = pd.read_csv(sorted_csvs[-1])

        # Match to the list of known FIPS values from Census data, also removing duplicate rows
        reduced_df = reduce_local_dataframe(raw_df, fips_df)

        # Provide progress report
        idx = len(sorted_csvs)
        # print(f'Processing Date #{idx}: {this_isodate}')

        if (idx == 1):
            # Create combined dataframe sorted by FIPS
            combined_cols = ['FIPS', 'Province_State', 'Lat', 'Long_']
            combined_state_df = reduced_df[combined_cols].copy()

            # Copy pre-existing state dates list and dataframes to store time series data
            dates_list = state_dates_list
            confirmed_df = state_confirmed_df.copy()
            deaths_df = state_deaths_df.copy()
            recovered_df = state_recovered_df.copy()
            incident_rate_df = state_incident_rate_df.copy()
            tested_df = state_tested_df.copy()
            hospitalized_df = state_hospitalized_df.copy()
            mortality_df = state_mortality_df.copy()
            testing_rate_df = state_testing_rate_df.copy()
            hospitalization_rate_df = state_hospitalization_rate_df.copy()

        ## Create dataframes for temporarily storing time series date
        # Append date to list of dates
        #dates_list.append(this_isodate)
        dates_list.append(datetime.strptime(this_isodate, '%Y-%m-%d').date())

        # Store Confirmed by merging reduced list and renaming column
        state_idx += 1
        this_col = "X"+f"{state_idx:03d}"
        confirmed_df = pd.merge(confirmed_df,reduced_df[['FIPS','Confirmed']],on='FIPS', how='left', copy=True)
        confirmed_df.rename(columns={'Confirmed': this_col}, errors="raise", inplace=True)

        # Store Deaths by merging reduced list and renaming column
        deaths_df = pd.merge(deaths_df,reduced_df[['FIPS','Deaths']],on='FIPS', how='left', copy=True)
        deaths_df.rename(columns={'Deaths': this_col}, errors="raise", inplace=True)

        # Store Recovered by merging reduced list and renaming column
        recovered_df = pd.merge(recovered_df,reduced_df[['FIPS','Recovered']],on='FIPS', how='left', copy=True)
        recovered_df.rename(columns={'Recovered': this_col}, errors="raise", inplace=True)

        # Store Incident Rate by merging reduced list and renaming column
        incident_rate_df = pd.merge(incident_rate_df,reduced_df[['FIPS','Incident_Rate']],on='FIPS', how='left', copy=True)
        incident_rate_df.rename(columns={'Incident_Rate': this_col}, errors="raise", inplace=True)

        # Store People Testing by merging reduced list and renaming column
        tested_df = pd.merge(tested_df,reduced_df[['FIPS','People_Tested']],on='FIPS', how='left', copy=True)
        tested_df.rename(columns={'People_Tested': this_col}, errors="raise", inplace=True)

        # Store People Hospitalized by merging reduced list and renaming column
        hospitalized_df = pd.merge(hospitalized_df,reduced_df[['FIPS','People_Hospitalized']],on='FIPS', how='left', copy=True)
        hospitalized_df.rename(columns={'People_Hospitalized': this_col}, errors="raise", inplace=True)

        # Store Mortality Rate by merging reduced list and renaming column
        mortality_df = pd.merge(mortality_df,reduced_df[['FIPS','Mortality_Rate']],on='FIPS', how='left', copy=True)
        mortality_df.rename(columns={'Mortality_Rate': this_col}, errors="raise", inplace=True)

        # Store Testing Rate by merging reduced list and renaming column
        testing_rate_df = pd.merge(testing_rate_df,reduced_df[['FIPS','Testing_Rate']],on='FIPS', how='left', copy=True)
        testing_rate_df.rename(columns={'Testing_Rate': this_col}, errors="raise", inplace=True)

        # Store Hospitalization Rate by merging reduced list and renaming column
        hospitalization_rate_df = pd.merge(hospitalization_rate_df,reduced_df[['FIPS','Hospitalization_Rate']],on='FIPS', how='left', copy=True)
        hospitalization_rate_df.rename(columns={'Hospitalization_Rate': this_col}, errors="raise", inplace=True)

    # Final cleanup (convert values that are integers to to integers)
    confirmed_df = confirmed_df.replace(np.nan,0).astype('int')
    deaths_df = deaths_df.replace(np.nan,0).astype('int')

    # Add lists of dates to the combined dataframe as a single 'Dates' column
    combined_state_df['Dates'] = [dates_list]*len(combined_state_df)
    # Add time-series list of confirmed to the combined dataframe as a single 'Confirmed' column
    confirmed_listOlists = confirmed_df[ confirmed_df.columns[confirmed_df.columns!='FIPS'] ].values.tolist()
    combined_state_df['Confirmed'] = confirmed_listOlists
    # Add time-series list of deaths to the combined dataframe as a single 'Deaths' column
    deaths_listOlists = deaths_df[ deaths_df.columns[deaths_df.columns!='FIPS'] ].values.tolist()
    combined_state_df['Deaths'] = deaths_listOlists
    # Add time-series list of recovered to the combined dataframe as a single 'Recovered' column
    recovered_listOlists = recovered_df[ recovered_df.columns[recovered_df.columns!='FIPS'] ].values.tolist()
    combined_state_df['Recovered'] = recovered_listOlists
    # Add time-series list of incident rate to the combined dataframe as a single 'Incident_Rate' column
    incident_rate_listOlists = incident_rate_df[ incident_rate_df.columns[incident_rate_df.columns!='FIPS'] ].values.tolist()
    combined_state_df['Incident_Rate'] = incident_rate_listOlists
    # Add time-series list of people tested to the combined dataframe as a single 'People_Tested' column
    tested_listOlists = tested_df[ tested_df.columns[tested_df.columns!='FIPS'] ].values.tolist()
    combined_state_df['People_Tested'] = tested_listOlists
    # Add time-series list of people hospitalized to the combined dataframe as a single 'People_Hospitalized' column
    hospitalized_listOlists = hospitalized_df[ hospitalized_df.columns[hospitalized_df.columns!='FIPS'] ].values.tolist()
    combined_state_df['People_Hospitalized'] = hospitalized_listOlists
    # Add time-series list of mortality rates to the combined dataframe as a single 'Mortality_Rate' column
    mortality_listOlists = mortality_df[ mortality_df.columns[mortality_df.columns!='FIPS'] ].values.tolist()
    combined_state_df['Mortality_Rate'] = mortality_listOlists
    # Add time-series list of testing rates to the combined dataframe as a single 'Testing_Rate' column
    testing_rate_listOlists = testing_rate_df[ testing_rate_df.columns[testing_rate_df.columns!='FIPS'] ].values.tolist()
    combined_state_df['Testing_Rate'] = testing_rate_listOlists
    # Add time-series list of hospitalization rates to the combined dataframe as a single 'Hospitalization_Rate' column
    hospitalization_rate_listOlists = hospitalization_rate_df[ hospitalization_rate_df.columns[hospitalization_rate_df.columns!='FIPS'] ].values.tolist()
    combined_state_df['Hospitalization_Rate'] = hospitalization_rate_listOlists

    # Convert the list of dates into numpy array of days since Jan. 1, 2020 for each observation
    dates = combined_state_df[combined_state_df['FIPS'] == MNFIPS]['Dates'].tolist()[0]
    dates_list = []
    for dat in dates:
        dates_list.append( days_since(dat) )
    dates_arr = np.array([dates_list]*len(combined_state_df))

    # Convert confirmed/deaths/recovered into arrays
    confirmed_arr = np.array(confirmed_listOlists)
    deaths_arr = np.array(deaths_listOlists)

    # At this point I have arrays where the rows are individiual FIPS (counties) and the columns are
    # (depending on the array) the days since 1/1/2020, number of confirmed cases, number of deaths,
    # and number of recovered.

    # Compute the derivatives (using forward derivative approach)
    dconfirmed_arr = derivative(dates_arr, confirmed_arr)
    ddeaths_arr = derivative(dates_arr, deaths_arr)
    dconfirmed7_arr = derivative_ndays(dates_arr, confirmed_arr, 7)
    ddeaths7_arr = derivative_ndays(dates_arr, deaths_arr, 7)

    # Compute the second derivatives (a bit hinky to use forward derivative again, but...)
    d2confirmed_arr = derivative(dates_arr, dconfirmed_arr)
    d2deaths_arr = derivative(dates_arr, ddeaths_arr)
    d2confirmed7_arr = derivative_ndays(dates_arr, dconfirmed7_arr, 7)
    d2deaths7_arr = derivative_ndays(dates_arr, ddeaths7_arr, 7)

    # Convert numpy arrays to lists of lists for storage in combined dataframe
    combined_state_df['dConfirmed'] = dconfirmed_arr.tolist()
    combined_state_df['d2Confirmed'] = d2confirmed_arr.tolist()
    combined_state_df['dDeaths'] = ddeaths_arr.tolist()
    combined_state_df['d2Deaths'] = d2deaths_arr.tolist()
    combined_state_df['dConfirmedWk'] = dconfirmed7_arr.tolist()
    combined_state_df['d2ConfirmedWk'] = d2confirmed7_arr.tolist()
    combined_state_df['dDeathsWk'] = ddeaths7_arr.tolist()
    combined_state_df['d2DeathsWk'] = d2deaths7_arr.tolist()

    # Add population data to same array
    combined_state_df = pd.merge(combined_state_df,state_data_df[['FIPS','POPESTIMATE2019', 'NPOPCHG_2019']], on='FIPS', how='left', copy=True)

    # Rename some columns before export
    combined_state_df.rename(columns={ 'Province_State': 'State', 
                                    'POPESTIMATE2019' : 'PopEst2019',
                                    'NPOPCHG_2019' : 'PopChg2019'}, inplace = True)

    # Clear variables
    del sorted_csvs, dates_list
    del fips_df, raw_df, confirmed_df, deaths_df, recovered_df
    del confirmed_listOlists, deaths_listOlists, recovered_listOlists, incident_rate_listOlists
    del tested_listOlists, hospitalized_listOlists, mortality_listOlists, testing_rate_listOlists, hospitalization_rate_listOlists
    del dates_arr, confirmed_arr, deaths_arr
    del dconfirmed_arr, ddeaths_arr, d2confirmed_arr, d2deaths_arr

    # Reset indices before exporting
    ts_us_confirmed_df.reset_index(drop=True, inplace=True)
    ts_us_dead_df.reset_index(drop=True, inplace=True)
    combined_cnty_df.reset_index(drop=True, inplace=True)
    combined_state_df.reset_index(drop=True, inplace=True)

    ## Return the John Hopkins dataframes
    return (ts_us_confirmed_df, ts_us_dead_df, combined_cnty_df, combined_state_df)


def retrieve_goog_mobility_data(county_data_df, state_data_df):
    ###
    ### Google Mobility Data (FIPS Cross-identification Performed)
    ###
    #
    # This data is described at https://www.google.com/covid19/mobility/ and can be downloaded in a single monolithic CSV
    # file at https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv
    #
    # > The data shows how visitors to (or time spent in) categorized places change compared to our baseline days.
    # > A baseline day represents a normal value for that day of the week. The baseline day is the median value from
    # > the 5‑week period Jan 3 – Feb 6, 2020.
    #
    # > For each region-category, the baseline isn’t a single value—it’s 7 individual values. The same number of visitors
    # > on 2 different days of the week, result in different percentage changes. So, we recommend the following:
    # 1. Don’t infer that larger changes mean more visitors or smaller changes mean less visitors.
    # 2. Avoid comparing day-to-day changes. Especially weekends with weekdays.
    # (https://support.google.com/covid19-mobility/answer/9824897?hl=en&ref_topic=9822927)
    #
    # > Note, *Parks* typically means official national parks and not the general outdoors found in rural areas.
    #
    # Also, I'll note that aggregated national data appears to be available by setting `sub_region_1` **and**
    #  `sub_region_2` to `NaN` and state-level data by setting only `sub_region_2` to `NaN`.
    #
    # **Suggested Citation**: Google LLC "Google COVID-19 Community Mobility Reports".
    # https://www.google.com/covid19/mobility/ Accessed: `<Date>.`

    # Google Mobility Data URL
    goog_mobility_csv_url = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
    goog_mobility_df=pd.read_csv(goog_mobility_csv_url, low_memory=False)

    ## Separate data into national-level, state-level, and county-level data deep copies
    goog_mobility_states = goog_mobility_df[(goog_mobility_df['country_region_code'] == 'US') & (goog_mobility_df['sub_region_1'].notna()) & (goog_mobility_df['sub_region_2'].isna())].copy()
    goog_mobility_cnty = goog_mobility_df[(goog_mobility_df['country_region_code'] == 'US') & (goog_mobility_df['sub_region_1'].notna()) & (goog_mobility_df['sub_region_2'].notna())].copy()

    # District of Columbia is both FIPS 11 and FIPS 110, so add its data to county-level mobility data
    dc_rows = goog_mobility_states[goog_mobility_states['sub_region_1'] == 'District of Columbia'].copy()
    dc_rows['sub_region_2'] = dc_rows['sub_region_1']
    goog_mobility_cnty = goog_mobility_cnty.append(dc_rows, ignore_index=True)

    # Undefine the clay county subframe
    del goog_mobility_df

    ##
    ## Determine range of dates of entire dataset to allow filling of unknown dates of data with NaN
    ##
    goog_mobility_cnty['date'].to_list()
    DATEs = [datetime.strptime(x, '%Y-%m-%d').date() for x in goog_mobility_cnty['date'].to_list()]
    first_day = min(DATEs)
    last_day = max(DATEs)
    all_dates_list = list(datetime_range(first_day, last_day))
    all_dates_str = dates2strings(all_dates_list)

    ##
    ## Cross match known US Bureau Census FIPS entries with Google Mobility Data here to allow easier cross-matching later.
    ##

    # Build a dataframe containing legitimate FIPS values using state and county level data
    state_fips_df = state_data_df.copy()
    state_fips_df.drop(columns=['POPESTIMATE2019', 'NPOPCHG_2019', 'PPOPCHG_2019'], inplace=True)
    cnty_fips_df = county_data_df.copy()
    cnty_fips_df.drop(columns=['POPESTIMATE2019', 'NPOPCHG_2019', 'PPOPCHG_2019'], inplace=True)

    ## Match state-level mobility data to FIPS and then drop redundant columns and rename state name to be consistent.
    goog_mobility_states_cleaned = pd.merge(state_fips_df,goog_mobility_states,left_on='STNAME', right_on='sub_region_1', how='left', copy=True)
    goog_mobility_states_cleaned.drop(columns=['country_region_code', 'country_region', 'sub_region_1', 'sub_region_2'], inplace=True)
    goog_mobility_states_cleaned.rename(columns={ 'STNAME': 'state'}, inplace = True)

    ##
    ## Match county-level mobility data to FIPS (trickier as it requires a state and county name match
    ##

    # Comparing the unique county names in cnty_fips_df['CTYNAME_MATCH'] versus goog_mobility_cnty['sub_region_2'] reveals a lot of mismatching is due to
    # US Census bureau naming convention including
    #  ' city' (for cities that are also counties [40 cases]) [This allows matching of Baltimore city versus Baltimore county also]
    #  ' Municipality' (for cities that are also counties)
    #  ' Census Area' (for rural areas, I think)
    cnty_fips_df['CTYNAME_MATCH'] = cnty_fips_df['CTYNAME']
    cnty_fips_df['CTYNAME_MATCH'] = cnty_fips_df['CTYNAME_MATCH'].str.replace(' city','')
    cnty_fips_df['CTYNAME_MATCH'] = cnty_fips_df['CTYNAME_MATCH'].str.replace(' Municipality','')
    cnty_fips_df['CTYNAME_MATCH'] = cnty_fips_df['CTYNAME_MATCH'].str.replace(' Census Area','')
    cnty_fips_df['CTYNAME_MATCH'] = cnty_fips_df['CTYNAME_MATCH'].str.replace(' City and Borough','')
    cnty_fips_df['CTYNAME_MATCH'] = cnty_fips_df['CTYNAME_MATCH'].str.replace(' Borough','')

    goog_mobility_cnty['sub_region_2_MATCH'] = goog_mobility_cnty['sub_region_2']
    goog_mobility_cnty['sub_region_2_MATCH'] = goog_mobility_cnty['sub_region_2_MATCH'].str.replace(' Borough','')
    # Fix alternate spelling of LaSalle Parish, Louisiana
    goog_mobility_cnty['sub_region_2_MATCH'] = goog_mobility_cnty['sub_region_2_MATCH'].str.replace('La Salle Parish','LaSalle Parish')

    # A lot of rural areas just appear to have no data in the Google Mobility dataset. The next cell (commented out) was used to confirm the only missing counies
    # were ones in which there was 'nan' for the date column, indicating no matches in the Google Mobility dataset.

    ## This leftward match means EVERY county FIPS should still be represented, although need to confirm mismatches
    goog_mobility_cnty_cleaned = pd.merge(cnty_fips_df,goog_mobility_cnty,left_on=['STNAME', 'CTYNAME_MATCH'], right_on=['sub_region_1', 'sub_region_2_MATCH'], how='left', copy=True)

    ##
    ## Check the date column in the reduced data to see if it is a real match or just a marker for a non-match
    ##

    # As a test, loop through all the county FIPS codes and see which are NOT represented in the mobility data
    unmatched_cnt = 0
    cleared_states = []
    states_list = []
    last_state = 'Alaska' # Set to avoid issue in loop below with undefined variable

    fatal_error = 0. # Count fatal_error and stop if we have any
    cnt = 0
    bad_entries = ""
    for FIPS in cnty_fips_df['FIPS']:
        # Check this FIPS number
        rows = goog_mobility_cnty_cleaned[goog_mobility_cnty_cleaned['FIPS'] == FIPS].copy()
        matches = rows.shape[0]
        city = rows['CTYNAME_MATCH'].iloc[0]
        state = rows['STNAME'].iloc[0]
        date = rows['date'].iloc[0]
        if (matches == 1 & pd.isna(date)):

            if state not in states_list:
                states_list.append(state)

            if ((state != last_state) & (last_state not in cleared_states)):
                # Print counties lists
                reduced_ctys = goog_mobility_cnty_cleaned[goog_mobility_cnty_cleaned['STNAME'] == last_state]['CTYNAME_MATCH'].unique()
                mobility_ctys = goog_mobility_cnty[goog_mobility_cnty['sub_region_1'] == last_state]['sub_region_2'].unique()

                mismatch = len(reduced_ctys) - len(mobility_ctys)  # Number of missing counties
                if (cnt != mismatch):
                    print(f"\nFor State {last_state}:")
                    print(bad_entries)
                    print(f"\n  reduced counties: {reduced_ctys} {len(reduced_ctys)}\n")
                    print(f"  mobility counties: {mobility_ctys} {len(mobility_ctys)}\n")
                    for cty in reduced_ctys:
                        if cty not in mobility_ctys:
                            print(f"{cty} missing from Mobility counties")

                    print(f"WARNING: {cnt} no real matches vs. {mismatch} fewer counties in Google mobility data!\n")
                    fatal_error += 1

                cnt = 0 # Reset the count of mismatches in this state
                bad_entries = ""

            if (state not in cleared_states):
                cnt += 1
                # Count this as a mismatch
                bad_entries += f'{cnt}) {city}, {state} ({FIPS}) contains no real matches.\n'

            last_state = state
            unmatched_cnt += 1

    # Print counties for last state considered=
    reduced_ctys = goog_mobility_cnty_cleaned[goog_mobility_cnty_cleaned['STNAME'] == last_state]['CTYNAME_MATCH'].unique()
    mobility_ctys = goog_mobility_cnty[goog_mobility_cnty['sub_region_1'] == last_state]['sub_region_2'].unique()
    mismatch = len(reduced_ctys) - len(mobility_ctys)  # Number of missing counties
    if (cnt != mismatch):
        print(f"\nFor State {last_state}:")
        print(bad_entries)
        print(f"\n  reduced counties: {reduced_ctys} {len(reduced_ctys)}\n")
        print(f"  mobility counties: {mobility_ctys} {len(mobility_ctys)}\n")
        for cty in reduced_ctys:
            if cty not in mobility_ctys:
                print(f"{cty} missing from Mobility counties")
        print(f"WARNING: {cnt} no real matches vs. {mismatch} fewer counties in Google mobility data!\n")
        fatal_error += 1

    #print(f"A total of {unmatched_cnt} FIPS not matched to Google mobility data (if nothing printed above this, all US Census Bureau counties accounted for)")
    if (fatal_error != 0):
        raise ValueError("Something went wrong with cross-ID of Google mobility county-level data to FIPS records.")

    ##
    ## Conversion of separate dataframe rows as dates into a single row per location with time series stored as lists.
    ## NOTE: These time serires have DIFFERENT LENGTHS for different FIPS as certain counties in particular as missing data.
    ## For the state data
    ##
    goog_mobility_states_reduced = state_fips_df.copy()

    # Create blank lists of lists
    dates_listOlists = []
    retail_listOlists = []
    grocery_listOlists = []
    parks_listOlists = []
    transit_listOlists = []
    workplaces_listOlists = []
    residential_listOlists = []

    for fips in state_fips_df['FIPS']:
        #print(f"Processing FIPS {fips}")
        # Pull only the data for this FIPS number and extract the time series
        subset = goog_mobility_states_cleaned[goog_mobility_states_cleaned['FIPS'] == fips].copy()
        desired_columns = subset.columns[(subset.columns!='FIPS') & (subset.columns!='state')]
        timeseries = subset[desired_columns].copy()
        timeseries = timeseries.set_index('date')
        trans = timeseries.T

        # Convert the time series into lists in memory
        dates_str_list = trans[ trans.columns[(trans.columns!='date')]].columns.tolist()

        # Convert the lists corresponding to each column into dictionaries for indexing
        retail_dict = dict(zip(dates_str_list, trans.loc['retail_and_recreation_percent_change_from_baseline'].values.tolist()))
        grocery_dict = dict(zip(dates_str_list, trans.loc['grocery_and_pharmacy_percent_change_from_baseline'].values.tolist()))
        parks_dict = dict(zip(dates_str_list, trans.loc['parks_percent_change_from_baseline'].values.tolist()))
        transit_dict = dict(zip(dates_str_list, trans.loc['transit_stations_percent_change_from_baseline'].values.tolist()))
        workplaces_dict = dict(zip(dates_str_list, trans.loc['workplaces_percent_change_from_baseline'].values.tolist()))
        residential_dict = dict(zip(dates_str_list, trans.loc['residential_percent_change_from_baseline'].values.tolist()))

        #
        # Now brute force through all dates from first to last day and look for matches in each date,
        # filling the voids with NaN
        #
        # Start with blank lists for this FIPS
        retail_list = []
        grocery_list = []
        parks_list = []
        transit_list = []
        workplaces_list = []
        residential_list = []

        for key in all_dates_str:
            try:
                retail_list.append(retail_dict[key])
                grocery_list.append(grocery_dict[key])
                parks_list.append(parks_dict[key])
                transit_list.append(transit_dict[key])
                workplaces_list.append(workplaces_dict[key])
                residential_list.append(residential_dict[key])
            except:
                retail_list.append(np.nan)
                grocery_list.append(np.nan)
                parks_list.append(np.nan)
                transit_list.append(np.nan)
                workplaces_list.append(np.nan)
                residential_list.append(np.nan)

        # Add lists to lists
        dates_listOlists.append(all_dates_list) # Add all dates
        retail_listOlists.append(retail_list)
        grocery_listOlists.append(grocery_list)
        parks_listOlists.append(parks_list)
        transit_listOlists.append(transit_list)
        workplaces_listOlists.append(workplaces_list)
        residential_listOlists.append(residential_list)

    # Load data (in form of lists of lists) into pandas dataframe
    goog_mobility_states_reduced['dates'] = dates_listOlists
    goog_mobility_states_reduced['retail_and_recreation_percent_change_from_baseline'] = retail_listOlists
    goog_mobility_states_reduced['grocery_and_pharmacy_percent_change_from_baseline'] = grocery_listOlists
    goog_mobility_states_reduced['parks_percent_change_from_baseline'] = parks_listOlists
    goog_mobility_states_reduced['transit_stations_percent_change_from_baseline'] = transit_listOlists
    goog_mobility_states_reduced['workplaces_percent_change_from_baseline'] = workplaces_listOlists
    goog_mobility_states_reduced['residential_percent_change_from_baseline'] = residential_listOlists

    # Rename STNAME column to state
    goog_mobility_states_reduced.rename(columns={'STNAME': 'state'}, errors="raise", inplace=True)

    ##
    ## Conversion of separate dataframe rows as dates into a single row per location with time series stored as lists
    ## For the county data
    ##
    goog_mobility_cnty_reduced = cnty_fips_df.copy()

    # Create blank lists of lists
    dates_listOlists = []
    retail_listOlists = []
    grocery_listOlists = []
    parks_listOlists = []
    transit_listOlists = []
    workplaces_listOlists = []
    residential_listOlists = []

    # Diagnosing FutureWarning
    # /Users/juan/miniconda3/envs/astro37/lib/python3.7/site-packages/pandas/core/indexes/base.py:122:
    # FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
    #  result = op(self.values, np.asarray(other))
    # Triggered around FIPS 2100 (Not the first NON-MATCHED cell), but can't isolate down
    # since the Warning doesn't stop code execution.

    for fips in cnty_fips_df['FIPS']:
        # print(f"Processing FIPS {fips}")
        # Pull only the data for this FIPS number and extract the time series
        # Convert fips to int64 to avoid numpy warning
        fips64 = np.int64(fips)
        subset = goog_mobility_cnty_cleaned[goog_mobility_cnty_cleaned['FIPS'] == fips64].copy()
        timeseries = subset[subset.columns[(subset.columns!='FIPS') & (subset.columns!='state')]].copy()
        timeseries = timeseries.set_index('date')
        trans = timeseries.T

        # Convert the time series into lists in memory
        dates_str_list = trans[ trans.columns[(trans.columns!='date')]].columns.tolist()

        # Convert the lists corresponding to each column into dictionaries for indexing
        retail_dict = dict(zip(dates_str_list, trans.loc['retail_and_recreation_percent_change_from_baseline'].values.tolist()))
        grocery_dict = dict(zip(dates_str_list, trans.loc['grocery_and_pharmacy_percent_change_from_baseline'].values.tolist()))
        parks_dict = dict(zip(dates_str_list, trans.loc['parks_percent_change_from_baseline'].values.tolist()))
        transit_dict = dict(zip(dates_str_list, trans.loc['transit_stations_percent_change_from_baseline'].values.tolist()))
        workplaces_dict = dict(zip(dates_str_list, trans.loc['workplaces_percent_change_from_baseline'].values.tolist()))
        residential_dict = dict(zip(dates_str_list, trans.loc['residential_percent_change_from_baseline'].values.tolist()))

        #
        # Now brute force through all dates from first to last day and look for matches in each date,
        # filling the voids with NaN
        #
        # Start with blank lists for this FIPS
        retail_list = []
        grocery_list = []
        parks_list = []
        transit_list = []
        workplaces_list = []
        residential_list = []

        for key in all_dates_str:
            try:
                retail_list.append(retail_dict[key])
                grocery_list.append(grocery_dict[key])
                parks_list.append(parks_dict[key])
                transit_list.append(transit_dict[key])
                workplaces_list.append(workplaces_dict[key])
                residential_list.append(residential_dict[key])
            except:
                retail_list.append(np.nan)
                grocery_list.append(np.nan)
                parks_list.append(np.nan)
                transit_list.append(np.nan)
                workplaces_list.append(np.nan)
                residential_list.append(np.nan)

        # Add lists to lists
        dates_listOlists.append(all_dates_list) # Add all dates
        retail_listOlists.append(retail_list)
        grocery_listOlists.append(grocery_list)
        parks_listOlists.append(parks_list)
        transit_listOlists.append(transit_list)
        workplaces_listOlists.append(workplaces_list)
        residential_listOlists.append(residential_list)

    # Results in error ValueError: Length of values does not match length of index
    goog_mobility_cnty_reduced['dates'] = dates_listOlists
    goog_mobility_cnty_reduced['retail_and_recreation_percent_change_from_baseline'] = retail_listOlists
    goog_mobility_cnty_reduced['grocery_and_pharmacy_percent_change_from_baseline'] = grocery_listOlists
    goog_mobility_cnty_reduced['parks_percent_change_from_baseline'] = parks_listOlists
    goog_mobility_cnty_reduced['transit_stations_percent_change_from_baseline'] = transit_listOlists
    goog_mobility_cnty_reduced['workplaces_percent_change_from_baseline'] = workplaces_listOlists
    goog_mobility_cnty_reduced['residential_percent_change_from_baseline'] = residential_listOlists

    # Rename STNAME and CTYNAME columns to state and county, drop redundant columns
    goog_mobility_cnty_reduced.rename(columns={'STNAME': 'state', 'CTYNAME': 'county'}, errors="raise", inplace=True)
    goog_mobility_cnty_reduced.drop(columns=['CTYNAME_MATCH'], inplace=True)

    # Reset indices before exporting
    goog_mobility_cnty_reduced.reset_index(drop=True, inplace=True)
    goog_mobility_states_reduced.reset_index(drop=True, inplace=True)

    return (goog_mobility_cnty_reduced, goog_mobility_states_reduced)


def retrieve_aapl_mobility_data(county_data_df, state_data_df):
    ###
    ### Apple Mobility Data  (FIPS cross-identification performed)
    ###
    #
    # This data is described at https://www.apple.com/covid19/mobility and can be downloaded
    # in a single monolithic CSV file whose URL is hidden in the mobility page link and appears
    # to be updated regularly. I did find a refernce to a JSON file online that contains
    # the necessary info to resconstruct the CSV file link.
    #
    # **About this Data (copied from Apple's site)**: The CSV file on this site show a relative
    # volume of directions requests per country/region, sub-region or city compared to a baseline
    # volume on January 13th, 2020. We define our day as midnight-to-midnight, Pacific time.
    # Cities are defined as the greater metropolitan area and their geographic boundaries remain
    # constant across the data set. In many countries/regions, sub-regions, and cities, relative
    # volume has increased since January 13th, consistent with normal, seasonal usage of Apple
    # Maps. Day of week effects are important to normalize as you use this data. Data that is
    # sent from users’ devices to the Maps service is associated with random, rotating
    # identifiers so Apple doesn’t have a profile of individual movements and searches. Apple
    # Maps has no demographic information about our users, so we can’t make any statements
    # about the representativeness of usage against the overall population.
    #
    # Apple tracks three kinds of Apple Maps routing requests: Driving, Walking, Transit.
    # But the only data available at the state and county level is the Driving data.
    #
    # **Developer Notes**: Apple's mobility data only exists for 2090 out of 3142 counties 
    # in the US.

    # Scraping the original Apple page was proving tricky as it had a bunch of javascript
    # used to generate the URL, so I poked around and found a reference
    # at https://www.r-bloggers.com/get-apples-mobility-data/ to a JSON file at a stable
    # URL that can be used to construct the appropriate URL for the current
    # datafile.

    aapl_mobility_json = "https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json"
    aapl_server = "https://covid19-static.cdn-apple.com/"
    result = requests.get(aapl_mobility_json)
    # Proceed if we successfully pulled the page (HTTP status code 200)
    if (result.status_code == 200):
        # Apple Mobility Data URL
        jsondata = result.json()
        aapl_mobility_csv_url = aapl_server+jsondata['basePath']+jsondata['regions']['en-us']['csvPath']
        aapl_mobility_df=pd.read_csv(aapl_mobility_csv_url, low_memory=False)

    # There are four 'geo_types' (aapl_mobility_df['geo_type' == ].unique() returns ['country/region', 'city', 'sub-region', 'county'])
    # Checking those types here

    # Get Washington DC Data
    dc_entry = aapl_mobility_df[(aapl_mobility_df['country'] == 'United States') 
                                & (aapl_mobility_df['region'] =='Washington DC') 
                                & (aapl_mobility_df['transportation_type'] =='driving')].copy()
    dc_entry['region'] = 'District of Columbia'
    dc_entry['sub-region'] = 'District of Columbia'

    # Get state-level mobility data from Apple (only contains 'transportation_type' of 'driving')
    aapl_mobility_states = aapl_mobility_df[(aapl_mobility_df['geo_type'] == 'sub-region') & (aapl_mobility_df['country'] == 'United States') ].copy()
    # Append DC data to state-level data
    aapl_mobility_states = aapl_mobility_states.append(dc_entry, ignore_index=True)
    # Remove redundant columns and rename columns to be more precise
    aapl_mobility_states.drop(columns=['country', 'geo_type', 'sub-region'], inplace=True)
    aapl_mobility_states.rename(columns={ 'region': 'state'}, inplace = True)
    state_transport = aapl_mobility_states['transportation_type'].unique().tolist()
    #print("Apple mobility data at state level transportation types: "+",".join(state_transport)+"\n" )
    # Assuming there is still only a 'driving' transportation type, drop those redundant columns
    if (len(state_transport) == 1):
        aapl_mobility_states.drop(columns=['transportation_type', 'alternative_name'], inplace=True)
    # Purge territories
    aapl_mobility_states = aapl_mobility_states[aapl_mobility_states.state != 'Guam'].copy()
    aapl_mobility_states = aapl_mobility_states[aapl_mobility_states.state != 'Puerto Rico'].copy()
    aapl_mobility_states = aapl_mobility_states[aapl_mobility_states.state != 'Virgin Islands'].copy()

    # Get county-level mobility data from Apple
    aapl_mobility_cnty = aapl_mobility_df[(aapl_mobility_df['geo_type'] == 'county') & (aapl_mobility_df['country'] == 'United States')].copy()
    aapl_mobility_cnty.drop(columns=['country', 'geo_type'], inplace=True)
    aapl_mobility_cnty.rename(columns={ 'sub-region': 'state', 'region': 'county'}, inplace = True)
    cnty_transport = aapl_mobility_cnty['transportation_type'].unique().tolist()
    #print("Apple mobility data at county level transportation types: "+",".join(state_transport)+"\n" )
    # Assuming there is still only a 'driving' transportation type, drop those redundant columns
    if (len(cnty_transport) == 1):
        aapl_mobility_cnty.drop(columns=['transportation_type', 'alternative_name'], inplace=True)

    # Purge complete Apple mobility dataframe once subsets built
    del aapl_mobility_df

    ##
    ## Cross-match known US Bureau Census FIPS entries with Apple Mobility Data here to allow easier cross-matching later.
    ##

    # Build a dataframe containing legitimate FIPS values using state and county level data
    state_fips_df = state_data_df.copy()
    state_fips_df.drop(columns=['POPESTIMATE2019', 'NPOPCHG_2019', 'PPOPCHG_2019'], inplace=True)
    cnty_fips_df = county_data_df.copy()
    cnty_fips_df.drop(columns=['POPESTIMATE2019', 'NPOPCHG_2019', 'PPOPCHG_2019'], inplace=True)

    ##
    ## Match state-level mobility data to FIPS and then drop redundant columns and rename state name to be consistent.
    ##
    aapl_mobility_states_cleaned = pd.merge(state_fips_df,aapl_mobility_states,left_on='STNAME', right_on='state', how='left', copy=True)
    aapl_mobility_states_cleaned.drop(columns=['STNAME'], inplace=True)

    # Convert all the mobility data into one massive list of lists (and columns into dates list), this will allow collapsing multiple columns into lists
    dates_str_list = aapl_mobility_states_cleaned[ aapl_mobility_states_cleaned.columns[(aapl_mobility_states_cleaned.columns!='FIPS') & (aapl_mobility_states_cleaned.columns!='state')] ].columns.tolist()
    dates_list = [datetime.fromisoformat(day).date() for day in dates_str_list]
    driving_mobility_listOlists = aapl_mobility_states_cleaned[ aapl_mobility_states_cleaned.columns[(aapl_mobility_states_cleaned.columns!='FIPS') & (aapl_mobility_states_cleaned.columns!='state')] ].values.tolist()

    # Create reduced mobility data file with all the data collapsed into lists
    aapl_mobility_states_reduced = aapl_mobility_states_cleaned[['FIPS','state']].copy()
    aapl_mobility_states_reduced['dates'] = [dates_list]*len(aapl_mobility_states_reduced)
    aapl_mobility_states_reduced['driving_mobility'] = driving_mobility_listOlists

    ##
    ## Match county-level mobility data to FIPS 
    ##
    # Rename county names which are actually cities
    aapl_mobility_cnty['county'] = aapl_mobility_cnty['county'].str.replace(' City',' city')
    aapl_mobility_cnty['county'] = aapl_mobility_cnty['county'].str.replace(' city and Borough',' City and Borough')
    aapl_mobility_cnty['county'] = aapl_mobility_cnty['county'].str.replace('Carson city','Carson City')
    aapl_mobility_cnty['county'] = aapl_mobility_cnty['county'].str.replace('James city County','James City County')

    # Attempt the match
    aapl_mobility_cnty_cleaned = pd.merge(cnty_fips_df,aapl_mobility_cnty,left_on=['STNAME', 'CTYNAME'], right_on=['state', 'county'], how='left', copy=True)

    #print(f"There are {len(cnty_fips_df)} counties, Apple mobility data exists for {len(aapl_mobility_cnty)} counties.")
    expected = len(cnty_fips_df)-len(aapl_mobility_cnty)
    nomatch = len(aapl_mobility_cnty_cleaned[aapl_mobility_cnty_cleaned['state'].isna()])
    n_errors = nomatch - expected
    #print(f"When cross-matching, there are {nomatch} Census counties with no match, expected {expected} [We need to account for {n_errors} errors].")

    states_list = []
    cleared_states = []
    last_state = "Alabama"
    fatal_error = 0
    bad_entries = ""
    cnt = 0
    unmatched_cnt = 0

    for FIPS in cnty_fips_df['FIPS']:
        # Check this FIPS number
        row  = aapl_mobility_cnty_cleaned[aapl_mobility_cnty_cleaned['FIPS'] == FIPS]
        city = row['CTYNAME'].iloc[0]
        state = row['STNAME'].iloc[0]
        aapl_state= row['state'].iloc[0]
        if (pd.isna(aapl_state)):

            if state not in states_list:
                states_list.append(state)

            if ((state != last_state) & (last_state not in cleared_states)):
                # Check on number of counties matching on per state basis
                reduced_ctys = aapl_mobility_cnty_cleaned[aapl_mobility_cnty_cleaned['STNAME'] == last_state]['CTYNAME'].unique()
                mobility_ctys = aapl_mobility_cnty[aapl_mobility_cnty['state'] == last_state]['county'].unique()
                mismatch = len(reduced_ctys) - len(mobility_ctys)

                if (cnt != mismatch):
                    print(f"\nFor State {last_state}:")
                    print(bad_entries)
                    print(f"\n  reduced counties: {reduced_ctys} {len(reduced_ctys)}\n")
                    print(f"  mobility counties: {mobility_ctys} {len(mobility_ctys)}\n")

                    print(f"WARNING: {cnt} not matched vs. {mismatch} fewer counties in Apple mobility data!\n")
                    fatal_error += 1

                cnt = 0 # Reset the count of mismatches in this state
                bad_entries = ""

            if (state not in cleared_states):
                cnt += 1
                # Count this as a mismatch
                bad_entries += f'{cnt}) {city}, {state} ({FIPS}) contains no real matches.\n'

            last_state = state
            unmatched_cnt += 1

    # Check lists of counties
    reduced_ctys = aapl_mobility_cnty_cleaned[aapl_mobility_cnty_cleaned['STNAME'] == last_state]['CTYNAME'].unique()
    mobility_ctys = aapl_mobility_cnty[aapl_mobility_cnty['state'] == last_state]['county'].unique()
    mismatch = len(reduced_ctys) - len(mobility_ctys)

    if (cnt != mismatch):
        print(f"\nFor State {last_state}:")
        print(bad_entries)
        print(f"\n  reduced counties: {reduced_ctys} {len(reduced_ctys)}\n")
        print(f"  mobility counties: {mobility_ctys} {len(mobility_ctys)}\n")

        print(f"WARNING: {cnt} not matched vs. {mismatch} fewer counties in Apple mobility data!\n")
        fatal_error += 1
    #print(f"A total of {unmatched_cnt} FIPS not matched to Apple mobility data (if nothing printed above this, all US Census Bureau counties accounted for)")
    if (fatal_error != 0):
        raise ValueError("Something went wrong with cross-ID of Apple mobility county-level data to FIPS records.")

    # Purge Redundant Columns
    aapl_mobility_cnty_cleaned.drop(columns=['county', 'state'], inplace=True)
    aapl_mobility_cnty_cleaned.rename(columns={ 'STNAME': 'state', 'CTYNAME': 'county'}, inplace = True)

    # Convert all the county level mobility data into one massive list of lists (and columns into dates list), this will allow collapsing multiple columns into lists
    dates_str_list = aapl_mobility_cnty_cleaned[ aapl_mobility_cnty_cleaned.columns[(aapl_mobility_cnty_cleaned.columns!='FIPS') & (aapl_mobility_cnty_cleaned.columns!='state') & (aapl_mobility_cnty_cleaned.columns!='county')] ].columns.tolist()
    dates_list = [datetime.fromisoformat(day).date() for day in dates_str_list]
    driving_mobility_listOlists = aapl_mobility_cnty_cleaned[ aapl_mobility_cnty_cleaned.columns[(aapl_mobility_cnty_cleaned.columns!='FIPS') & (aapl_mobility_cnty_cleaned.columns!='state')& (aapl_mobility_cnty_cleaned.columns!='county')] ].values.tolist()

    # Create reduced mobility data file with all the data collapsed into lists
    # Did this goofy line next becayse 'county' wasn't being recognized with shorthand approach.  WTF?
    aapl_mobility_cnty_reduced = aapl_mobility_cnty_cleaned[ aapl_mobility_cnty_cleaned.columns[(aapl_mobility_cnty_cleaned.columns=='FIPS') | (aapl_mobility_cnty_cleaned.columns=='state') | (aapl_mobility_cnty_cleaned.columns=='county')] ].copy()
    aapl_mobility_cnty_reduced['dates'] = [dates_list]*len(aapl_mobility_cnty_reduced)
    aapl_mobility_cnty_reduced['driving_mobility'] = driving_mobility_listOlists

    # Reset indices before exporting
    aapl_mobility_cnty_reduced.reset_index(drop=True, inplace=True)
    aapl_mobility_states_reduced.reset_index(drop=True, inplace=True)

    return(aapl_mobility_cnty_reduced, aapl_mobility_states_reduced)


def retrieve_imhe_data(county_data_df, state_data_df):
    ###
    ### Institute for Health Metrics and Evaluation (IMHE) Data on Local Resources
    ### (FIPS cross-identification performed)
    #
    # Institute for Health Metrics and Evaluation data on local resources is available
    # at http://www.healthdata.org/covid/data-downloads although data only has
    # state level resolution.
    #
    # **Suggested Citation**: Institute for Health Metrics and Evaluation (IHME).
    # COVID-19 Hospital Needs and Death Projections. Seattle, United States of America:
    # Institute for Health Metrics and Evaluation (IHME), University of Washington, 2020.

    ##
    ## Retrieve the IMHE data which is in the form of a ZIP file
    ##

    # I modeled the extraction of the ZIP data downloaded from a URL without writing to disk on examples found at
    # https://stackoverflow.com/questions/5710867/downloading-and-unzipping-a-zip-file-without-writing-to-disk

    # Retrieve the ZIP file into memory and get the filelist
    imhe_url = "https://ihmecovid19storage.blob.core.windows.net/latest/ihme-covid19.zip"
    imhe_result = requests.get(imhe_url).content
    zipfile = ZipFile(BytesIO(imhe_result))
    filelist = zipfile.namelist()

    # The challenge is that the files in the ZIP file are placed in a date based directory, so perform a search for the proper strings in the filenames
    summary_csv = [name for name in filelist if "Summary_stats_all_locs" in name][0]
    hospitalization_csv = [name for name in filelist if "Reference_hospitalization_all_locs" in name][0]

    # Get the CSV data into pandas dataframes
    imhe_summary_df=pd.read_csv(zipfile.open(summary_csv), low_memory=False)
    imhe_hospitalizations_df=pd.read_csv(zipfile.open(hospitalization_csv), low_memory=False)

    ##
    ## Summary data processing
    ##

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

    # Match to FIPS data
    state_fips_df = state_data_df.copy()
    state_fips_df.drop(columns=['POPESTIMATE2019', 'NPOPCHG_2019', 'PPOPCHG_2019'], inplace=True)
    imhe_summary_cleaned = pd.merge(state_fips_df,imhe_summary_df,left_on='STNAME', right_on='location_name', how='left', copy=True)

    # Dropped redundant state name and excess capacity of beds, since computable from available columns
    imhe_summary_cleaned.drop(columns=['STNAME', 'available_all_nbr', 'available_icu_nbr'], inplace=True)
    imhe_summary_cleaned.rename(columns={ 'location_name': 'state' }, inplace = True)

    # Convert all dates to datetime objects in memory
    cols_w_dates = ['peak_bed_day_mean', 'peak_bed_day_lower', 'peak_bed_day_upper',
                    'peak_icu_bed_day_mean', 'peak_icu_bed_day_lower', 'peak_icu_bed_day_upper',
                    'peak_vent_day_mean', 'peak_vent_day_lower', 'peak_vent_day_upper',
                    'travel_limit_start_date', 'travel_limit_end_date',
                    'stay_home_start_date', 'stay_home_end_date',
                    'educational_fac_start_date', 'educational_fac_end_date',
                    'any_gathering_restrict_start_date', 'any_gathering_restrict_end_date',
                    'any_business_start_date', 'any_business_end_date',
                    'all_non-ess_business_start_date', 'all_non-ess_business_end_date']
    for col in cols_w_dates:
        imhe_summary_cleaned[col]= pd.to_datetime(imhe_summary_cleaned[col])

    ##
    ## Hospitalization data processing
    ##

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

    # Match to FIPS data
    imhe_hospitalizations_cleaned = pd.merge(state_fips_df,imhe_hospitalizations_df,left_on='STNAME', right_on='location_name', how='left', copy=True)

    # Dropped redundant state name columns (and data on mobility, and bed overuse since its computable from other data )
    imhe_hospitalizations_cleaned.drop(columns=['STNAME', 'V1', 'bedover_mean', 'bedover_lower', 'bedover_upper', 
                                                'icuover_mean', 'icuover_lower', 'icuover_upper', 
                                                'mobility_data_type', 'mobility_composite' ], inplace=True)
    imhe_hospitalizations_cleaned.rename(columns={ 'location_name': 'state' }, inplace = True)

    ##
    ## Add conversion of separate dataframe rows as dates into a single row per location with time series stored as lists
    ## For the county data
    ##
    imhe_hospitalizations_reduced = state_fips_df.copy()
    imhe_hospitalizations_reduced.rename(columns={ 'STNAME': 'state' }, inplace = True)

    # Create blank lists of lists
    dates_listOlists = []
    allbed_mean_listOlists = []
    allbed_lower_listOlists = []
    allbed_upper_listOlists = []
    ICUbed_mean_listOlists = []
    ICUbed_lower_listOlists = []
    ICUbed_upper_listOlists = []
    InvVen_mean_listOlists = []
    InvVen_lower_listOlists = []
    InvVen_upper_listOlists = []
    deaths_mean_listOlists = []
    deaths_lower_listOlists = []
    deaths_upper_listOlists = []
    admis_mean_listOlists = []
    admis_lower_listOlists = []
    admis_upper_listOlists = []
    newICU_mean_listOlists = []
    newICU_lower_listOlists = []
    newICU_upper_listOlists = []
    totdea_mean_listOlists = []
    totdea_lower_listOlists = []
    totdea_upper_listOlists = []
    deaths_mean_smoothed_listOlists = []
    deaths_lower_smoothed_listOlists = []
    deaths_upper_smoothed_listOlists = []
    totdea_mean_smoothed_listOlists = []
    totdea_lower_smoothed_listOlists = []
    totdea_upper_smoothed_listOlists = []
    total_tests_data_type_listOlists = []
    total_tests_listOlists = []
    confirmed_infections_listOlists = []
    est_infections_mean_listOlists = []
    est_infections_lower_listOlists = []
    est_infections_upper_listOlists = []

    for fips in state_fips_df['FIPS']:
        # Pull only the data for this FIPS number and extract the time series
        subset = imhe_hospitalizations_cleaned[imhe_hospitalizations_cleaned['FIPS'] == fips].copy()
        timeseries = subset[subset.columns[(subset.columns!='FIPS') & (subset.columns!='state')   ]].copy()
        timeseries = timeseries.set_index('date')
        trans = timeseries.T

        # Convert the time series into lists in memory
        dates_str_list = trans[ trans.columns[(trans.columns!='date')]].columns.tolist()
        # Convert dates list to datetime.dates list (this approach handles blank counties, returning empty list)
        dates_list = [datetime.fromisoformat(day).date() for day in dates_str_list]

        allbed_mean_list = trans.loc['allbed_mean'].values.tolist()
        allbed_lower_list = trans.loc['allbed_lower'].values.tolist()
        allbed_upper_list = trans.loc['allbed_upper'].values.tolist()
        ICUbed_mean_list = trans.loc['ICUbed_mean'].values.tolist()
        ICUbed_lower_list = trans.loc['ICUbed_lower'].values.tolist()
        ICUbed_upper_list = trans.loc['ICUbed_upper'].values.tolist()
        InvVen_mean_list = trans.loc['InvVen_mean'].values.tolist()
        InvVen_lower_list = trans.loc['InvVen_lower'].values.tolist()
        InvVen_upper_list = trans.loc['InvVen_upper'].values.tolist()
        deaths_mean_list = trans.loc['deaths_mean'].values.tolist()
        deaths_lower_list = trans.loc['deaths_lower'].values.tolist()
        deaths_upper_list = trans.loc['deaths_upper'].values.tolist()
        admis_mean_list = trans.loc['admis_mean'].values.tolist()
        admis_lower_list = trans.loc['admis_lower'].values.tolist()
        admis_upper_list = trans.loc['admis_upper'].values.tolist()
        newICU_mean_list = trans.loc['newICU_mean'].values.tolist()
        newICU_lower_list = trans.loc['newICU_lower'].values.tolist()
        newICU_upper_list = trans.loc['newICU_upper'].values.tolist()
        totdea_mean_list = trans.loc['totdea_mean'].values.tolist()
        totdea_lower_list = trans.loc['totdea_lower'].values.tolist()
        totdea_upper_list = trans.loc['totdea_upper'].values.tolist()
        deaths_mean_smoothed_list = trans.loc['deaths_mean_smoothed'].values.tolist()
        deaths_lower_smoothed_list = trans.loc['deaths_lower_smoothed'].values.tolist()
        deaths_upper_smoothed_list = trans.loc['deaths_upper_smoothed'].values.tolist()
        totdea_mean_smoothed_list = trans.loc['totdea_mean_smoothed'].values.tolist()
        totdea_lower_smoothed_list = trans.loc['totdea_lower_smoothed'].values.tolist()
        totdea_upper_smoothed_list = trans.loc['totdea_upper_smoothed'].values.tolist()
        total_tests_data_type_list = trans.loc['total_tests_data_type'].values.tolist()
        total_tests_list = trans.loc['total_tests'].values.tolist()
        confirmed_infections_list = trans.loc['confirmed_infections'].values.tolist()
        est_infections_mean_list = trans.loc['est_infections_mean'].values.tolist()
        est_infections_lower_list = trans.loc['est_infections_lower'].values.tolist()
        est_infections_upper_list = trans.loc['est_infections_upper'].values.tolist()

        # Add lists to lists
        dates_listOlists.append(dates_list)
        allbed_mean_listOlists.append(allbed_mean_list)
        allbed_lower_listOlists.append(allbed_lower_list)
        allbed_upper_listOlists.append(allbed_upper_list)
        ICUbed_mean_listOlists.append(ICUbed_mean_list)
        ICUbed_lower_listOlists.append(ICUbed_lower_list)
        ICUbed_upper_listOlists.append(ICUbed_upper_list)
        InvVen_mean_listOlists.append(InvVen_mean_list)
        InvVen_lower_listOlists.append(InvVen_lower_list)
        InvVen_upper_listOlists.append(InvVen_upper_list)
        deaths_mean_listOlists.append(deaths_mean_list)
        deaths_lower_listOlists.append(deaths_lower_list)
        deaths_upper_listOlists.append(deaths_upper_list)
        admis_mean_listOlists.append(admis_mean_list)
        admis_lower_listOlists.append(admis_lower_list)
        admis_upper_listOlists.append(admis_upper_list)
        newICU_mean_listOlists.append(newICU_mean_list)
        newICU_lower_listOlists.append(newICU_lower_list)
        newICU_upper_listOlists.append(newICU_upper_list)
        totdea_mean_listOlists.append(totdea_mean_list)
        totdea_lower_listOlists.append(totdea_lower_list)
        totdea_upper_listOlists.append(totdea_upper_list)
        deaths_mean_smoothed_listOlists.append(deaths_mean_smoothed_list)
        deaths_lower_smoothed_listOlists.append(deaths_lower_smoothed_list)
        deaths_upper_smoothed_listOlists.append(deaths_upper_smoothed_list)
        totdea_mean_smoothed_listOlists.append(totdea_mean_smoothed_list)
        totdea_lower_smoothed_listOlists.append(totdea_lower_smoothed_list)
        totdea_upper_smoothed_listOlists.append(totdea_upper_smoothed_list)
        total_tests_data_type_listOlists.append(total_tests_data_type_list)
        total_tests_listOlists.append(total_tests_list)
        confirmed_infections_listOlists.append(confirmed_infections_list)
        est_infections_mean_listOlists.append(est_infections_mean_list)
        est_infections_lower_listOlists.append(est_infections_lower_list)
        est_infections_upper_listOlists.append(est_infections_upper_list)

    # Results in error ValueError: Length of values does not match length of index
    imhe_hospitalizations_reduced['dates'] = dates_listOlists
    imhe_hospitalizations_reduced['allbed_mean'] = allbed_mean_listOlists
    imhe_hospitalizations_reduced['allbed_lower'] = allbed_lower_listOlists
    imhe_hospitalizations_reduced['allbed_upper'] = allbed_upper_listOlists
    imhe_hospitalizations_reduced['ICUbed_mean'] = ICUbed_mean_listOlists
    imhe_hospitalizations_reduced['ICUbed_lower'] = ICUbed_lower_listOlists
    imhe_hospitalizations_reduced['ICUbed_upper'] = ICUbed_upper_listOlists
    imhe_hospitalizations_reduced['InvVen_mean'] = InvVen_mean_listOlists
    imhe_hospitalizations_reduced['InvVen_lower'] = InvVen_lower_listOlists
    imhe_hospitalizations_reduced['InvVen_upper'] = InvVen_upper_listOlists
    imhe_hospitalizations_reduced['deaths_mean'] = deaths_mean_listOlists
    imhe_hospitalizations_reduced['deaths_lower'] = deaths_lower_listOlists
    imhe_hospitalizations_reduced['deaths_upper'] = deaths_upper_listOlists
    imhe_hospitalizations_reduced['admis_mean'] = admis_mean_listOlists
    imhe_hospitalizations_reduced['admis_lower'] = admis_lower_listOlists
    imhe_hospitalizations_reduced['admis_upper'] = admis_upper_listOlists
    imhe_hospitalizations_reduced['newICU_mean'] = newICU_mean_listOlists
    imhe_hospitalizations_reduced['newICU_lower'] = newICU_lower_listOlists
    imhe_hospitalizations_reduced['newICU_upper'] = newICU_upper_listOlists
    imhe_hospitalizations_reduced['totdea_mean'] = totdea_mean_listOlists
    imhe_hospitalizations_reduced['totdea_lower'] = totdea_lower_listOlists
    imhe_hospitalizations_reduced['totdea_upper'] = totdea_upper_listOlists
    imhe_hospitalizations_reduced['deaths_mean_smoothed'] = deaths_mean_smoothed_listOlists
    imhe_hospitalizations_reduced['deaths_lower_smoothed'] = deaths_lower_smoothed_listOlists
    imhe_hospitalizations_reduced['deaths_upper_smoothed'] = deaths_upper_smoothed_listOlists
    imhe_hospitalizations_reduced['totdea_mean_smoothed'] = totdea_mean_smoothed_listOlists
    imhe_hospitalizations_reduced['totdea_lower_smoothed'] = totdea_lower_smoothed_listOlists
    imhe_hospitalizations_reduced['totdea_upper_smoothed'] = totdea_upper_smoothed_listOlists
    imhe_hospitalizations_reduced['total_tests_data_type'] = total_tests_data_type_listOlists
    imhe_hospitalizations_reduced['total_tests'] = total_tests_listOlists
    imhe_hospitalizations_reduced['confirmed_infections'] = confirmed_infections_listOlists
    imhe_hospitalizations_reduced['est_infections_mean'] = est_infections_mean_listOlists
    imhe_hospitalizations_reduced['est_infections_lower'] = est_infections_lower_listOlists
    imhe_hospitalizations_reduced['est_infections_upper'] = est_infections_upper_listOlists

    # Reset indices before exporting
    imhe_summary_cleaned.reset_index(drop=True, inplace=True)
    imhe_hospitalizations_reduced.reset_index(drop=True, inplace=True)

    # Return dataframes
    return(imhe_summary_cleaned, imhe_hospitalizations_reduced)


def retrieve_Rt_live_data(state_data_df):
##  Estimated Effective Reproduction Rate $R_t$
##
## Kevin Systrom and Mike Krieger (co-founders of Instagram) and Tom Vladeck
## (owner of Gradient Metrics) put together a unaffiliated project to tracked
## modelled $R_t$ values for each state at http://rt.live/
##
## "$R_t$ represents the effective reproduction rate of the virus calculated for each
## locale. It lets us estimate how many secondary infections are likely to occur from
## a single infection in a specific area. Values over 1.0 mean we should expect more
## cases in that area, values under 1.0 mean we should expect fewer."

    # Grab the CSV file from rt.live and load into a Pandas Dataframe
    rt_csv = "https://d14wlfuexuxgcm.cloudfront.net/covid/rt.csv"
    rt_df = pd.read_csv(rt_csv, encoding='latin-1')

    # Build a state name to FIPS dictionary
    # Start by converting states dataframe into a dictionary
    FIPSdf = state_data_df[['FIPS','STNAME']].copy()
    FIPSd = FIPSdf.set_index('STNAME').T.to_dict('records')[0]

    # Determine range of dates
    DATEs = [datetime.strptime(x, '%Y-%m-%d').date() for x in rt_df['date'].to_list()]
    first_day = min(DATEs)
    last_day = max(DATEs)
    all_dates_list = list(datetime_range(first_day, last_day))
    all_dates_str = dates2strings(all_dates_list)

    # Create blank lists of lists
    FIPS_list = []
    states_list = []
    dates_listOlists = []
    Rt_mean_listOlists = []
    Rt_median_listOlists = []
    Rt_lower_80_listOlists = []
    Rt_upper_80_listOlists = []

    # For each state, extract the necessary data
    for key in code2name:
        # Grab the subset of data for this state
        rtstate_df = rt_df[rt_df['region'] == key][['date', 'mean', 'median', 'lower_80', 'upper_80']]

        # Convert the time series into lists in memory
        dates_str_list = rtstate_df['date'].to_list()

        # Convert the lists corresponding to each column into dictionaries for indexing
        Rt_mean_dict = dict(zip(dates_str_list, rtstate_df['mean'].tolist()))
        Rt_median_dict = dict(zip(dates_str_list, rtstate_df['median'].tolist()))
        Rt_lower_80_dict = dict(zip(dates_str_list, rtstate_df['lower_80'].tolist()))
        Rt_upper_80_dict = dict(zip(dates_str_list, rtstate_df['upper_80'].tolist()))

        #
        # Now brute force through all dates from first to last day and look for matches in each date,
        # filling the voids with NaN
        #
        # Start with blank lists for this State
        Rt_mean_list = []
        Rt_median_list = []
        Rt_lower_80_list = []
        Rt_upper_80_list = []

        for datekey in all_dates_str:
            try:
                Rt_mean_list.append(Rt_mean_dict[datekey])
                Rt_median_list.append(Rt_median_dict[datekey])
                Rt_lower_80_list.append(Rt_lower_80_dict[datekey])
                Rt_upper_80_list.append(Rt_upper_80_dict[datekey])
            except:
                Rt_mean_list.append(np.nan)
                Rt_median_list.append(np.nan)
                Rt_lower_80_list.append(np.nan)
                Rt_upper_80_list.append(np.nan)

        # Add variables lists to lists
        FIPS_list.append(FIPSd[code2name[key]])
        states_list.append(code2name[key])
        dates_listOlists.append(all_dates_list) # Add all dates
        Rt_mean_listOlists.append(Rt_mean_list)
        Rt_median_listOlists.append(Rt_median_list)
        Rt_lower_80_listOlists.append(Rt_lower_80_list)
        Rt_upper_80_listOlists.append(Rt_upper_80_list)

    # Build Pandas Dataframe for reduced data
    rt_reduced_df = pd.DataFrame()
    rt_reduced_df['FIPS'] = FIPS_list
    rt_reduced_df['state'] = states_list
    rt_reduced_df['dates'] = dates_listOlists
    rt_reduced_df['Rt_mean'] = Rt_mean_listOlists
    rt_reduced_df['Rt_median'] = Rt_median_listOlists
    rt_reduced_df['Rt_lower_80'] = Rt_lower_80_listOlists
    rt_reduced_df['Rt_upper_80'] = Rt_upper_80_listOlists

    return rt_reduced_df