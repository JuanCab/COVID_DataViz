##
## This is a set of functions for presenting COVID data to the user via a Dashboard
##

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime
import time
from IPython.core.display import display, HTML
from COVIDlib.collectors import days_since
import COVIDlib.data_IO as COVID_IO

##
## Define variables to be accessed within Dashboard functions
##

# Define titles/labels in English for John Hopkins and Mobility Data
var_descript = {'FIPS' : 'Federal Information Processing Standards State/County Number',
                'county' : 'County Name',
                'state' : 'State Name',
                'Lat': 'Latitude',
                'Long_' : 'Longitude',
                'dates' : 'Dates',
                'Confirmed' : 'Confirmed COVID Infections',
                'Deaths' : 'Confirmed and Probable COVID Deaths',
                'Recovered' : 'Confirmed and Probable COVID Recoveries',
                'Active' : 'Confirmed and Probable Active COVID Cases',
                'Incident_Rate' : 'Confirmed COVID Cases',  # As provided by John Hopkins
                'People_Tested' : 'Total tested for COVID',
                'People_Hospitalized' : 'Total hospitalized for COVID',
                'Mortality_Rate' : 'Mortality Rate',
                'Testing_Rate' : 'Total Tested for COVID',
                'Hospitalization_Rate' : 'Hospitalization Rate',
                'dConfirmed' : 'COVID Infection Rate',
                'd2Confirmed' : 'Change in COVID Infection Rate',
                'dDeaths' : 'COVID Death Rate',
                'd2Deaths' : 'Change in COVID Death Rate',
                'PopEst2019' : 'Estimated Population (July 1, 2019)',
                'PopChg2019' : 'Estimated Population Increase (2018-19)',
                'ConfirmedRate' : 'Confirmed COVID Infections (per 100,000 persons)', # As computed by us
                'DeathRate' : 'Confirmed COVID Deaths (per 100,000 people)', # As computed by us
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
                'Deaths' : 'Confirmed and Probable Deaths',
                'Recovered' : 'Confirmed and Probable Recoveries',
                'Active' : 'Confirmed and Probable Active Cases',
                'Incident_Rate' : 'Confirmed Cases (per 100,000 persons)',  # As provided by John Hopkins
                'People_Tested' : 'Total tested for COVID',
                'People_Hospitalized' : 'Total hospitalized for COVID',
                'Mortality_Rate' : 'Percent',
                'Testing_Rate' : 'Tested (per 100,000 people)',
                'Hospitalization_Rate' : 'Percent',
                'dConfirmed' : 'Infection Rate (Infections per day)',
                'd2Confirmed' : 'Change in Infection Rate (Infections/day per day)',
                'dDeaths' : 'New Deaths per day',
                'd2Deaths' : 'Change in Death Rate (Deaths/day per day)',
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


# Define functions to be used below
def derivative1D(x, y):
    """
    Compute forward difference estimate for the derivative of y with respect
    to x.  The x and y arrays are vectors with a single row each.

    The input and must be the same size.

    Note that we copy the first known derivative values into the zeroth column, since
    the derivatve for the first point is not a known value.
    """
    # Compute the numerator (y[i+1] - y[i]) for the entire row at once
    dy = y[1:] - y[0:-1]
    # Compute the denominator (x[i+1] - x[i]) for the entire row at once
    dx = x[1:] - x[0:-1]
    # Compute the derivatives for the entire row at once
    dydx = dy / dx
    # Get first column to horizontal stack with numpy array to pad the array
    first_col = dydx[0]
    return np.hstack((first_col, dydx))


def cleanJHdata(JH_dataframe):
    # Rename column to be consistent with other dataframes (simplifies plotting)
    JH_dataframe.rename(columns={ 'Dates': 'dates', 'State' : 'state', 'County': 'county'}, inplace = True)

    # If the maximum FIPS number is <100, this is state data, spend the extra time to process it to
    # get totals for the United States
    if (JH_dataframe['FIPS'].max() < 100):
        # Get sums of other columns
        tested_arr = np.sum(np.array(JH_dataframe['People_Tested'].to_list()), axis=0)
        # using nansum here because it treats NaN as 0
        hospitalized_arr = np.nansum( np.array(JH_dataframe['People_Hospitalized'].to_list()) , axis=0)

        # Sum these values for the entire US as well and process them into rates
        confirmed_arr = np.array(JH_dataframe['Confirmed'].to_list())
        deaths_arr = np.array(JH_dataframe['Deaths'].to_list())
        recovered_arr = np.array(JH_dataframe['Recovered'].to_list())
        confirmed_us_arr = np.sum(confirmed_arr, axis=0)
        deaths_us_arr = np.sum(deaths_arr, axis=0)
        recovered_us_arr = np.nansum(recovered_arr, axis=0)

        pop_us = JH_dataframe['PopEst2019'].sum() # Get sum as a scalar
        mortality = np.round_((deaths_us_arr/confirmed_us_arr)*100,2).tolist()
        hospitalized_rate_us = np.round_(((hospitalized_arr/pop_us)*100000),2).tolist()
        testing_rate_us = np.round_(((tested_arr/pop_us)*100000),2).tolist()

        # Compute derivatives for US
        dates_list = []
        for dat in JH_dataframe['dates'][0]:
            dates_list.append( days_since(dat) )
        dates_arr = np.array([dates_list][0])

        # Compute the derivatives (using forward derivative approach)
        dconfirmed_us_arr = derivative1D(dates_arr, confirmed_us_arr)
        ddeaths_us_arr = derivative1D(dates_arr, deaths_us_arr)
        # Compute the second derivatives (a bit hinky to use forward derivative again, but...)
        d2confirmed_us_arr = derivative1D(dates_arr, dconfirmed_us_arr)
        d2deaths_us_arr = derivative1D(dates_arr, ddeaths_us_arr)


        # Build a dataframe of US information
        us_df = pd.DataFrame({"FIPS" : 0,
                              "state" : "United States",
                              'Lat':  37.09,
                              'Long_' : -95.71,
                              'dates' : [JH_dataframe['dates'][0]],
                              'Confirmed' : [confirmed_us_arr.tolist()],
                              'Deaths' : [deaths_us_arr.tolist()],
                              'Recovered' : [recovered_us_arr.tolist()],
                              'Incident_Rate' : [[]],  # No data provided by John Hopkins
                              'People_Tested' : [tested_arr.tolist()],
                              'People_Hospitalized' : [hospitalized_arr.tolist()],
                              'Mortality_Rate' : [mortality],
                              'Testing_Rate' : [testing_rate_us],
                              'Hospitalization_Rate' : [hospitalized_rate_us],
                              'dConfirmed' : [dconfirmed_us_arr.tolist()],
                              'd2Confirmed' : [d2confirmed_us_arr.tolist()],
                              'dDeaths' : [ddeaths_us_arr.tolist()],
                              'd2Deaths' : [d2deaths_us_arr.tolist()],
                              'PopEst2019' : pop_us,
                              'PopChg2019' : JH_dataframe['PopChg2019'].sum() })

        # Append US to States dataframe
        JH_dataframe = JH_dataframe.append(us_df, ignore_index = True)

    # This function take a John Hopkins dataframe and adds confirmed infection and death rates per 100,000
    # people to the frame as columns (including US totals)
    confirmed_arr = np.array(JH_dataframe['Confirmed'].to_list())
    deaths_arr = np.array(JH_dataframe['Deaths'].to_list())
    daily_confirmed_arr = np.array(JH_dataframe['dConfirmed'].to_list())
    daily_deaths_arr = np.array(JH_dataframe['dDeaths'].to_list())
    daily_delta_confirmed_arr = np.array(JH_dataframe['d2Confirmed'].to_list())
    daily_delta_deaths_arr = np.array(JH_dataframe['d2Deaths'].to_list())
    recovered_arr = np.array(JH_dataframe['Recovered'].to_list())
    pop_arr = np.array(JH_dataframe['PopEst2019'].to_list())

    # Write the updated columns
    JH_dataframe['ConfirmedRate'] = np.round_(((confirmed_arr/pop_arr[:, None])*100000),2).tolist()
    JH_dataframe['DeathRate'] = np.round_(((deaths_arr/pop_arr[:, None])*100000),2).tolist()
    JH_dataframe['dConfirmedRate'] = np.round_(((daily_confirmed_arr/pop_arr[:, None])*100000),2).tolist()
    JH_dataframe['dDeathsRate'] = np.round_(((daily_deaths_arr/pop_arr[:, None])*100000),2).tolist()
    JH_dataframe['d2ConfirmedRate'] = np.round_(((daily_delta_confirmed_arr/pop_arr[:, None])*100000),2).tolist()
    JH_dataframe['d2DeathsRate'] = np.round_(((daily_delta_deaths_arr/pop_arr[:, None])*100000),2).tolist()
    JH_dataframe['Active'] = (confirmed_arr - deaths_arr - recovered_arr).tolist()

    # Computed confirmed new cases and deaths as fraction of population

    return JH_dataframe


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


def cleanAAPLdata(aapl_dataframe):
    # This function takes Apple mobility dataframes and converts the mobility to percent change from baseline
    # to be consistent with Google Mobility dataframes.

    mobility_arr = np.array(aapl_dataframe['driving_mobility'].to_list())
    mobility_percent = mobility_arr - 100
    aapl_dataframe['driving_mobility_Percent'] = np.round_(mobility_percent,2).tolist()

    return


def html_status(dataframe, fips, hospital_summary_df=None):
    ## Print an HTML statement of current status (Confirmed, Deaths, Recovered)
    ## based on Johns Hopkins dataframes (county or State)

    ## Check if FIPS input is reasonable
    if (type(fips) == int):
        fips = [fips]
    elif (type(fips) != list):
        raise ValueError('Input fips must be integer or list of integers')

    # Loop through the FIPS values
    for FIPS in fips:
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
        html_out = f"<h3>Status of {namestr} as of {last_day}</h3>"
        html_out += f"<table style='padding: 5px;'>"
        html_out += f"<tr><td style='text-align: right;vertical-align: top;'><b style='font-size: 140%;'>{last_infect_tot:,.0f} Total Cases</b><br/>({last_infectrate} per 100,000 people)</td>"
        html_out += f"<td style='text-align: left;vertical-align: top;'>"

        if (FIPS < 100): # Only list active and recovered for states
            html_out += f"<b><span style='color:#ff0000;font-size: 120%;'>{last_active_tot:,.0f} Active ({active_percent:.1f}%)</span> / <span style='color:rgb(0,128,20);font-size: 120%;'>{last_recovered_tot:,.0f} Recovered ({recovered_percent:.1f}%)</span> / <span style='font-size: 120%;'>{last_death_tot:,.0f} Dead ({dead_percent:.1f}%)</span></b><br/>"
        else:
            html_out += f"<b><span style='font-size: 120%;'>{last_death_tot:,.0f} Dead ({dead_percent:.1f}%)</span></b><br/>"
        html_out += f"<ul><li><b>{last_infect_change:,.0f} new infections</b> ({last_infect_change_rate:,.2f} per 100,000 people) in last day which is a change of {last_infect_change2:+,.0f} ({last_infect_change2_rate:+,.2f} per 100,000 people) from previous day.</li>"
        html_out += f"<li><b>{last_death_change:,.0f} new deaths</b> ({last_death_change_rate:,.2f} per 100,000 people) in last day which is a change of {last_death_change2:+,.0f} ({last_death_change2_rate:+,.2f} per 100,000 people) from previous day..</li></ul>"
        # If a hospitalization summary dataframe is provided, process it and produce HTML report
        if ((FIPS>0)&(FIPS < 100)&(hospital_summary_df is not None)):
            html_out += str(html_status_beds(hospital_summary_df, FIPS, display=False))
        html_out += f"</td></tr></table>"

        # Display HTML to screen
        display(HTML(html_out))

    return


def html_status_beds(dataframe, fips, display=True):
    ## Print an HTML statement of current status (Confirmed, Deaths, Recovered)
    ## based on Johns Hopkins dataframes (county or State)

    ## Check if FIPS input is reasonable
    if (type(fips) == int):
        fips = [fips]
    elif (type(fips) != list):
        raise ValueError('Input fips must be integer or list of integers')

    # Loop through the FIPS values for states
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
        if (display):
            html_out = f"<h3>Status of Hospital Beds in {namestr}</h3>"
            html_out += f"<table style='padding: 5px;'>"
            html_out += f"<tr><td style='text-align: left;vertical-align: top;'><b style='font-size: 140%;'>{num_icu_beds_used} of {num_icu_beds_total} ICU Beds ({percent_icu_used:,.2f}%) Being Used.</b><br/></td>"
            html_out += f"<tr><td style='text-align: left;vertical-align: top;'><b style='font-size: 140%;'>{num_reg_beds_used} of {num_reg_beds_total} Regular Beds ({percent_reg_used:,.2f}%) Being Used.</b><br/></td>"
            html_out += "</td></tr></table>"
            display(HTML(html_out))
            return
        else:
            # Build HTML to returm for inclusion in other report
            html_out = f"<b style='font-size: 120%;'>Status of Hospital Beds in {namestr}</b>"
            html_out += f"<ul><li>{num_icu_beds_used} of {num_icu_beds_total} ICU Beds ({percent_icu_used:,.2f}%) in use.</li>"
            html_out += f"<li>{num_reg_beds_used} of {num_reg_beds_total} Regular Beds ({percent_reg_used:,.2f}%) in use.</li></ul>"
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


def ts_plot(dataframe, colname, fips, connectdots=False, ylog=False, running_avg=0, fig=None, ax=None):
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
        ax.plot(dates, var, marker='o', markersize=3, linestyle=ls, label=labelstr)

    # Adjust y axis to be logarithmic if requested
    if (ylog):
        ax.set_yscale('log')

    # Add legend
    legend = ax.legend()


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