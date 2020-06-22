##
## This is a set of functions for presenting COVID data to the user as a Map
##

import os
import pandas as pd
import numpy as np
import ipywidgets as widgets

# imports for ipyleaflet
import ipyleaflet as lf
import json
from branca.colormap import linear

# Grab some Dashboard functions
import COVIDlib.dashboard_IO as COVID_Dash

##
## Define useful functions for building Maps of Data
##

# Load variable descriptions
JHVarDict = COVID_Dash.BuildJHVarDict()

def load_state_geojson(filename = './ipyleaflet_json/us-states.json'):
    # Define this function for loading the state boundaries JSON (grabbed from the GitHub repo for iPyLeaflet)
    with open(filename, 'r') as f:
        # Had to set the encoding for this import to work on Mac (should still work on PC)
        return json.load(f)


def load_county_geojson_raw(filename = './ipyleaflet_json/cb_2018_us_county_20m.geojson'):
    # Define this function for loading the county boundaries JSON file which was originally
    # produced by grabbing the US Census Bureau's 20m resolution boundary files and processing them through
    # https://mygeodata.cloud
    with open(filename, 'r') as f:
        # Had to set the encoding for this import to work on Mac (should still work on PC)
        return json.load(f)


def load_county_geojson(filename = './ipyleaflet_json/cb_2018_us_county_20m.geojson'):
    # This function is a wrapper around load_countygeojson_full() which will allow reformatting of
    # the raw output into a properly 'id' formatted version.  Could eventually add filtering of the
    # county geojson data in case we want to eventually filter by state[s], for now, it just filters
    # out US Territories

    geodict = load_county_geojson_raw(filename)

    # Add an 'id' entry to the features dictionaries for each entry that pulls from the GEO_ID the FIPS number
    features_list=[]
    for entry in geodict['features']:
        entry['id'] = entry['properties']['GEOID']
        # Purge any entries with FIPS > 60000 (these are territories which we don't have data for)
        if (int(entry['id']) < 60000):
                features_list.append(entry)
    geodict['features'] = features_list

    return geodict


def format_cnty_dict(dataframe):
    # ipyleaflet expects dictionaries of data, keyed by the ID, so converting county data to be right format
    data_dict = {}
    for key in dataframe.to_dict():
        #newKey = f'0500000US{key:05d}'
        newKey = f'{key:05d}'
        data_dict[newKey] = dataframe[key]
    return data_dict


def get_state_dict(dataframe, colname):
    # Pull the one column of data (the last day for time series) and return it as a dictionary indexed by postal code
    state_data = dataframe[dataframe['state'] != 'United States'].set_index('postal').copy()
    # Replace column of lists with just last values
    state_data[colname] = np.array(state_data[colname].to_list())[:,-1].tolist()
    return state_data[colname].to_dict()


def get_cnty_dict(dataframe, colname):
    # Pull the one column of data and return it as a dictionary indexed by FIPS
    county_data = dataframe.set_index('FIPS').copy()
    # Replace column of lists with just last values
    county_data[colname] = np.array(county_data[colname].to_list())[:,-1].tolist()
    return format_cnty_dict(county_data[colname])


def update_state_overlay(feature, **kwargs):
    global this_state_colname, state_data_dict, overlay, JHVarDict, state_overlay

    # Get data value for this state and set the overlay to indicate it
    val = state_data_dict[feature['id']]
    units = JHVarDict[this_state_colname]['valdescript']
    state = feature['properties']['name']

    state_overlay.value = f"<div style='text-align: center;'><b>{state}</b><br/>{val:.2f} {units}</div>"


def build_us_statesmap(dataframe, colname):
    global geojson_states, this_state_colname, state_overlay, state_data_dict

    # This function builds a US Choropleth Map (but doesn't display it) for the state-level
    # data provided.

    # Load state boundary data
    geojson_states = load_state_geojson()

    # ipyleaflet requires a dictionary for the choro_data field/the variable to be visualized,
    # so convert the Pandas data series into the appropriate dictionary setting keys to postal
    # codes used in geojson_states
    state_data_dict = get_state_dict(dataframe, colname)

    # Define map center and zoom
    center = [38.0, -93.0]
    zoom = 3.9

    # Determine range of values for colormap, then define colormap (need to also pass
    # max/min values to Choropleth builder or they are ignored)
    minval = state_data_dict[min(state_data_dict, key=state_data_dict.get)]
    maxval = state_data_dict[max(state_data_dict, key=state_data_dict.get)]
    cmap=linear.YlOrRd_04.scale(minval, maxval)

    # Break range into steps to build colormap legend dictionary
    nsteps = 5
    step = (maxval-minval)/(nsteps-1)
    legendDict = {}
    for i in range(nsteps):
        val = minval+i*step
        valstr = f"{val:,.1f}"
        legendDict[valstr] = cmap(val)

    # Creating the map
    states_map = lf.Map(center = center, zoom = zoom)

    # Draw a functional states layer
    states_layer = lf.Choropleth(geo_data=geojson_states,
                                 choro_data=state_data_dict,
                                 key_on='id',
                                 # Below here is some formatting/coloring from the documentation
                                 colormap=cmap,
                                 value_min=minval,
                                 value_max=maxval,
                                 border_color='black',
                                 hover_style={'fillOpacity': 1.0, 'dashArray': '0'},
                                 style={'fillOpacity': 0.6, 'dashArray': '5, 5'} )
    states_map.add_layer(states_layer)

    # Display a legend
    legend = lf.LegendControl(legendDict, name="Legend", position="bottomleft")
    states_map.add_control(legend)

    # Display data in overlay
    this_state_colname = colname
    state_overlay = widgets.HTML("Hover over States for Details")
    state_control = lf.WidgetControl(widget=state_overlay, position='topright')
    states_map.add_control(state_control)
    states_layer.on_hover(update_state_overlay)

    return(states_map)


def update_us_statesmap(dataframe, colname, statemap):
    global geojson_states, this_state_colname, state_data_dict

    # This function updates an existing US State-level Choropleth map

    # Load the new data and determine the new colormap limits
    state_data_dict = get_state_dict(dataframe, colname)
    minval = state_data_dict[min(state_data_dict, key=state_data_dict.get)]
    maxval = state_data_dict[max(state_data_dict, key=state_data_dict.get)]
    cmap=linear.YlOrRd_04.scale(minval, maxval)

    # Break range into steps to build colormap legend dictionary
    nsteps = 5
    step = (maxval-minval)/(nsteps-1)
    legendDict = {}
    for i in range(nsteps):
        val = minval+i*step
        valstr = f"{val:,.1f}"
        legendDict[valstr] = cmap(val)

    # Remove old legent and add new legend
    legend = lf.LegendControl(legendDict, name="Legend", position="bottomleft")
    statemap.clear_controls()
    statemap.add_control(legend)

    # Draw a functional states layer
    states_layer = lf.Choropleth(geo_data=geojson_states,
                                 choro_data=state_data_dict,
                                 key_on='id',
                                 # Below here is some formatting/coloring from the documentation
                                 colormap=cmap,
                                 value_min=minval,
                                 value_max=maxval,
                                 border_color='black',
                                 hover_style={'fillOpacity': 1.0, 'dashArray': '0'},
                                 style={'fillOpacity': 0.6, 'dashArray': '5, 5'} )

    # Clear existing layers and add new one
    statemap.clear_layers()
    statemap.add_layer(states_layer)

    # Update column name used by state overlay to look up values
    this_state_colname = colname

    return


def update_cnty_overlay(feature, **kwargs):
    global this_cnty_colname, county_data_dict, loc_dict, JHVarDict, cnty_overlay

    # Get data value for this county and set the overlay to indicate it
    val = county_data_dict[feature['id']]
    units = JHVarDict[this_cnty_colname]['valdescript']
    FIPS = int(feature['id'])
    location = loc_dict[FIPS]

    cnty_overlay.value = f"<div style='text-align: center;'><b>{location}</b><br/>{val:.2f} {units}</div>"


def build_us_cntymap(dataframe, colname):
    global geojson_cnty, this_cnty_colname, cnty_overlay, county_data_dict, loc_dict

    # This function builds a US Choropleth Map (but doesn't display it) for the county-level
    # data provided.

    # Construct dictionary of placenames by FIPS number for use by overlay
    loc_df = dataframe[['FIPS', 'county', 'state']].set_index('FIPS').copy()
    loc_df['Location'] = loc_df['county'].map('{:} ('.format)+loc_df['state'].map('{:})'.format)
    loc_df.drop(columns=['county', 'state'], inplace=True) 
    loc_dict = loc_df['Location'].to_dict()

    # Load the 2018 county bounary data
    geojson_cnty = load_county_geojson()

    # ipyleaflet requires a dictionary for the choro_data field/the variable to be visualized,
    # so convert the Pandas data series into the appropriate dictionary setting keys to postal
    # codes used in geojson_states
    county_data_dict = get_cnty_dict(dataframe, colname)

    # Define map center and zoom
    center = [38.0, -93.0]
    zoom = 3.9

    # Determine range of values for colormap, then define colormap (need to also pass
    # max/min values to Choropleth builder or they are ignored)
    vals=np.array(list(county_data_dict.values()))
    minval = np.percentile(vals, 5) # Min at 5th Percentile
    maxval = np.percentile(vals, 95) # Max out at 95th percentile
    cmap=linear.YlOrRd_04.scale(minval, maxval)

    # Break range into steps to build colormap legend dictionary
    nsteps = 5
    step = (maxval-minval)/(nsteps-1)
    legendDict = {}
    for i in range(nsteps):
        val = minval+i*step
        valstr = f"{val:,.1f}"
        legendDict[valstr] = cmap(val)

    # Creating the map
    cnty_map = lf.Map(center = center, zoom = zoom)

    # Draw a functional counties layer
    cnty_layer = lf.Choropleth(geo_data=geojson_cnty,
                                 choro_data=county_data_dict,
                                 key_on='id',
                                 # Below here is some formatting/coloring from the documentation
                                 colormap=cmap,
                                 value_min=minval,
                                 value_max=maxval,
                                 border_color='black',
                                 hover_style={'fillOpacity': 1.0, 'dashArray': '0'},
                                 style={'fillOpacity': 0.6, 'dashArray': '5, 5'} )
    cnty_map.add_layer(cnty_layer)

    # Display a legend
    legend = lf.LegendControl(legendDict, name="Legend", position="bottomleft")
    cnty_map.add_control(legend)

    # Display data in overlay
    this_cnty_colname = colname
    cnty_overlay = widgets.HTML("Hover over Location for Details")
    cnty_control = lf.WidgetControl(widget=cnty_overlay, position='topright')
    cnty_map.add_control(cnty_control)
    cnty_layer.on_hover(update_cnty_overlay)

    return(cnty_map)


def update_us_cntymap(dataframe, colname, cntymap):
    global geojson_cnty, this_cnty_colname, county_data_dict

    # This function updates an existing US County-level Choropleth map

    # Load the new data and determine the new colormap limits
    county_data_dict = get_cnty_dict(dataframe, colname)
    vals=np.array(list(county_data_dict.values()))
    minval = np.percentile(vals, 5) # Min at 5th Percentile
    maxval = np.percentile(vals, 95) # Max out at 95th percentile
    cmap=linear.YlOrRd_04.scale(minval, maxval)

    # Break range into steps to build colormap legend dictionary
    nsteps = 5
    step = (maxval-minval)/(nsteps-1)
    legendDict = {}
    for i in range(nsteps):
        val = minval+i*step
        valstr = f"{val:,.1f}"
        legendDict[valstr] = cmap(val)

    # Remove old legent and add new legend
    legend = lf.LegendControl(legendDict, name="Legend", position="bottomleft")
    cntymap.clear_controls()
    cntymap.add_control(legend)

    # Draw a functional counties layer
    cnty_layer = lf.Choropleth(geo_data=geojson_cnty,
                                 choro_data=county_data_dict,
                                 key_on='id',
                                 # Below here is some formatting/coloring from the documentation
                                 colormap=cmap,
                                 value_min=minval,
                                 value_max=maxval,
                                 border_color='black',
                                 hover_style={'fillOpacity': 1.0, 'dashArray': '0'},
                                 style={'fillOpacity': 0.6, 'dashArray': '5, 5'} )

    # Clear existing layers and add new one
    cntymap.clear_layers()
    cntymap.add_layer(cnty_layer)

    # Update column name used by state overlay to look up values
    this_cnty_colname = colname

    return