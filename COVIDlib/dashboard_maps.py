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

# Example of contents of state GEOJSON data structure (WHICH DOES WORK)
# {'type': 'FeatureCollection',
#  'features': [{'type': 'Feature',
#    'id': 'AL',
#    'properties': {'name': 'Alabama'},
#    'geometry': {'type': 'Polygon',
#     'coordinates': [[[-87.359296, 35.00118],
#       [-85.606675, 34.984749],
#       [-85.431413, 34.124869],
#       [-85.184951, 32.859696],
#       [-85.069935, 32.580372],
#       [-84.960397, 32.421541],

# Example of the 2018 US Census Bureau derived county boundaries (WHICH DOESN'T WORK WITHOUT MODIFICATION)
# {'type': 'FeatureCollection',
#  'name': 'cb_2018_us_county_20m',
#  'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::4269'}},
#  'features': [{'type': 'Feature',
#    'properties': {'STATEFP': '37',
#     'COUNTYFP': '017',
#     'COUNTYNS': '01026336',
#     'AFFGEOID': '0500000US37017',
#     'GEOID': '37017',
#     'NAME': 'Bladen',
#     'LSAD': '06',
#     'ALAND': 2265887723,
#     'AWATER': 33010866},
#    'geometry': {'type': 'Polygon',
#     'coordinates': [[[-78.901998, 34.835268],
#       [-78.799597, 34.850857],
#       [-78.516123, 34.845919],
#       [-78.494705, 34.856182],
#       [-78.390748, 34.749463],
#       [-78.391978, 34.741265],
#       [-78.374363, 34.700722],
#       [-78.254441, 34.553595],
#       [-78.255468, 34.508614],

# If we just loop through the list of dictionaries in 'features' and create an 'id' entry equal to the
# 'GEOID' of that entry's 'properties' we can get plotting to work.  This is now handled by load_county_geojson().

##
## This first block of code is basically are geared around reformatting data or listing the
## variables we will consider plotting.
##

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


def BuildMapVarDict():
    # Construct this dictionary from the John Hopkins and Mobility Data dictionaries
    JHDict = COVID_Dash.BuildJHVarDict()
    MobileDict = COVID_Dash.BuildMobilityVarDict()
    var_dict = JHDict.copy()
    var_dict.update(MobileDict)

    return var_dict


##
## The Code below are the routines that really do the map making and control
##


def scrub(datadict):
    # Purge nan values from dictionary, replacing them with minmaximumimum value for plotting purposes
    if (any(np.isnan(val) for val in datadict.values())):
        minval = np.nanmin(np.array(list(datadict.values())))

        # Loop throught and purge the nan values
        clean_dict = datadict.copy()
        for k in datadict:
            if np.isnan(datadict[k]):
                clean_dict[k] = minval
        return clean_dict
    else:
        return datadict


def set_cm_limits(datadict):
    # Determine maximum and minimum values for the color map, starting at enclosing 90% of the data
    # but working out as needed.
    vals=np.array(list(datadict.values()))
    minval = np.nanpercentile(vals, 5) # Min at 5th Percentile (ignoring nan values)
    maxval = np.nanpercentile(vals, 95) # Max out at 95th percentile
    # Move out boundaries if needed
    if (minval == maxval):
        # Move out the boundaries
        minval = np.nanpercentile(vals, 1) # Min at 1st Percentile (avoid wierd outliers like corrections to death counts)
        maxval = np.nanpercentile(vals, 99) # Max out at 99th Percentile (avoids returning zero for daily death counts for counties)
        if (minval == maxval):
            # Move out the boundaries if still not far enough
            minval = np.nanpercentile(vals, 0) # Min
            maxval = np.nanpercentile(vals, 100) # Max

    return (minval, maxval)


def update_state_overlay(feature, **kwargs):
    global this_state_colname, state_data_dict, MapsVDict, state_overlay

    # Get data value for this state and set the overlay to indicate it
    state = feature['properties']['name']
    units = MapsVDict[this_state_colname]['valdescript']
    form = MapsVDict[this_state_colname]['format']
    val = state_data_dict[feature['id']]
    if not np.isnan(val):
        if ('d' in form):
            val = int(val)
        struct = "<div style='text-align: center;'><b>{0}</b><br/>{1:"+form+"} {2}</div>"
        state_overlay.value = struct.format(state, val, units)
    else: # Handle nan values
        struct = "<div style='text-align: center;'><b>{0}</b><br/><em>(unknown)</em> {1}</div>"
        state_overlay.value = struct.format(state, units)


def build_us_genericmap():
    # This function is called by either the county/state map building functions and
    # loads all the data needed for either as global variables
    global MapsVDict, geojson_states, geojson_cnty, mapcenter, mapzoom

    # Load Maps Variable Descriptions for use in overlays if it doesn't exist
    try:
        MapsVDict
    except NameError:
        MapsVDict = BuildMapVarDict()

    # Load county/state boundary data geojsons
    geojson_cnty = load_county_geojson()
    geojson_states = load_state_geojson()

    # Define map center and zoom
    mapcenter = [38.0, -93.0]
    mapzoom = 4.2

    return


def BuildLocationDict(dataframe):
    # Construct dictionary of placenames by FIPS number for use by county overlay
    loc_df = dataframe[['FIPS', 'county', 'state']].set_index('FIPS').copy()
    loc_df['Location'] = loc_df['county'].map('{:} ('.format)+loc_df['state'].map('{:})'.format)
    loc_df.drop(columns=['county', 'state'], inplace=True) 
    loc_dict = loc_df['Location'].to_dict()
    return loc_dict


def BuildLegendDict(minval, maxval, cmap):
    # Construct Legend Dictionary
    nsteps = 5
    step = (maxval-minval)/(nsteps-1)
    legendDict = {}
    for i in range(nsteps):
        val = minval+i*step
        valstr = f"{val:,.1f}"
        legendDict[valstr] = cmap(val)
    return legendDict


def build_us_statesmap(dataframe, colname):
    global geojson_states, this_state_colname, state_overlay, state_data_dict
    global state_layer
    global MapsVDict, geojson_states, mapcenter, mapzoom, loc_dict

    # This function builds a US Choropleth Map (but doesn't display it) for the state-level
    # data provided.

    # Load data needed to build either kind of map
    build_us_genericmap()

    # ipyleaflet requires a dictionary for the choro_data field/the variable to be visualized,
    # so convert the Pandas data series into the appropriate dictionary setting keys to postal
    # codes used in geojson_states
    state_data_dict = get_state_dict(dataframe, colname)

    # Determine range of values for colormap, then define colormap (need to also pass
    # max/min values to Choropleth builder or they are ignored). Set up legend dictionary
    # to show this range.
    (minval, maxval) = set_cm_limits(state_data_dict)
    cmap = linear.YlOrRd_06.scale(minval, maxval)
    legendDict = BuildLegendDict(minval, maxval, cmap)

    # Creating the map
    states_map = lf.Map(center = mapcenter, zoom = mapzoom)

    # Draw a functional states layer
    state_layer = lf.Choropleth(geo_data=geojson_states,
                                 choro_data=scrub(state_data_dict),
                                 key_on='id',
                                 # Below here is some formatting/coloring from the documentation
                                 colormap=cmap,
                                 value_min=minval,
                                 value_max=maxval,
                                 border_color='black',
                                 hover_style={'fillOpacity': 1.0, 'dashArray': '0'},
                                 style={'fillOpacity': 0.6, 'dashArray': '5, 5'} )
    states_map.add_layer(state_layer)

    # Display a legend
    state_legend = lf.LegendControl(legendDict, name="Legend", position="bottomleft")
    states_map.add_control(state_legend)

    # Display data in overlay
    this_state_colname = colname
    state_overlay = widgets.HTML("Hover over States for Details")
    state_control = lf.WidgetControl(widget=state_overlay, position='topright')
    states_map.add_control(state_control)
    state_layer.on_hover(update_state_overlay)

    return(states_map, state_legend, state_overlay)


def update_us_statesmap(dataframe, colname, thismap, thislegend, thisoverlay):
    global geojson_states, this_state_colname, state_data_dict, state_overlay

    # This function updates an existing US State-level Choropleth map

    # Load the new data and determine the new colormap limits and set up legend dictionary
    # to show this range.
    state_data_dict = get_state_dict(dataframe, colname)
    (minval, maxval) = set_cm_limits(state_data_dict)
    cmap = linear.YlOrRd_06.scale(minval, maxval)
    legendDict = BuildLegendDict(minval, maxval, cmap)

    # Assign updated legend dictionary
    thislegend.legends = legendDict

    # Draw a functional states layer
    state_layer_update = lf.Choropleth(geo_data=geojson_states,
                                       choro_data=scrub(state_data_dict),
                                       key_on='id',
                                       # Below here is some formatting/coloring from the documentation
                                       colormap=cmap,
                                       value_min=minval,
                                       value_max=maxval,
                                       border_color='black',
                                       hover_style={'fillOpacity': 1.0, 'dashArray': '0'},
                                       style={'fillOpacity': 0.6, 'dashArray': '5, 5'} )

    # Replace existing Choropleth layer (which is always the second layer with new layer
    state_layer = thismap.layers[1]
    thismap.substitute_layer(state_layer, state_layer_update)

    # Update column name used by state overlay to look up values
    this_state_colname = colname
    thisoverlay.value = "Hover over States for Details"
    state_overlay = thisoverlay
    state_layer_update.on_hover(update_state_overlay)

    return


def update_cnty_overlay(feature, **kwargs):
    global this_cnty_colname, county_data_dict, loc_dict, MapsVDict, cnty_overlay

    # Get data value for this county and set the overlay to indicate it
    FIPS = int(feature['id'])
    location = loc_dict[FIPS]
    units = MapsVDict[this_cnty_colname]['valdescript']
    form = MapsVDict[this_cnty_colname]['format']
    val = county_data_dict[feature['id']]
    if not np.isnan(val):
        if ('d' in form):
            val = int(val)
        struct = "<div style='text-align: center;'><b>{0}</b><br/>{1:"+form+"} {2}</div>"
        cnty_overlay.value = struct.format(location, val, units)
    else: # Handle nan values
        struct = "<div style='text-align: center;'><b>{0}</b><br/><em>(unknown)</em> {1}</div>"
        cnty_overlay.value = struct.format(location, units)


def build_us_cntymap(dataframe, colname):
    global geojson_cnty, this_cnty_colname, cnty_overlay, county_data_dict
    global cnty_layer, cnty_legend, cnty_control
    global MapsVDict, geojson_cnty, mapcenter, mapzoom, loc_dict

    # This function builds a US Choropleth Map (but doesn't display it) for the county-level
    # data provided.

    # Load data needed to build either kind of map
    build_us_genericmap()

    # Build location dictionary if doesn't yet exist
    try:
        loc_dict
    except NameError:
        loc_dict = BuildLocationDict(dataframe)

    # ipyleaflet requires a dictionary for the choro_data field/the variable to be visualized,
    # so convert the Pandas data series into the appropriate dictionary setting keys to postal
    # codes used in geojson_states
    county_data_dict = get_cnty_dict(dataframe, colname)

    # Determine range of values for colormap, then define colormap (need to also pass
    # max/min values to Choropleth builder or they are ignored). Set up legend dictionary
    # to show this range.
    (minval, maxval) = set_cm_limits(county_data_dict)
    cmap = linear.YlOrRd_06.scale(minval, maxval)
    legendDict = BuildLegendDict(minval, maxval, cmap)

    # Creating the map
    cnty_map = lf.Map(center = mapcenter, zoom = mapzoom)

    # Draw a functional counties layer
    cnty_layer = lf.Choropleth(geo_data=geojson_cnty,
                                 choro_data=scrub(county_data_dict),
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
    cnty_legend = lf.LegendControl(legendDict, name="Legend", position="bottomleft")
    cnty_map.add_control(cnty_legend)

    # Display data in overlay
    this_cnty_colname = colname
    cnty_overlay = widgets.HTML("Hover over Location for Details")
    cnty_control = lf.WidgetControl(widget=cnty_overlay, position='topright')
    cnty_map.add_control(cnty_control)
    cnty_layer.on_hover(update_cnty_overlay)

    return(cnty_map, cnty_legend, cnty_overlay)


def update_us_cntymap(dataframe, colname, thismap, thislegend, thisoverlay):
    global geojson_cnty, this_cnty_colname, county_data_dict, cnty_overlay, loc_dict

    # This function updates an existing US County-level Choropleth map

    # Build location dictionary if doesn't yet exist
    try:
        loc_dict
    except NameError:
        loc_dict = BuildLocationDict(dataframe)

    # Load the new data and determine the new colormap limits
    county_data_dict = get_cnty_dict(dataframe, colname)
    (minval, maxval) = set_cm_limits(county_data_dict)
    cmap = linear.YlOrRd_06.scale(minval, maxval)
    legendDict = BuildLegendDict(minval, maxval, cmap)

    # Assign updated legend dictionary
    thislegend.legends = legendDict

    # Draw a functional counties layer
    cnty_layer_update = lf.Choropleth(geo_data=geojson_cnty,
                                      choro_data=scrub(county_data_dict),
                                      key_on='id',
                                      # Below here is some formatting/coloring from the documentation
                                      colormap=cmap,
                                      value_min=minval,
                                      value_max=maxval,
                                      border_color='black',
                                      hover_style={'fillOpacity': 1.0, 'dashArray': '0'},
                                      style={'fillOpacity': 0.6, 'dashArray': '5, 5'} )

    # Replace existing Choropleth layer (which is always the second layer with new layer
    cnty_layer = thismap.layers[1]
    thismap.substitute_layer(cnty_layer, cnty_layer_update)
    cnty_layer = cnty_layer_update

    # Update column name used by state overlay to look up values
    this_cnty_colname = colname
    thisoverlay.value = "Hover over Location for Details"
    cnty_overlay = thisoverlay
    cnty_layer_update.on_hover(update_cnty_overlay)

    return