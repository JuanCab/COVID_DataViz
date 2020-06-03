# COVID Visualization Project
# 
This is our CSIS 340 project.  The idea was to take public data sources on
COVID-19 and combine them into a new dashboard that could be potentially
useful for local planners.

All that is here right now is a single Jupyter notebook that accesses the
data. It exploits Pandas, a library for handling tabular data.  Pandas can be
really awesome in some ways and really annoying in others, so I am not wedded
to sticking with Pandas, we could just store the data in numpy arrays or
something simple like that.

## Requirements
Right now, you need to install the following python extensions

- gitpython 
- pandas
- requests

If you installed Jupyter using the Anaconda (or miniconda) python distribution
then you can install all these using the following command on the command
line: `conda install gitpython pandas requests`

## Initialization

The Apple and Google mobility data are single files and are retrieved on the
fly.

The John Hopkins and NY Times data on the spread of the disease consist of
entire directory structures full of files.  As such, I decided it made sense
to pull their entire directories off GitHub and use 'git pull' to keep it
updated (since that would be quick).

1. **Download the John Hopkins data**: In the same directory storing this
README file and the Jupyter notebooks:

   - `git clone https://github.com/CSSEGISandData/COVID-19.git` - This creates
   a `COVID-19` directory, which I renamed `JH_Data`.

2. **Download the New York Times data**: In the same directory storing this
README file and the Jupyter notebooks:

   - `git clone https://github.com/nytimes/covid-19-data` - This creates a
   `covid-19-data` directory, which I renamed `NYT_Data`.

3. **Create output data directory**: Create the `our_data` directory for
storing data.