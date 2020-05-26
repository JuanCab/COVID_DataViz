# COVID Visualization Project

This is our CSIS 340 project.  The idea was to take public data sources on
COVID-19 and combine them into a new dashboard that could be potentially useful
for local planners.

All that is here right now is a single Jupyter notebook that accesses the data.
It exploits Pandas, a library for handling tabular data.  Pandas can be really
awesome in some ways and really annoying in others, so I am not wedded to
sticking with Pandas, we could just store the data in numpy arrays or something
simple like that.

## Requirements
Right now, you need to install the following python extensions 
- gitpython
- pandas
- requests
If you installed Jupyter using the Anaconda (or miniconda) python distribution
then you can install all these using the following command on the command line:
`conda install gitpython pandas requests`

## Initialization

The Apple and Google mobility data are single files and are retrieved on the
fly. The John Hopkins data on the spread of the disease is a entire directory
structure full of files.  As such, I decided it made sense to pull the entire
directory off GitHub and use 'git pull' to keep it updated (since that would be
quick).

So to initialize the directory, in the same directory storing this notebook I
issued a `git clone https://github.com/CSSEGISandData/COVID-19.git` command to
pull the complete dataset from GitHub.  That created a `COVID-19` directory,
which I renamed `JH_Data`. With that done, maintaining that data shouldn't take
too long.