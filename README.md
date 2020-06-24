## COVID Visualization Project
#### By Juan Cabanela, Luke Hebert, and Dio Lopez Vasquez

This is our CSIS 340 (Software Engineering) project for Summer (May-June) 2020.  The idea was to take public data sources on COVID-19 and combine them into a new dashboard that could be potentially useful for local planners.

### Requirements
In addition to Jupyter Notebooks or Jupyter Lab, you need to be running Python 3.7+ with the following python extensions for these notebooks to run.

- gitpython 
- pandas
- requests
- ipyleaflet
- ipywidgets
- ipympl

If you installed Jupyter using the Anaconda (or miniconda) python distribution
then you can install all these using the following command on the command
line: `conda install gitpython pandas requests ipyleaflet ipywidgets ipympl`

### Updating the Data Locally (Should be Done Daily)

Updates are probably most easily run from the command line by typing 

`python Collect_Data.py`  

(although the Jupyter Notebook `Collect_Data.ipynb` will accomplish the same).  This script pulls the entire US Census Bureau population files, Apple and Google mobility data on the fly. However, The John Hopkins data on the spread of the disease consist of entire directory structures full of files.  As such, it made sense to simply pull the GitHub repo to update the changes after the initial cloning.  The first time the script is run, it will create all the local directories necessary to store the data.

### Using The Dashboard

Once a day you should update the data using `Collect_Data.py`, after that, you can run the provided `Dashboard.ipynb` notebook in Jupyter to view the Dashboard locally.  

### Other Notebooks

- `Examples.ipynb`: Show a couple of the plotting and mapping commands in action in case you want to try to build your own custom dashboard.
- `Generate_Test_Data.ipynb`: Used to generate test data with known values in the same format as our data files in order to confirm the software was reading the data properly.
- `Test_DataIO_new.ipynb`: The actual notebook used to test our functions for reading the datafiles.
- `Comparing_DataIO.ipynb`: This was a notebook used to demonstrate the speed differences between the various functions/approaches to reading different datafile formats, it helped us settle on pickle files as the quickest way to preserve our structured pandas dataframes once we produced them with `Collect_Data.py`