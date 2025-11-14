
# Hurricane Isabel WRF Model Data

This directory contains several datasets of compressed "bricks" of floats. Each file represents a single atmospheric variable for one timestep. The file naming convention is **VAR**f**NN**.bin.gz, where **VAR** is the variable name (CLOUD, PRECIP, etc.), and **NN** is the timestep (1 per hour).

There is also and single 2D file (HGTdata.bin.gz) containing the height field of the surface topography.

The Weather Research and Forecasting (WRF) Model is developed by NCAR and its partners <http://wrf-model.org>, and the simulation of Hurricane Isabel and data processing are performed by Wei Wang, Cindy Bruyere, and Bill Kuo of Mesoscale and Microscale Meteorology Division, NCAR, and the SCD visualization group.

If you use the data set, please provide the following attribution:

The authors will like to thank Bill Kuo, Wei Wang, Cindy Bruyere, Tim Scheitlin, and Don Middleton of the U.S. National Center for Atmospheric Research (NCAR), and the U.S. National Science Foundation (NSF) for providing the Weather Research and Forecasting (WRF) Model simulation data of Hurricane Isabel.

A shorter attribution is:

Hurricane Isabel data produced by the Weather Research and Forecast (WRF) model, courtesy of NCAR, and the U.S. National Science Foundation (NSF).

### File Compression

Each file is a compressed (gzip) brick of floats. Each file will expand to 100000000 bytes when uncompressed. The byte order is Big Endian.

### Variable Dimensions for each file

* XDIM (horizontal longitude) = 500
* YDIM (horizontal latitude) = 500
* ZDIM (vertical) = 100 equally spaced levels
* TDIM (time) = 1 for each file with a total of 48 timesteps for each variable.

The height data file (HGTdata.bin), contains a 2D file of floats with dimensions 500x500x1.

### Array ordering

The data can be accessed as follows:

```c
fread((float *) data, TDIM*ZDIM*XDIM*YDIM*sizeof(float),1, fp);
```

Where XDIM=YDIM=500, ZDIM=100, TDIM=1, and fp is a FILE pointer (fopen) to the input dataset.

A single element that has an index of x,y,z, and t, is accessed as:

```c
data[x + y*XDIM + z*XDIM*YDIM + t*XDIM*YDIM*ZDIM];
```

Note: If you are reading a file with a single timestep, t=0. Alternatively, you can concatenate (using the Unix cat command) several timesteps together, and access the data with the above formula where 0<=t<=Number_of_timesteps-1.

### Data Coordinates

The coordinates for the data are:

* x (Longitude) coordinate runs from 83 W to 62 W (approx)
* y (Latitude) coordinate runs from 23.7 N to 41.7 N (approx)
* z (Vertical) coordinate runs from .035 KM to 19.835 KM (100 equally spaced levels with delta=.2 KM)

### Missing Values

Land values, where there is no valid atmospheric data, are designated with the value of 1.0e35.

### Variable Descriptions

| Variable | Description | Min/Max | Units |
| --- | --- | --- | --- |
| QCLOUD | Cloud Water | 0.00000/0.00332 | kg/kg |
| QGRAUP | Graupel | 0.00000/0.01638 | kg/kg |
| QICE | Cloud Ice | 0.00000/0.00099 | kg/kg |
| QRAIN | Rain | 0.00000/0.01132 | kg/kg |
| QSNOW | Snow | 0.00000/0.00135 | kg/kg |
| QVAPOR | Water Vapor | 0.00000/0.02368 | kg/kg |
| CLOUD | Total cloud (QICE+QCLOUD) | 0.00000/0.00332 | kg/kg |
| PRECIP | Total Precipitation (QGRAUP+QRAIN+QSNOW) | 0.00000/0.01672 | kg/kg |
| P   | Pressure; weight of the atmosphere above a grid point | -5471.85791/3225.42578 | Pascals |
| TC  | Temperature | -83.00402/31.51576 | Degrees Celsius |
| U   | X wind component; west-east wind component in model coordinate, postive means winds blow from west to east | -79.47297/85.17703 | m s-1 |
| V   | Y wind component; south-north wind component in model coordinate, postive means winds blow from south to north | -76.03391/82.95293 | m s-1 |
| W   | Z wind component; vertical wind component in model coordinate, positive means upward motion | -9.06026/28.61434 | m s-1 |

Note: All of the 'Q' variables represent moisture in the atmosphere in one form or another. For example, rain water differs from cloud water in terms of their particle sizes. Rain water has larger drop sizes so they can fall through the air, while cloud water has smaller drop sizes so they stay in the air. Rain water differs from snow and graupel in terms of their densities, and so does cloud water from cloud ice.

### Suggested Links for more Hurricane Information

* <http://cimss.ssec.wisc.edu/tropic/tropic.html>
* <http://www.nhc.noaa.gov/archive/2003/>
