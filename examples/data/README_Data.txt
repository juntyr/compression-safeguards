### Helsinki_PR.csv ###

Contains the hourly precipitation (PR) in mm and the date and time (DateTime) in UTC from the 2024-04-02 00:00:00 to 2024-04-04 23:00:00
for Helsinki, Finland

Source: https://smear.avaa.csc.fi/download
Request:
	Station: SMEAR III Helsinki Kumpula
	Select variable category: Meteorology
	Date Range: 2024-04-02 -> 2024-04-04
	Processing Level: ANY
	Averaging: 1 hour
	Averaging Type: SUM

### Belem_PR.csv ###

Contains the hourly precipitation (PR) in mm and the date and time (DateTime) in UTC from the 2024-04-02 00:00:00 to 2024-04-04 23:00:00
for Belem, Brazil

Source: https://tempo.inmet.gov.br/TabelaEstacoes/A201
Request: 
	Date Range: 2024-04-02 -> 2024-04-04
	Table data hourly sums

### ERA5_PR.nc ###

Contains the hourly precipitation (tp) in m from the 2024-04-02 00:00:00 to 2024-04-04 23:00:00 UTC

Source: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download
Request: 
	Product type: Reanalysis
	Variable: Total precipitation
	Date Range: 2024-04-02 -> 2024-04-04
	Frequency: Hourly

### ERA5_UV.nc ###

Contains the U-component of wind (u) and V-component of wind (v) in m/s at 500 hPa at 2024-04-02 12:00 UTC

Source: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download
Request: 
	Product type: Reanalysis
	Variable: U-component of wind, V-component of wind
	Pressure level: 500 hPa
	Date Range: 2024-04-02 12:00

### ERA5_LH.nc ###

Contains the surface latent heat flux (slhf) in W/m2 and the lans/sea mask (lsm) at 2024-04-02 12:00 UTC

Source: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download
Request: 
	Product type: Reanalysis
	Variable: Surface latent heat flux, land/sea mask
	Date Range: 2024-04-02 12:00

### ERA5_Q.nc ###

Contains the specific humidity (q) in kg/kg at 2024-04-02 12:00 UTC

Source: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download
Request:
        Product type: Reanalysis
        Variable: Specific humidity
        Date Range: 2024-04-02 12:00
