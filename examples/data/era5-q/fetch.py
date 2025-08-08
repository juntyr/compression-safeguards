import cdsapi

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ["specific_humidity"],
    "year": ["2024"],
    "month": ["04"],
    "day": ["02"],
    "time": ["12:00"],
    "pressure_level": ["850"],
    "data_format": "netcdf",
    "download_format": "unarchived",
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
