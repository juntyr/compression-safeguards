import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ["2m_dewpoint_temperature", "2m_temperature"],
    "year": ["2025"],
    "month": ["07"],
    "day": ["31"],
    "time": ["12:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
