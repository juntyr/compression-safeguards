import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ["surface_latent_heat_flux", "land_sea_mask"],
    "year": ["2024"],
    "month": ["04"],
    "day": ["02"],
    "time": ["12:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
