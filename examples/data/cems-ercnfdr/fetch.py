import cdsapi

dataset = "cems-fire-historical-v1"
request = {
    "product_type": "reanalysis",
    "dataset_type": "consolidated_dataset",
    "system_version": ["4_1"],
    "year": ["2024"],
    "month": ["07"],
    "day": ["18"],
    "grid": "original_grid",
    "data_format": "grib",
    "variable": ["energy_release_component"],
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
