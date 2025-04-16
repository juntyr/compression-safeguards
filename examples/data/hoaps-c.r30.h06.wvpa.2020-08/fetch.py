import cdsapi

dataset = "satellite-total-column-water-vapour-ocean"
request = {
    "origin": "c3s",
    "climate_data_record_type": "icdr",
    "temporal_aggregation": "6_hourly",
    "year": ["2020"],
    "month": ["08"],
    "variable": "all",
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
