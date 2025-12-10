import gzip
import shutil
from urllib.request import urlopen

# URL from http://sciviscontest-staging.ieeevis.org/2004/data.html
base_url = "https://cloud.sdsc.edu/v1/AUTH_sciviscontest/2004/isabeldata/"

with urlopen(base_url + "Pf48.bin.gz") as response_gz:
    with gzip.open(response_gz, "rb") as response:
        with open("Pf48.bin2", "wb") as file:
            shutil.copyfileobj(response, file)

with urlopen(base_url + "Uf48.bin.gz") as response_gz:
    with gzip.open(response_gz, "rb") as response:
        with open("Uf48.bin2", "wb") as file:
            shutil.copyfileobj(response, file)
