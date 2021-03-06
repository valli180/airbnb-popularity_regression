# Author: Valli Akella

"""
Download the csv file from the given url and save it in the filepath given

Usage: src/download_data.py --url=<url> --filepath=<filepath>

Options:
--url=<url>                 URL of the csv file
--filepath=<filepath>       Path where the file is stored in the local
Example: python3 src/download_data.py --url=http://data.insideairbnb.com/united-states/ny/new-york-city/2021-11-02/visualisations/listings.csv --filepath=data/raw/airbnb.csv
"""

from docopt import docopt 
import pandas as pd
import os

opt = docopt(__doc__)

def main(url, filepath):
    
    data = pd.read_csv(url, header=None)

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    data.to_csv(filepath, encoding="utf-8", index=False, header=False)

if __name__=="__main__":
    main(opt["--url"], opt["--filepath"])
    
    