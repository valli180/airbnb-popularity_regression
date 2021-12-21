# Author: Valli Akella

"""
Download the csv file from the given url and save it in the filepath given

Usage: src/download_data.py --url=<url> --filepath=<filepath>

Options:
--url=<url>                 URL of the csv file
--filepath=<filepath>       Path where the file is stored in the local

"""

from docopt import docopt 
import pandas as pd
import os

opt = docopt(__doc__)

def main(url, filepath):
    
    data = pd.read_csv(url, header=None)

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    data.to_csv(filepath, encoding="utf-8")

if __name__=="__main__":
    main(opt["--url"], opt["--filepath"])
    
    