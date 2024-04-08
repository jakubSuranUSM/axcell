"""
This file serves as a helper file for the evaluation_sota.ipynb notebook.
It is used to download the orginal source files from arxiv, extract them,
and process them into a format that can be used by AxCell.

Direcitons:
    1. If you haven't already, download the SOTA dataset from the following link: https://github.com/jd-coderepos/sota/tree/master 
    2. Specify the PROJECT_ROOT directory to the root of the downloaded SOTA dataset. - Has to be same as in evaluate_sota.ipynb!!!
    3. Create a new directory where you want to download eprints and specify it to AXCELL_SOTA_ROOT variable. - Has to be same as in evaluate_sota.ipynb!!!
    4. Run the main method (you can specify if you want to download the eprints, extract the papers)
"""
import os
import requests
import time
import pandas as pd

from axcell.helpers.paper_extractor import PaperExtractor
from pathlib import Path
from joblib import delayed, Parallel
from tqdm import tqdm


PROJECT_ROOT = Path('/home/jakub.suran/netstore1/COS470/project/sota')
AXCELL_SOTA_ROOT = Path('/home/jakub.suran/netstore1/COS470/axcell') / 'data_sota'

VALIDATION_DATA_ROOT = PROJECT_ROOT / 'dataset' / 'validation'
SOURCES_PATH = AXCELL_SOTA_ROOT / 'sources'
PAPERS_PATH = AXCELL_SOTA_ROOT / 'papers'

download_papers = False
extract_papers = True


def get_eprint_link(paper):
    return f'http://export.arxiv.org/e-print/{paper.arxiv_id}'


def download_eprints(sota_leaderboards):
    links = sota_leaderboards.apply(get_eprint_link, axis=1)

    for i, link in tqdm(enumerate(links), "Downloading eprints...", total=len(links)):
        # play nice by arxiv - https://info.arxiv.org/help/bulk_data.html#harvest 
        if i % 4 == 0:
            time.sleep(1)
            
        response = requests.get(link)

        filename = link.split('/')[-1]
        path = SOURCES_PATH / filename
            
        with open(path, 'wb') as file:
            file.write(response.content)
        
    print(f"Downloaded {len(links)} eprints")


def main():
    if not os.path.exists(SOURCES_PATH):
        os.makedirs(SOURCES_PATH)
    
    validation_ids = [dir.name for dir in VALIDATION_DATA_ROOT.iterdir() if dir.is_dir()]

    sota_leaderboards = pd.DataFrame(validation_ids, columns=['arxiv_id'])

    if download_papers:
        download_eprints(sota_leaderboards)
            
    if extract_papers:
        extract = PaperExtractor(AXCELL_SOTA_ROOT)

        files = sorted([path for path in SOURCES_PATH.glob('**/*') if path.is_file()])

        print(f"Processing {len(files)} files")
        statuses = []
        for file in tqdm(files, "Processing files..."):
            statuses.append(extract(file)) 

        print("Results of paper extraction:")
        status_count = pd.Series(statuses).value_counts()
        print(status_count)

    extracted_papers = [dir.name for dir in PAPERS_PATH.iterdir() if dir.is_dir()]
    print(f"Successfullly extracted {len(extracted_papers)} papers")

    sota_leaderboards = sota_leaderboards[sota_leaderboards['arxiv_id'].isin(extracted_papers)]
    assert len(sota_leaderboards) == len(extracted_papers)


if __name__ == '__main__':
    main()