"""
This file serves as a helper file for the evaluation_sota.ipynb notebook.
It is used to download the orginal source file from arxiv, extract them,
and process them into a format that can be used by AxCell.

Direcitons:
    1. If you haven't already, download the SOTA dataset from the following link: https://github.com/jd-coderepos/sota/tree/master 
    2. Specify the PROJECT_ROOT directory to the root of the downloaded dataset.
"""
import tarfile
import json
import requests
import time
import pandas as pd

from axcell.data.paper_collection import PaperCollection
from axcell.helpers.results_extractor import ResultsExtractor
from axcell.helpers.datasets import read_tables_annotations
from axcell.helpers.paper_extractor import PaperExtractor
from axcell.helpers.evaluate import evaluate
from fastai.core import download_url
from pathlib import Path
from joblib import delayed, Parallel
from tqdm import tqdm


def get_eprint_link(paper):
    return f'http://export.arxiv.org/e-print/{paper.arxiv_id}'


def download_eprints(sota_leaderboards):
    links = sota_leaderboards.apply(get_eprint_link, axis=1)

    for i, link in enumerate(links):
        # play nice by arxiv - https://info.arxiv.org/help/bulk_data.html#harvest 
        if i % 4 == 0:
            time.sleep(1)
            
        response = requests.get(link)

        filename = link.split('/')[-1]
        path = Path("data_sota/sources") / filename
            
        with open(path, 'wb') as file:
            file.write(response.content)

        print(f"Downloaded {filename}")
        
    print(f"Downloaded {len(links)} eprints")

PROJECT_ROOT = Path('/home/jakub.suran/netstore1/COS470/project/sota')
VALIDATION_DATA_ROOT = PROJECT_ROOT / 'dataset' / 'validation'

validation_ids = [dir.name for dir in VALIDATION_DATA_ROOT.iterdir() if dir.is_dir()]

sota_leaderboards = pd.DataFrame(validation_ids, columns=['arxiv_id'])

tables = []
for arxiv_id in validation_ids:
    annotations_file = VALIDATION_DATA_ROOT / arxiv_id / 'annotations.json'
    table = {}
    table['index'] = 0
    table['records'] = []
    with open(annotations_file, 'r') as f:
        content = f.read()
        if content.strip() != 'unanswerable':
            content = content.replace('\'', '\"')
            annotations = json.loads(content)
            for leaderboard in annotations:
                leaderboard = leaderboard['LEADERBOARD']
                record = {
                    'task': leaderboard['Task'],
                    'dataset': leaderboard['Dataset'],
                    'metric': leaderboard['Metric'],
                    'value': leaderboard['Score']
                }
                table['records'].append(record)
    tables.append([table])
    
sota_leaderboards['tables'] = tables

download_eprints(sota_leaderboards)
        
AXCELL_SOTA_ROOT_PATH = Path('/home/jakub.suran/netstore1/COS470/axcell') / 'data_sota'
SOURCES_PATH = AXCELL_SOTA_ROOT_PATH / 'sources'
papers_extracted = True

if not papers_extracted:
    extract = PaperExtractor(AXCELL_SOTA_ROOT_PATH)

    # access extract from the global context to avoid serialization
    def extract_single(file): return extract(file)

    files = sorted([path for path in SOURCES_PATH.glob('**/*') if path.is_file()])

    print(f"Processing {len(files)} files")
    statuses = Parallel(backend='multiprocessing', n_jobs=-1)(delayed(extract_single)(file) for file in files)

    print("Results of paper extraction:")
    status_count = pd.Series(statuses).value_counts()
    print(status_count)

PAPERS_PATH = AXCELL_SOTA_ROOT_PATH / 'papers'
extracted_papers = [dir.name for dir in PAPERS_PATH.iterdir() if dir.is_dir()]
print(f"Successfullly extracted {len(extracted_papers)} papers")

sota_leaderboards = sota_leaderboards[sota_leaderboards['arxiv_id'].isin(extracted_papers)]
assert len(sota_leaderboards) == len(extracted_papers)

# V1_URL = 'https://github.com/paperswithcode/axcell/releases/download/v1.0/'
# MODELS_URL = V1_URL + 'models.tar.xz'
# MODELS_ARCHIVE = 'models.tar.xz'
# MODELS_PATH = Path('models')

# download_url(MODELS_URL, MODELS_ARCHIVE)
# with tarfile.open(MODELS_ARCHIVE, 'r:*') as archive:
#     archive.extractall()

# extract_results = ResultsExtractor(MODELS_PATH)

# papers = []
# our_taxonomy = set(extract_results.taxonomy.taxonomy)

# gold_records = []
# for _, paper in sota_leaderboards.iterrows():
#     for table in paper.tables:
#         for record in table['records']:
#             r = dict(record)
#             r['arxiv_id'] = paper.arxiv_id
#             tdm = (record['task'], record['dataset'], record['metric'])
#             if tdm in our_taxonomy:
#                 gold_records.append(r)
#                 papers.append(paper.arxiv_id)
# gold_records = pd.DataFrame(gold_records)
# papers = sorted(set(papers))    


# pc = PaperCollection.from_files(PAPERS_PATH)
# pc = PaperCollection([pc.get_by_id(p) for p in papers])


# def process_single(index):
#     extract_results = ResultsExtractor(MODELS_PATH)
#     return extract_results(pc[index])


# print(f"Processing {len(pc)} papers.")
# results = []
# for index in tqdm(range(len(pc)), "Processing papers..."):
#     results.append(process_single(index))
    
    
# predicted_records = []

# for paper, records in zip(pc, results):
#     r = records.copy()
#     r['arxiv_id'] = paper.arxiv_no_version
#     predicted_records.append(r)
# predicted_records = pd.concat(predicted_records)
# predicted_records.to_json('axcell-predictions-on-sota.json.xz', orient='records')

# print(f"Predicted records: {len(predicted_records)}")
# print(f"Gold records: {len(gold_records)}")

# predicted_records.to_csv('axcell-predictions-on-sota.csv')
# gold_records.to_csv('sota_gold_records.csv')

# eval = evaluate(predicted_records, gold_records)
# print(eval)
