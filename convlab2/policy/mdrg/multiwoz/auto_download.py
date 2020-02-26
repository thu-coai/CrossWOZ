import json
import os
import random
import zipfile
from convlab2.util.file_util import cached_path


def auto_download():
    model_path = os.path.join(os.path.dirname(__file__),  'model')
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    db_path = os.path.join(os.path.dirname(__file__), 'db')
    root_path = os.path.dirname(__file__)

    urls = {model_path: 'https://tatk-data.s3-ap-northeast-1.amazonaws.com/mdrg_model.zip',
            data_path: 'https://tatk-data.s3-ap-northeast-1.amazonaws.com/mdrg_data.zip',
            db_path: 'https://tatk-data.s3-ap-northeast-1.amazonaws.com/mdrg_db.zip'}

    for path in [model_path, data_path, db_path]:
        if not os.path.exists(path):
            file_url = urls[path]
            print('Downloading from: ', file_url)
            archive_file = cached_path(file_url)
            print('Extracting...')
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(root_path)