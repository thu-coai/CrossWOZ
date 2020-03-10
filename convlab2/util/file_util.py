from pathlib import Path
import zipfile
import json
from convlab2.util.allennlp_file_utils import cached_path as allennlp_cached_path


def cached_path(file_path, cached_dir=None):
    if not cached_dir:
        cached_dir = str(Path(Path.home() / '.convlab2') / "cache")

    return allennlp_cached_path(file_path, cached_dir)


def read_zipped_json(zip_path, filepath):
    archive = zipfile.ZipFile(zip_path, 'r')
    return json.load(archive.open(filepath))


def dump_json(content, filepath):
    json.dump(content, open(filepath, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


def write_zipped_json(zip_path, filepath):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(filepath)
