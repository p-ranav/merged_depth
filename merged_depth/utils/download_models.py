'''
Guide for usage:
In your terminal, run the command:
python download_gdrive.py GoogleFileID /path/for/this/file/to/download/file.type
Credited to 
https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
author: https://stackoverflow.com/users/1475331/user115202
'''

import requests
from tqdm import tqdm
import os
import zipfile

def download_file_from_google_drive(id, destination):
  def get_confirm_token(response):
    for key, value in response.cookies.items():
      if key.startswith('download_warning'):
        return value

    return None

  def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
      with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as bar:
        for chunk in response.iter_content(CHUNK_SIZE):
          if chunk:  # filter out keep-alive new chunks
            f.write(chunk)
            bar.update(CHUNK_SIZE)

  URL = "https://docs.google.com/uc?export=download"

  session = requests.Session()

  response = session.get(URL, params = { 'id' : id }, stream = True)
  token = get_confirm_token(response)

  if token:
    params = { 'id' : id, 'confirm' : token }
    response = session.get(URL, params = params, stream = True)

  save_response_content(response, destination)

def download_models(gdrive_fileid, output_path):
  dirname = os.path.dirname(output_path)
  if not os.path.exists(dirname):
    os.makedirs(dirname)

  if not os.path.exists(output_path):
    download_file_from_google_drive(gdrive_fileid, output_path)

def extract_models(file):
  dirname = os.path.dirname(file)
  with zipfile.ZipFile(file, 'r') as zip_ref:
    zip_ref.extractall(dirname)

def main():
  pretrained_models = os.path.join("pretrained", "pretrained.zip")
  download_models(gdrive_fileid="1SXZP3VEacwIGH6HP_czFKZJCzxX5rYF9", output_path=pretrained_models)
  extract_models(pretrained_models)

if __name__ == '__main__':
  main()
