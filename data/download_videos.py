import requests

url = "https://www.dropbox.com/s/example/filename.zip?dl=1"
# Replace the "example" and "filename.zip" with your actual Dropbox folder path and file name

url = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABJtkTnNc8LcVTfH1gE_uFoa/Anomaly-Videos-Part-2.zip?dl=1"
url = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AAAbdSEUox64ZLgVAntr2WgSa/Anomaly-Videos-Part-2.zip?dl=1"
url ="https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AADEUCsLOCN_jHmmx7uFcUhHa/Training-Normal-Videos-Part-1.zip?dl=1"
file_name = "Training-Normal-Videos-Part-1.zip"

r = requests.get(url, stream=True, verify=False)

with open(file_name, "wb") as f:
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)