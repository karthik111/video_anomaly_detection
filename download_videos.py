import requests

def get_file_name(string):
    # Find the index of the last occurrence of '/'
    last_slash_index = string.rfind('/')

    # Find the index of the first occurrence of '?'
    first_question_mark_index = string.find('?')

    # Extract the substring between the last '/' and the first '?'
    substring = string[last_slash_index + 1:first_question_mark_index]

    return (substring)  # Output: path/to/some/resource'

url = "https://www.dropbox.com/s/example/filename.zip?dl=1"
# Replace the "example" and "filename.zip" with your actual Dropbox folder path and file name

# change dl to 1 when copying urls below.

url = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABJtkTnNc8LcVTfH1gE_uFoa/Anomaly-Videos-Part-2.zip?dl=1"
url = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AAAbdSEUox64ZLgVAntr2WgSa/Anomaly-Videos-Part-2.zip?dl=1"
url ="https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AADEUCsLOCN_jHmmx7uFcUhHa/Training-Normal-Videos-Part-1.zip?dl=1"
url = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AAD5w76F_SZLdBgxVdvko-z5a/UCF_Crimes-Train-Test-Split.zip?dl=1"
url = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AACeDPUxpB6sY2jKgLGzaEdra/Testing_Normal_Videos.zip?dl=1"
url = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AAAgpsRNSHI_BtRnSCxxR7j9a/Anomaly-Videos-Part-3.zip?dl=1"
url = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABqY-3fJSmSMafFIlJXRE-9a/Anomaly-Videos-Part-4.zip?dl=1"
url = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AACyDI-0oRqqiqUcAulw_x5wa/Normal_Videos_for_Event_Recognition.zip?dl=1"
url = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AAAHZByMMGCVms4hhHZU2pMBa/Training-Normal-Videos-Part-2.zip?dl=1"
url = 'https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AADjSOQ-NLIsCVNWT0Mrhp5ca/Temporal_Anomaly_Annotation_for_Testing_Videos.txt?dl=1'

file_name = get_file_name(url)

r = requests.get(url, stream=True, verify=False)

with open(file_name, "wb") as f:
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)