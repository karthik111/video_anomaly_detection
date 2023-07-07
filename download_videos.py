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

url = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABJtkTnNc8LcVTfH1gE_uFoa/Anomaly-Videos-Part-2.zip?dl=1"
url = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AAAbdSEUox64ZLgVAntr2WgSa/Anomaly-Videos-Part-2.zip?dl=1"
url ="https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AADEUCsLOCN_jHmmx7uFcUhHa/Training-Normal-Videos-Part-1.zip?dl=1"
url = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AAD5w76F_SZLdBgxVdvko-z5a/UCF_Crimes-Train-Test-Split.zip?dl=1"

file_name = get_file_name(url)

r = requests.get(url, stream=True, verify=False)

with open(file_name, "wb") as f:
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)