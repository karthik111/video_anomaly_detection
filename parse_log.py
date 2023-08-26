import pandas as pd
def search_line_in_file(file_paths, search_text):
    df = pd.DataFrame(columns=["file_name", "line_num", 'text', 'video'])
    for file_name in file_paths:
        try:
            with open(file_name, 'r') as file:
                for line_number, line in enumerate(file, start=1):
                    if search_text in line:
                        df.loc[len(df)] = [file_name, line_number, line.strip(), line[line.rindex(":")+2:].strip()]
                        #return line_number, line.strip()
        except FileNotFoundError:
            return None, None

    return df

# Provide the path to the file and the text you want to search for
file_paths = ['.\log_file.log',
            ".\log_file_19-07-23.log",
             ".\log_file_30-07-23.log",
             ]

search_text = 'ERROR - An error occurred with file'

df1 = search_line_in_file(file_paths, search_text)

