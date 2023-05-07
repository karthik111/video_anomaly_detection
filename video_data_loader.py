import zipfile

with zipfile.ZipFile("C:\\Users\\karthik.venkat\\PycharmProjects\\video_anomaly_detection\\data\\Testing_Normal_Videos.zip") as zip:
    print("As table:")
    print(zip.printdir()) # display files and folders in tabular format
    print("\nAs list:")
    print(zip.namelist()) # list of files and folders
    print("\nAs list of objects:")
    print(zip.infolist()) # get files as ZipInfo objects

#for obj in zip.infolist():
    #if not obj.is_dir()