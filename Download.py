# download the data from Kaggle  url = 'https://www.kaggle.com/c/spaceship-titanic/download-all'
import os
import zipfile

def download_data():
    # Make directory
    os.makedirs('./data', exist_ok=True)
    # Download the data
    # do bash command to download the data
    os.system('kaggle competitions download -c spaceship-titanic')
    # unzip the data
    zip_ref = zipfile.ZipFile('spaceship-titanic.zip', 'r')
    zip_ref.extractall('./data')
    zip_ref.close()
    # remove the zip file
    os.remove('spaceship-titanic.zip')


if __name__ == '__main__':
    download_data()
