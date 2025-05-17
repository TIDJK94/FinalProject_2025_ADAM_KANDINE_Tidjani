import os
import urllib.request
import zipfile
import pandas as pd
import json
from pathlib import Path




def load_data(clean=True):
    DATA_FOLDER = "data_final_project"
    PATH_DATA_EXTRACTED = "data_final_project/KuaiRec 2.0/data"
    DATA_FOLDER = Path(DATA_FOLDER)
    PATH_DATA_EXTRACTED = Path(PATH_DATA_EXTRACTED)

    if not os.path.isdir(DATA_FOLDER):
        # Download the ZIP file
        url = 'https://drive.usercontent.google.com/download?id=1qe5hOSBxzIuxBb1G_Ih5X-O65QElollE&export=download&confirm=t&uuid=b2002093-cc6e-4bd5-be47-9603f0b33470'
        zip_path = "KuaiRec.zip"
        urllib.request.urlretrieve(url, zip_path)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_FOLDER)

        #  Remove the ZIP file
        os.remove(zip_path)

    small_matrix = pd.read_csv(PATH_DATA_EXTRACTED / "small_matrix.csv")
    big_matrix = pd.read_csv(PATH_DATA_EXTRACTED / "big_matrix.csv")
    item_categories = pd.read_csv(PATH_DATA_EXTRACTED / "item_categories.csv")
    item_features = pd.read_csv(PATH_DATA_EXTRACTED / "item_daily_features.csv")
    social_network = pd.read_csv(PATH_DATA_EXTRACTED / "social_network.csv")
    user_features = pd.read_csv(PATH_DATA_EXTRACTED / "user_features.csv")
    captions = pd.read_csv(PATH_DATA_EXTRACTED / "kuairec_caption_category.csv", engine='python', on_bad_lines='skip',encoding='utf-8')

    print("Data loaded.")

    if not clean:
        return small_matrix, big_matrix, item_categories, item_features, social_network, user_features, captions
    
    # preprocess data

    small_matrix = small_matrix.dropna()
    small_matrix = small_matrix.drop_duplicates()
    small_matrix = small_matrix[small_matrix["timestamp"] >= 0]

    big_matrix = big_matrix.dropna()
    big_matrix = big_matrix.drop_duplicates()
    big_matrix = big_matrix[big_matrix["timestamp"] >= 0]


    item_categories = item_categories.dropna()
    item_categories = item_categories.drop_duplicates()
    item_categories['feat'] = item_categories['feat'].apply(lambda x: json.loads(x))

    user_features = user_features[["user_active_degree", "follow_user_num"] + [f"onehot_feat{x}" for x in range(18)]]
    user_features.fillna(-1, inplace=True)

    print("Data cleaned.")


    return small_matrix, big_matrix, item_categories, item_features, social_network, user_features, captions