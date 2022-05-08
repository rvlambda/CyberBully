import joblib
import requests
import os

def get_model(model_path):
    
    try:
        with open(model_path, "rb") as mh:
            rf = joblib.load(mh)
    except:
        print("Cannot fetch model from local downloading from drive")
        if not 'RTA_severity model.pkl' in os.listdir('.'):
            # example url: "https://drive.google.com/u/1/uc?id=18IxYOI-whucBTZmt5qTvvYgjlxleaSqO&export=download&confirm=t"
            url = "https://drive.google.com/file/d/1hkif1hgzHIE7EvWN_Mfl-PfWJwmgtFtL/view?usp=sharing"
            r = requests.get(url, allow_redirects=True)
            open(r"CyberBully_DL_Model.h5", 'wb').write(r.content)
            del r
        with open(r"CyberBully_DL_Model.h5", "rb") as m:
            rf = joblib.load(m)
    return rf