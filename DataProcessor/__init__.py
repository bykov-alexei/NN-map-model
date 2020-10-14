import os
import json
import re
import numpy as np

from .process_timestamp import process_doy, process_tod, process_season
from .process_maps import default_processor

# Feature processors
# For some features specific fields are required
# E.g. doy requires times key in file
FEATURE_REGEX = {
    'doy': re.compile('doy\S*'),
    'tod': re.compile('tod\S*'),
    'season': re.compile('season_(?P<days>\d+)')
}

FEATURE_SOURCE = {
    'doy': 'times',
    'tod': 'times',
    'season': 'times',
}

FEATURE_PROCESSORS = {
    'default': lambda x: x,
    'doy': process_doy,       # day of year
    'tod': process_tod,       # time of day
    'season': process_season, # day in cycle (specified as 'season_27')
}

# Output processors
# To process output should be specified its key in file
OUTPUT_PROCESSORS = {
    'default': default_processor
}


class DataProcessor:
    
    def __init__(self, data_folder=None, data_filename=None, config_name='processor.json'):
        self.__config_path = os.path.join(data_folder, config_name)
        if os.path.exists(self.__config_path):
            f = open(os.path.join(self.__config_path), 'r')
            self.__config = json.load(f)
            f.close()
        else:
            self.__config = self.__generate_config(data_folder, data_filename)
            self.__save_config()
        self.data = {key:value for key, value in np.load(os.path.join(self.__config['data_folder'], self.__config['data_filename'])).items()}
            
    def __generate_config(self, data_folder: str, data_filename: str):
        assert data_folder is not None, 'Specify a folder with data'
        assert data_filename is not None, 'Specify filename of data file'
        config = {
            'data_folder': data_folder,
            'data_filename': data_filename,
            'processing_data': {
                'available_features': [],
                'available_outputs': [],
            }
        }
        return config
    
    def __get_match(self, string, regex):
        for key, value in regex.items():
            match = value.match(string)
            if match is not None:
                return key, match
        raise Exception(f"Such feature {string} is not supported. Check the correctness of feature name")
    
    def __save_config(self):
        f = open(self.__config_path, 'w')
        print(self.__config)
        json.dump(self.__config, f)
        f.close()
                    
    def __save_data(self):
        np.savez(os.path.join(self.__config['data_folder'], self.__config['data_filename']), **self.data)
       
    def __get_feature(self, feature, **kwargs):
        if feature not in self.__config['processing_data']['available_features']:
            key, match = self.__get_match(feature, FEATURE_REGEX)
            FEATURE_PROCESSORS[key](self.data, FEATURE_SOURCE[key], **kwargs)
            self.__save_data()
        return self.data[feature].reshape(len(self.data[feature]), -1)
    
    def get_training_data(self, features: list, output: list):
        X = []
        for feature in features:
            key, match = self.__get_match(feature, FEATURE_REGEX)
            X.append(self.__get_feature(feature, **match.groupdict()))
        X = np.concatenate(X, axis=1)
        return X
        
        
            
        
        