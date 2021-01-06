import os

from utils import io
from constants import config__PATH

class Config:
    def __init__(self):
        super(Config, self).__init__()

    @staticmethod
    def yaml_config(config=None):
        file_config = {}
        if config is None and os.path.isfile(config__PATH):
            config = config__PATH

        if config is not None:
            try:
                file_config = io.read_config_file(config)
            except Exception as e:
                raise ValueError(
                    'Failed to read configuration file "{}". '
                    'Error:{}.'
                        .format(config, e)
                )

        list_config = []
        dict_config = {}
        if 'default' in file_config.keys():
            list_config.extend(file_config['default'])

        if 'current_config' in file_config.keys():
            current_config = file_config['current_config']

            for key in file_config.keys():
                if key == current_config:
                    list_config.extend(file_config[current_config])
                    break

        if list_config:
            for item in list_config:
                for key, value in item.items():
                    dict_config[key] = value
            return dict_config
        return None
