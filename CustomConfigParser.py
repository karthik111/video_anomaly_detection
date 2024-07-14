import configparser
import datetime

class CustomConfigParser:
    def __init__(self, config_file='config.ini', interpolation=configparser.BasicInterpolation()):
        self.config = configparser.ConfigParser(interpolation=interpolation)
        self.config.read(config_file)

    def get(self, section, option, fallback=None):
        try:
            return self.config.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

    def get_int(self, section, option, fallback=0):
        try:
            return self.config.getint(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback

    def get_float(self, section, option, fallback=0.0):
        try:
            return self.config.getfloat(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback

    def get_boolean(self, section, option, fallback=False):
        try:
            return self.config.getboolean(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback


class CurrentTimeInterpolation(configparser.BasicInterpolation):
    def before_get(self, parser, section, option, value, defaults):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        defaults['current_time'] = current_time
        return super().before_get(parser, section, option, value, defaults)