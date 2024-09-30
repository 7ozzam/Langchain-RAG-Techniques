import os
from dotenv import load_dotenv

class EnvManager:
    def __init__(self):
        # Load environment variables from the .env file
        load_dotenv()

        # Dynamically load environment variables prefixed with CONFIG__
        self.config = self._load_config()

    def _load_config(self):
        # Collect environment variables starting with CONFIG__
        config_dict = {
            key.split("__")[1].lower(): os.getenv(key)
            for key in os.environ
            if key.startswith("CONFIG__")
        }
        
        # Convert the port to an integer, or use 3301 as the default
        if 'port' in config_dict:
            config_dict['port'] = int(config_dict['port'])

        return config_dict

    def get(self, key, default=None):
        # Fetch a configuration value, with an optional default
        return self.config.get(key, default)

    def all(self):
        # Return the entire config as a dictionary
        return self.config

# Usage example
config = EnvManager()

# Access a specific config value
host = config.get("host")
port = config.get("port")

# Access the entire config
# print(config.all())
