import json
import os

class Tracer:
    _instance = None

    def __new__(cls, filename):
        if cls._instance is None:
            cls._instance = super(Tracer, cls).__new__(cls)
            cls._instance.filename = filename
            cls._instance.active_index = None
            cls._instance.data = None

            if not os.path.exists(cls._instance.filename):
                with open(cls._instance.filename, 'w'):
                    pass  # Create an empty file if it doesn't exist

        return cls._instance

    def add_data(self, index, key, value):
        if self.active_index is None:
            self.active_index = index
        
        if index != self.active_index:
            # file write the previous one
            self.save_to_file()
        
        if self.data is None:
            self.data = {
                key: value
            }
            return

        self.data[key] = value

    def save_to_file(self):
        with open(self.filename, 'a') as file:
            dump = {self.active_index: self.data}
            json.dump(dump, file)
            file.write('\n')
        self.active_index = None
        self.data = None