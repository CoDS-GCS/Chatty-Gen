import json
import os
import pathlib

class Tracer:
    def __init__(self, filename):
        self.filename = filename
        self.active_index = None
        self.data = None

        directory = pathlib.Path(self.filename).parent
        directory.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.filename):
            with open(self.filename, 'w'):
                pass  # Create an empty file if it doesn't exist


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