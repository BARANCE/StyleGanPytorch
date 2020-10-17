"""
ハイパーパラメータが記載されているjsonを読み出す
"""
from pathlib import Path
import json

class HyperParams:
    def __init__( self ):
        self.JSON_FILE_NAME = 'setting.json'
        self.file_dir = Path(__file__).parent
        
        json_path = self.file_dir.joinpath(self.JSON_FILE_NAME)
        
        try:
            with open(json_path, mode='r', encoding='utf-8') as fp:
                self.settings = json.load(fp)
        except FileNotFoundError:
            self.settings = None
    
    def get( self ) -> dict:
        return self.settings