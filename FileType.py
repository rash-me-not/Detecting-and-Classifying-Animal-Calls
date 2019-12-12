from abc import ABC, abstractmethod
import os
import numpy as np

class FileType(ABC):

    def __init__(self, base_dir, converted_dir):
        self.base_dir = base_dir
        self.converted_dir = converted_dir
        pass

    @abstractmethod
    def get_combined_spec_label_dir(self):
        pass

    @abstractmethod
    def get_spec_dir(self):
        pass

    @abstractmethod
    def save_spec_label(self, s_db, begin_time, end_time, file):
        pass

    def save_combined(self, combined_file, s_db, label_matrix=None):
        combined = np.array((s_db, label_matrix))
        np.save(os.path.join(self.get_combined_spec_label_dir(), combined_file), combined)

    def create_save_folder(self, save_folder):
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

