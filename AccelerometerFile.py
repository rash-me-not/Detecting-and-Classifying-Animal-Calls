from FileType import FileType
import os
import numpy as np

class AccelerometerFile(FileType):

    def __init__(self, base_dir, converted_dir, ch):
        super().__init__(base_dir, converted_dir)
        self.ch = ch

    def get_combined_spec_label_dir(self):
        combined_spec_label_dir = os.path.join(self.base_dir, "combined", "combined_spec_label_acc_ch_" + str(self.ch))
        self.create_save_folder(combined_spec_label_dir)
        return combined_spec_label_dir


    def get_spec_dir(self):
        spec_dir = os.path.join(self.converted_dir, "spec_acc_ch_" + str(self.ch))
        self.create_save_folder(spec_dir)
        return spec_dir

    def save_spec_label(self, s_db, begin_time, end_time, file, label_matrix):
        # Saving the spec file if there is at least one call present in the 6 sec segment
        spec_file = file + '_' + str(begin_time) + 'sto' + str(end_time) + 's'
        spec_path = self.get_spec_dir()
        np.save(os.path.join(spec_path, spec_file), s_db)
        combined_file = spec_file + 'SPEC_LAB'
        self.save_combined(combined_file, s_db)


