import nibabel as nib

class SampleInfo:
    def __init__(self, patient_id, path):
        self.patient_id = patient_id
        self.path = path
        self._ct_data = None
        self._voxel = None
        self._segments = {}
        self.log = []

    def log_error(self, msg):
        self.log.append(msg)

    @property
    def ct_data(self):
        if self._ct_data is None:
            nifti = nib.load(self.path.fullCT)
            self._ct_data = nifti.get_fdata()
            self._voxel = tuple(round(v, 3) for v in nifti.header.get_zooms())
        return self._ct_data

    @property
    def ct_voxel(self):
        if self._voxel is None:
            _ = self.ct_data  # triggers loading
        return self._voxel

    def load_segments(self):
        for key, seg_path in self.path.segments.items():
            self._segments[key] = nib.load(seg_path).get_fdata()

    @property
    def lv(self): return self._segments.get("LV")
    @property
    def rv(self): return self._segments.get("RV")
    @property
    def la(self): return self._segments.get("LA")
    @property
    def ra(self): return self._segments.get("RA")
    @property
    def myo(self): return self._segments.get("MYO")