import numpy as np


class F0Predictor(object):
    def compute_f0(self, wav, p_len):
        """
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length]
        """
        pass

    def compute_f0_uv(self, wav, p_len):
        """
        input: wav:[signal_length]
               p_len:int
        output: f0:[signal_length//hop_length],uv:[signal_length//hop_length]
        """
        pass

    def interpolate_f0(self, f0):
        vuv_vector = np.zeros_like(f0, dtype=np.float32)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0

        nzindex = np.nonzero(f0)[0]
        data = f0[nzindex]
        nzindex = nzindex.astype(np.float32)
        time_org = self.hop_length / self.sampling_rate * nzindex
        time_frame = np.arange(f0.shape[0]) * self.hop_length / self.sampling_rate

        if data.shape[0] <= 0:
            return np.zeros(f0.shape[0], dtype=np.float32), vuv_vector

        if data.shape[0] == 1:
            return np.ones(f0.shape[0], dtype=np.float32) * f0[0], vuv_vector

        f0 = np.interp(time_frame, time_org, data, left=data[0], right=data[-1])

        return f0, vuv_vector
