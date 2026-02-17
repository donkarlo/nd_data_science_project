import numpy as np
from scipy.signal import find_peaks

from nd_utility.data.storage.kind.file.numpi.multi_valued import MultiValued as NpMultiValued
from nd_utility.os.file_system.file.file import File as OsFile
from nd_utility.os.file_system.path.file import File as FilePath


class TestPeriodInGpsData:
    def setup_method(self):
        file_path = FilePath(
            "/home/donkarlo/Dropbox/repo/robotic_lab_project/data/experiment/member/oldest/robotic_group/robotic_group/uav1/grouping/member/mind/memory/explicit/long_term/episodic/normal/gaussianed_quaternion_kinematic/time_position/time_position.npz"
        )
        os_file = OsFile.init_from_path(file_path)
        storage = NpMultiValued(os_file, False)
        storage.load()
        self._sequence = storage.get_ram()[:, 1:]  # shape: (time_steps, features)

    def test_clusterer(self):
        min_period = 10
        sequence = self._sequence

        if sequence.ndim != 2:
            raise ValueError("sequence must have shape (time_steps, features)")

        num_time_steps = sequence.shape[0]

        centered = sequence - sequence.mean(axis=0, keepdims=True)

        autocorr = self._multivariate_autocorrelation_fft(centered)

        peaks, _ = find_peaks(autocorr[min_period:])
        if len(peaks) == 0:
            raise RuntimeError("No period detected")

        period = int(peaks[0] + min_period)

        segments = []
        start_index = 0
        while start_index + period <= num_time_steps:
            segments.append(sequence[start_index:start_index + period])
            start_index += period

        print("Estimated period:", period)
        print("Number of cycles:", len(segments))
        print("One segment shape:", segments[0].shape)

    def _multivariate_autocorrelation_fft(self, centered_sequence: np.ndarray) -> np.ndarray:
        """
        Computes multivariate auto_correlation:
            R(lag) = mean_t sum_k x[t, k] * x[t+lag, k]
        in O(features * time_steps log time_steps) using FFT.
        """
        num_time_steps = centered_sequence.shape[0]
        num_features = centered_sequence.shape[1]

        fft_length = 1
        while fft_length < 2 * num_time_steps:
            fft_length *= 2

        autocorr_sum = np.zeros(num_time_steps, dtype=np.float64)

        feature_index = 0
        while feature_index < num_features:
            signal_1d = centered_sequence[:, feature_index].astype(np.float64, copy=False)
            spectrum = np.fft.rfft(signal_1d, n=fft_length)
            power_spectrum = spectrum * np.conj(spectrum)
            correlation_full = np.fft.irfft(power_spectrum, n=fft_length)
            autocorr_sum += correlation_full[:num_time_steps]
            feature_index += 1

        normalization = np.arange(num_time_steps, 0, -1, dtype=np.float64)
        autocorr = autocorr_sum / normalization
        return autocorr
