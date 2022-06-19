import os
import librosa
import warnings
import numpy as np
from operator import itemgetter
warnings.simplefilter(action='ignore', category=FutureWarning)


def chuanHoa(arr):
    max_sc = max(arr)
    for index, x in enumerate(arr):
        arr[index] = arr[index]/max_sc
    return arr;    

solo_file = "audio/solo4.wav"

# Thực hiện việc trích rút đặc trưng
solo, sr = librosa.load(solo_file)
# for s in range(5):
#     start_sample = num_samples_per_segment * s
#     finish_sample = start_sample + num_samples_per_segment
sc = librosa.feature.spectral_centroid(solo, sr=sr, n_fft=2048, hop_length=512)[0]
spr = librosa.feature.spectral_rolloff(solo, sr=sr, n_fft=2048, hop_length=512)[0]
zr  = librosa.feature.zero_crossing_rate(solo, frame_length=2048, hop_length=512)[0]
rms = librosa.feature.rms(solo, frame_length=2048, hop_length=512)[0]
arr_sc = sc.tolist()
arr_sr = spr.tolist()
arr_zr = zr.tolist()
arr_rms = rms.tolist()

# Thực hiện chuẩn hóa dữ liệu ở miền thời gian và miền tần số
arr_sc = chuanHoa(arr_sc)
arr_sr = chuanHoa(arr_sr)
arr_zr = chuanHoa(arr_zr)
arr_rms = chuanHoa(arr_rms)


# Cộng các đặc trưng của từng frame
sc_string = ""
sr_string = ""
zr_string = ""
rms_string = ""
sc_string += str(sum(arr_sc))
sr_string += str(sum(arr_sr))
zr_string += str(sum(arr_zr))
rms_string += str(sum(arr_rms))


# Ghi các đặc trưng ra file
file = open("data/solo4.txt", "a+")
file.write(sc_string + "\n")
file.write(sr_string + "\n")
file.write(zr_string + "\n")
file.write(rms_string)
file.close()

                