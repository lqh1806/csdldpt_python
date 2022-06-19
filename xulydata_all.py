import os ##thư viện đọc ghi tệp
import librosa ##thư viện truy xuất đặc trưng dữ liệu âm thanh 
import warnings ##thư viện thông báo (dùng để bỏ thông báo khi trích xuất dữ liệu)
import numpy as np ##thư viện làm việc với mảng
from operator import itemgetter ## operator là thư viện để 
warnings.simplefilter(action='ignore', category=FutureWarning) #ignore thông báo warning không cần thiết khi trích xuất

#hàm chuẩn hóa
#lấy giá trị từng phần tử trong mảng chia phẩn tử lớn nhất của mảng
def chuanHoa(arr):
    max_sc = max(arr)
    for index, x in enumerate(arr):
        arr[index] = arr[index]/max_sc
    return arr;

for path, dirs, files in os.walk("/Users/lequo/Desktop/CSDLDPT/csdldpt_python/audio/"):
    for f in files:
        filename = os.path.join(path, f)
        print(str(os.path.basename(filename)))
        # Thực hiện việc trích rút đặc trưng
        solo, sr = librosa.load(filename) ##solo là mẫu audio, sr là tỉ lệ lấy mẫu (=22050)

        ##Tốc độ lấy mẫu là 1s lấy 22050 lần, 1 frame dài 2048 mẫu thì để lấy hết mẫu trong frame đó hết 93ms
        ##các hàm trích rút đặc trưng dưới đây trả về 1 mảng giá trị ứng với mỗi frame
        sc = librosa.feature.spectral_centroid(solo, sr=sr, n_fft=2048, hop_length=512)[0] ##solo là audio, sr là tỷ lệ lấy mẫu, n_fft là số bin trên 1 fft, hop_length là framing (lấy mẫu các phần frame liên tiếp chồng lên nhau)
        spr = librosa.feature.spectral_rolloff(solo, sr=sr, n_fft=2048, hop_length=512)[0]
        ##Vì 2 cái đặc trưng trên là miền tần số nên cần tham số đầu vào là n_fft để sử dụng biến đổi Fourier chuyển từ miền thời gian sang miền tần số(n_fft = frame_length luôn để sau khi biến đổi cả miền tg lẫn ts đều có kích thước frame = nhau tiện cho so sánh)
        zr  = librosa.feature.zero_crossing_rate(solo, frame_length=2048, hop_length=512)[0]
        rms = librosa.feature.rms(solo, frame_length=2048, hop_length=512)[0]
        arr_sc = sc.tolist()
        arr_sr = spr.tolist()
        arr_zr = zr.tolist()
        arr_rms = rms.tolist()
        arr_sc = chuanHoa(arr_sc)
        arr_sr = chuanHoa(arr_sr)
        arr_zr = chuanHoa(arr_zr)
        arr_rms = chuanHoa(arr_rms)
        sc_string = ""
        sr_string = ""
        zr_string = ""
        rms_string = ""
        sc_string += str(sum(arr_sc))
        sr_string += str(sum(arr_sr))
        zr_string += str(sum(arr_zr))
        rms_string += str(sum(arr_rms))
        file_name_txt = str(os.path.basename(filename)).rsplit('.', 1)[0]
        file_name_txt += ".txt"
        file = open("data/" + file_name_txt, "a+")
        file.write(sc_string + "\n")
        file.write(sr_string + "\n")
        file.write(zr_string + "\n")
        file.write(rms_string)
        file.close()
