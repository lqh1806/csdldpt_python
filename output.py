import os
import librosa
import warnings
import numpy as np


# Đọc file và xử lý
f = open("data/solo10.txt")
line = f.readline()
ar_sc_input = line.split(" ")
line = f.readline()
ar_sr_input = line.split(" ")
line = f.readline()
ar_zr_input = line.split(" ")
line = f.readline()
ar_rms_input = line.split(" ")
fileInputName = os.path.basename(f.name)
f.close()

knn_arr = np.empty(shape=[0,2])

for path, dirs, files in os.walk("/Users/lequo/Desktop/CSDLDPT/csdldpt_python/data/"):
    for f in files:
        filename = os.path.join(path, f)
        with open(filename, "r") as myFile:
            # Đọc lần lượt 4 dòng của file txt để lưu vào các biến dùng để so sánh
            line = myFile.readline()
            ar_sc_output = line.split(" ")
            line = myFile.readline()
            ar_sr_output = line.split(" ")
            line = myFile.readline()
            ar_zr_output = line.split(" ")
            line = myFile.readline()
            ar_rms_output = line.split(" ")
            sum1 = 0.0
            # tính khoảng cách d
            for i in range(1):
                sum1 += abs(float(ar_sc_input[i]) - float(ar_sc_output[i]))**2
                sum1 += abs(float(ar_sr_input[i]) - float(ar_sr_output[i]))**2
                sum1 += abs(float(ar_zr_input[i]) - float(ar_zr_output[i]))**2
                sum1 += abs(float(ar_rms_input[i]) - float(ar_rms_output[i]))**2
            sum1 = sum1**0.5
            # lưu các khoảng cách d vào trong một mảng
            knn_arr = np.append(knn_arr, [[sum1, os.path.basename(myFile.name)]], axis = 0)
            # if(sum1 < knn_min and sum1 != 0):
            #     knn_min = sum1

# Sắp xếp mảng khoảng cách
list_knn = sorted(knn_arr.tolist(), key=lambda x: float(x[0]))

# in ra 3 khoảng cách ngắn nhất
dontau_cnt = 0
songtau_cnt = 0
hoatau_cnt = 0
namee = list_knn[1][1]
for i in range (1,4):
    if("solo" in list_knn[i][1]): dontau_cnt += 1
    if("duet" in list_knn[i][1]): songtau_cnt += 1
    if("hoatau" in list_knn[i][1]): hoatau_cnt += 1
    print(list_knn[i])

if(dontau_cnt == 1 and songtau_cnt == 1 and hoatau_cnt == 1):
    print(namee)

res = {dontau_cnt : "dontau", songtau_cnt : "songtau", hoatau_cnt : "hoatau"}
print(res.get(max(res)))