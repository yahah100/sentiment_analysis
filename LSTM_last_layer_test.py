import matplotlib.pyplot as plt

from TestHelper import TestHelper


net_name = "LSTM"

testhelper = TestHelper()
review_loss, review_acc = testhelper.train_input("Review", net_name, epochs=3)

testhelper.train_last_layer("Twitter", net_name, print_time=True)
testhelper.train_last_layer("Medical", net_name, print_time=True)
testhelper.train_last_layer("Prime", net_name, print_time=True)


tw_loss, tw_acc = testhelper.train_input("Twitter", net_name, epochs=3)


testhelper.train_last_layer("Review", net_name, print_time=True)
testhelper.train_last_layer("Medical", net_name, print_time=True)
testhelper.train_last_layer("Prime", net_name, print_time=True)



"""
Output:

Review
delete old models
/home/yannik/PycharmProjects/bachelorarbeit/TestHelper.py:39: DtypeWarning: Columns (1) have mixed types. Specify dtype option on import or set low_memory=False.
  input_data = self.decide_which_input(input_name)
509411
Tensor("lstm_net/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("lstm_net_1/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   LSTM ---
2018-08-31 12:15:15.718423: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-08-31 12:15:15.802526: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-08-31 12:15:15.802935: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 970 major: 5 minor: 2 memoryClockRate(GHz): 1.253
pciBusID: 0000:01:00.0
totalMemory: 3.94GiB freeMemory: 3.34GiB
2018-08-31 12:15:15.802954: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-31 12:15:15.994794: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-31 12:15:15.994824: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-31 12:15:15.994829: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-31 12:15:15.994962: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/LSTM/pretrained_model.ckpt-0

Saving...
saved to models/LSTM/pretrained_model.ckpt-4482

500 	 [353.18038114 188.48129624 360.66124281]
0 	val accuracy:  0.64233994 	 f_! score:  [0.70636076 0.37696259 0.72132249]


Saving...
saved to models/LSTM/pretrained_model.ckpt-8965

500 	 [360.86399119 226.60948227 370.94087718]
1 	val accuracy:  0.66466004 	 f_! score:  [0.72172798 0.45321896 0.74188175]


Saving...
saved to models/LSTM/pretrained_model.ckpt-13448

500 	 [366.08965702 214.87041231 378.94685001]
2 	val accuracy:  0.67533994 	 f_! score:  [0.73217931 0.42974082 0.7578937 ]


Saving...
saved to models/LSTM/pretrained_model.ckpt-17931

500 	 [359.77141545 242.94912962 382.46533114]
3 	val accuracy:  0.67876 	 f_! score:  [0.71954283 0.48589826 0.76493066]


Saving...
saved to models/LSTM/pretrained_model.ckpt-22414

500 	 [369.39723903 238.76129357 386.85205356]
4 	val accuracy:  0.68962 	 f_! score:  [0.73879448 0.47752259 0.77370411]


Saving...
saved to models/LSTM/pretrained_model.ckpt-26897

500 	 [367.80172106 255.1516987  383.39252349]
5 	val accuracy:  0.68596 	 f_! score:  [0.73560344 0.5103034  0.76678505]


Saving...
saved to models/LSTM/pretrained_model.ckpt-31380

500 	 [373.1346767  240.26395809 389.88101806]
6 	val accuracy:  0.69498 	 f_! score:  [0.74626935 0.48052792 0.77976204]

500 	 [373.1346767  240.26395809 389.88101806]
500 	 [350.03436597 276.02084448 389.1228806 ]
500 	 [402.39886705 216.40361581 393.17000281]
500 	 [18794. 13823. 17383.]
--- Test   Review ---
0.69498
f1:  [0.74626935 0.48052792 0.77976204]
65730
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-31 13:05:29.520731: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-31 13:05:29.520782: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-31 13:05:29.520811: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-31 13:05:29.520818: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-31 13:05:29.520915: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-31 13:05:32.979661: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-31 13:05:32.979703: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-31 13:05:32.979710: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-31 13:05:32.979714: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-31 13:05:32.979792: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
0 : loss:  1.7295047 	 acc:  0.28
2 : loss:  1.895873 	 acc:  0.31
4 : loss:  1.3200644 	 acc:  0.32
6 : loss:  1.2802606 	 acc:  0.42
8 : loss:  1.0739208 	 acc:  0.38
10 : loss:  1.4299858 	 acc:  0.33
12 : loss:  1.2417468 	 acc:  0.31
14 : loss:  1.1714935 	 acc:  0.35
16 : loss:  1.6073401 	 acc:  0.34
18 : loss:  1.1280888 	 acc:  0.43
20 : loss:  1.2148967 	 acc:  0.32
22 : loss:  1.118957 	 acc:  0.46
24 : loss:  1.1667697 	 acc:  0.41
26 : loss:  1.181428 	 acc:  0.39
28 : loss:  1.2622468 	 acc:  0.37
30 : loss:  1.1122317 	 acc:  0.38
32 : loss:  1.2771972 	 acc:  0.33
34 : loss:  1.0530792 	 acc:  0.4
36 : loss:  1.0974957 	 acc:  0.43
38 : loss:  0.97478354 	 acc:  0.49
40 : loss:  1.0372597 	 acc:  0.41
42 : loss:  0.99502516 	 acc:  0.47
44 : loss:  0.9919098 	 acc:  0.51
46 : loss:  1.0253655 	 acc:  0.52
48 : loss:  1.1765146 	 acc:  0.44
50 : loss:  1.1126252 	 acc:  0.42
52 : loss:  1.0742685 	 acc:  0.43
54 : loss:  1.1883547 	 acc:  0.38
56 : loss:  1.1163383 	 acc:  0.41
58 : loss:  1.1532174 	 acc:  0.39
60 : loss:  1.0163647 	 acc:  0.47
62 : loss:  1.1084318 	 acc:  0.45
64 : loss:  0.9742606 	 acc:  0.54
66 : loss:  1.0221529 	 acc:  0.44
68 : loss:  1.1227372 	 acc:  0.44
70 : loss:  0.9819203 	 acc:  0.43
72 : loss:  1.0760826 	 acc:  0.46
74 : loss:  0.9676375 	 acc:  0.52
76 : loss:  1.1242433 	 acc:  0.41
78 : loss:  1.0158824 	 acc:  0.45
80 : loss:  1.0498315 	 acc:  0.43
82 : loss:  1.051713 	 acc:  0.44
84 : loss:  1.0186921 	 acc:  0.48
86 : loss:  1.0319695 	 acc:  0.41
88 : loss:  1.1778876 	 acc:  0.39
90 : loss:  0.98589486 	 acc:  0.5
92 : loss:  1.1055962 	 acc:  0.38
94 : loss:  1.0224303 	 acc:  0.43
96 : loss:  1.1010706 	 acc:  0.5
98 : loss:  1.109581 	 acc:  0.43
100 : loss:  1.003264 	 acc:  0.51
102 : loss:  1.0094339 	 acc:  0.51
104 : loss:  1.0461472 	 acc:  0.47
106 : loss:  1.0067264 	 acc:  0.45
108 : loss:  1.0253228 	 acc:  0.45
110 : loss:  0.9108816 	 acc:  0.61
112 : loss:  0.9463284 	 acc:  0.52
114 : loss:  1.0858448 	 acc:  0.43
116 : loss:  1.0119892 	 acc:  0.49
118 : loss:  1.0932517 	 acc:  0.42
120 : loss:  1.0414455 	 acc:  0.46
122 : loss:  1.0816679 	 acc:  0.45
124 : loss:  1.059947 	 acc:  0.38
126 : loss:  1.1326867 	 acc:  0.43
128 : loss:  1.0040722 	 acc:  0.47
130 : loss:  1.0441709 	 acc:  0.45
132 : loss:  1.0362575 	 acc:  0.5
134 : loss:  1.0446701 	 acc:  0.42
136 : loss:  0.9990719 	 acc:  0.53
138 : loss:  1.0161939 	 acc:  0.5
140 : loss:  0.9630542 	 acc:  0.54
142 : loss:  1.0601276 	 acc:  0.43
144 : loss:  0.99949765 	 acc:  0.48
146 : loss:  0.9635922 	 acc:  0.53
148 : loss:  1.0346274 	 acc:  0.45
150 : loss:  1.0520493 	 acc:  0.48
152 : loss:  0.97126335 	 acc:  0.53
154 : loss:  0.9930491 	 acc:  0.51
156 : loss:  0.92653227 	 acc:  0.57
158 : loss:  1.0432221 	 acc:  0.47
160 : loss:  1.0330368 	 acc:  0.42
162 : loss:  0.95993286 	 acc:  0.55
164 : loss:  0.9394179 	 acc:  0.59
166 : loss:  1.0088711 	 acc:  0.45
168 : loss:  1.11135 	 acc:  0.37
170 : loss:  1.0254467 	 acc:  0.49
172 : loss:  1.0826122 	 acc:  0.41
174 : loss:  1.0477921 	 acc:  0.48
176 : loss:  1.0175964 	 acc:  0.47
178 : loss:  1.0284228 	 acc:  0.48
180 : loss:  1.037424 	 acc:  0.47
182 : loss:  1.0650854 	 acc:  0.38
184 : loss:  1.0192617 	 acc:  0.47
186 : loss:  1.0093446 	 acc:  0.5
188 : loss:  0.95923024 	 acc:  0.51
190 : loss:  1.0234182 	 acc:  0.46
192 : loss:  0.9941567 	 acc:  0.54
194 : loss:  1.1246179 	 acc:  0.45
196 : loss:  0.98309577 	 acc:  0.49
198 : loss:  1.0563248 	 acc:  0.42
200 : loss:  0.9492884 	 acc:  0.51
202 : loss:  1.0240295 	 acc:  0.54
204 : loss:  0.9041577 	 acc:  0.61
206 : loss:  1.0801153 	 acc:  0.39
208 : loss:  0.97575563 	 acc:  0.5
210 : loss:  1.0517901 	 acc:  0.41
212 : loss:  0.92230445 	 acc:  0.55
214 : loss:  0.99214697 	 acc:  0.51
216 : loss:  1.1311288 	 acc:  0.4
218 : loss:  0.91671723 	 acc:  0.55
220 : loss:  1.1224742 	 acc:  0.43
222 : loss:  1.0515945 	 acc:  0.47
224 : loss:  1.0572625 	 acc:  0.46
226 : loss:  1.0704751 	 acc:  0.45
228 : loss:  0.94287145 	 acc:  0.53
230 : loss:  1.0313393 	 acc:  0.46
232 : loss:  0.9119688 	 acc:  0.58
234 : loss:  1.1538509 	 acc:  0.45
236 : loss:  1.0325673 	 acc:  0.51
238 : loss:  1.0252572 	 acc:  0.49
240 : loss:  1.1107514 	 acc:  0.38
242 : loss:  0.9303659 	 acc:  0.5
244 : loss:  1.1636604 	 acc:  0.39
246 : loss:  1.0599195 	 acc:  0.41
248 : loss:  1.0557594 	 acc:  0.46
250 : loss:  1.0484073 	 acc:  0.51
252 : loss:  1.1550769 	 acc:  0.46
254 : loss:  1.0268755 	 acc:  0.48
256 : loss:  1.0933375 	 acc:  0.5
258 : loss:  1.0956005 	 acc:  0.43
260 : loss:  1.2113006 	 acc:  0.39
262 : loss:  1.0869838 	 acc:  0.41
264 : loss:  0.99358153 	 acc:  0.52
266 : loss:  1.0937957 	 acc:  0.41
268 : loss:  0.96632093 	 acc:  0.5
270 : loss:  1.0704522 	 acc:  0.5
272 : loss:  1.0609431 	 acc:  0.45
274 : loss:  1.0226269 	 acc:  0.52
276 : loss:  1.013534 	 acc:  0.44
278 : loss:  1.0098706 	 acc:  0.48
280 : loss:  0.9900865 	 acc:  0.52
282 : loss:  1.1088086 	 acc:  0.43
284 : loss:  1.0339552 	 acc:  0.47
286 : loss:  1.0016264 	 acc:  0.51
288 : loss:  1.0391116 	 acc:  0.47
290 : loss:  0.91842484 	 acc:  0.54
292 : loss:  0.9711919 	 acc:  0.51
294 : loss:  1.048625 	 acc:  0.41
296 : loss:  0.9597027 	 acc:  0.47
298 : loss:  1.0220551 	 acc:  0.46
300 : loss:  1.13032 	 acc:  0.37
302 : loss:  1.0367774 	 acc:  0.52
304 : loss:  0.9131485 	 acc:  0.57
306 : loss:  1.071185 	 acc:  0.46
308 : loss:  1.0234414 	 acc:  0.42
310 : loss:  1.0752827 	 acc:  0.44
312 : loss:  1.004174 	 acc:  0.46
314 : loss:  0.96441925 	 acc:  0.56
316 : loss:  1.0322366 	 acc:  0.45
318 : loss:  0.9984195 	 acc:  0.47
320 : loss:  1.0238667 	 acc:  0.38
322 : loss:  1.0470192 	 acc:  0.47
324 : loss:  1.0131745 	 acc:  0.49
326 : loss:  0.9320643 	 acc:  0.48
328 : loss:  0.9808796 	 acc:  0.56
330 : loss:  0.9454161 	 acc:  0.55
332 : loss:  1.0176537 	 acc:  0.47
334 : loss:  0.9773147 	 acc:  0.48
336 : loss:  1.0506485 	 acc:  0.45
338 : loss:  0.9406931 	 acc:  0.58
340 : loss:  0.9227222 	 acc:  0.53
342 : loss:  0.9930559 	 acc:  0.52
344 : loss:  0.9793512 	 acc:  0.47
346 : loss:  0.94256014 	 acc:  0.57
348 : loss:  0.9771925 	 acc:  0.53
350 : loss:  1.0183352 	 acc:  0.47
352 : loss:  1.0366083 	 acc:  0.53
354 : loss:  1.0427547 	 acc:  0.45
356 : loss:  1.0151079 	 acc:  0.48
358 : loss:  0.9198288 	 acc:  0.58
360 : loss:  1.0346054 	 acc:  0.49
362 : loss:  0.8918891 	 acc:  0.57
364 : loss:  1.0349097 	 acc:  0.48
366 : loss:  0.93672293 	 acc:  0.57
368 : loss:  0.93397254 	 acc:  0.53
370 : loss:  0.9445471 	 acc:  0.55

100 	 [60.5076212  49.31804056 40.41858486]
0 	val accuracy:  0.5197 	 f_! score:  [0.60507621 0.49318041 0.40418585]

100 	 [60.5076212  49.31804056 40.41858486]
100 	 [49.78985745 55.51685646 52.3584123 ]
100 	 [78.07499779 45.00417334 33.46607303]
100 	 [3286. 3352. 3362.]
---train_last_layer Test  Twitter ---
0.5197
f1:  [0.60507621 0.49318041 0.40418585]
(7733, 3)
7733
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-31 13:06:38.651217: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-31 13:06:38.651251: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-31 13:06:38.651257: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-31 13:06:38.651261: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-31 13:06:38.651340: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-31 13:06:41.457757: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-31 13:06:41.457798: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-31 13:06:41.457804: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-31 13:06:41.457809: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-31 13:06:41.457899: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
0 : loss:  1.1877625 	 acc:  0.55
2 : loss:  1.7435123 	 acc:  0.5
4 : loss:  0.8208047 	 acc:  0.75757575
6 : loss:  1.1312548 	 acc:  0.51
8 : loss:  0.4009604 	 acc:  0.92
10 : loss:  0.58213264 	 acc:  0.86
12 : loss:  1.3039217 	 acc:  0.68
14 : loss:  1.4102467 	 acc:  0.46
16 : loss:  1.0754752 	 acc:  0.61
18 : loss:  0.9960483 	 acc:  0.68
20 : loss:  0.5654698 	 acc:  0.86
22 : loss:  0.85033417 	 acc:  0.75
24 : loss:  0.54380924 	 acc:  0.88
26 : loss:  1.2240627 	 acc:  0.36
28 : loss:  0.97364444 	 acc:  0.58
30 : loss:  0.98702806 	 acc:  0.59
32 : loss:  0.9474401 	 acc:  0.65
34 : loss:  0.75694746 	 acc:  0.8
36 : loss:  0.9225502 	 acc:  0.76
38 : loss:  0.9665638 	 acc:  0.5
40 : loss:  1.5000364 	 acc:  0.17
/home/yannik/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)

14 	 [1.65016494 0.         6.38069623]
0 	val accuracy:  0.31428573 	 f_! score:  [0.11786892 0.         0.45576402]

14 	 [1.65016494 0.         6.38069623]
14 	 [ 0.95391839  0.         11.52067012]
14 	 [8.67912088 0.         4.49547581]
14 	 [ 105.  116. 1179.]
---train_last_layer Test  Medical ---
0.31428573
f1:  [0.11786892 0.         0.45576402]
(37126,)
(37126,)
37126
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-31 13:07:15.480612: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-31 13:07:15.480654: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-31 13:07:15.480662: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-31 13:07:15.480669: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-31 13:07:15.480759: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-31 13:07:18.257377: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-31 13:07:18.257419: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-31 13:07:18.257425: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-31 13:07:18.257429: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-31 13:07:18.257515: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
0 : loss:  0.45486283 	 acc:  0.89
2 : loss:  1.55518 	 acc:  0.77
4 : loss:  0.6850934 	 acc:  0.9
6 : loss:  0.6654037 	 acc:  0.83
8 : loss:  1.0036399 	 acc:  0.55
10 : loss:  1.0582852 	 acc:  0.58
12 : loss:  0.8083588 	 acc:  0.66
14 : loss:  1.0136315 	 acc:  0.69
16 : loss:  0.3534069 	 acc:  0.92
18 : loss:  1.0463219 	 acc:  0.68
20 : loss:  1.1599976 	 acc:  0.49
22 : loss:  1.3399305 	 acc:  0.43
24 : loss:  0.5203784 	 acc:  0.87
26 : loss:  0.50229204 	 acc:  0.91
28 : loss:  1.7310828 	 acc:  0.57
30 : loss:  1.1556097 	 acc:  0.65
32 : loss:  0.70991576 	 acc:  0.7
34 : loss:  1.0115069 	 acc:  0.49
36 : loss:  0.9511825 	 acc:  0.6
38 : loss:  0.67141175 	 acc:  0.78
40 : loss:  1.1694628 	 acc:  0.62
42 : loss:  0.846147 	 acc:  0.76
44 : loss:  0.47449353 	 acc:  0.88
46 : loss:  0.7681656 	 acc:  0.74
48 : loss:  0.6589713 	 acc:  0.8
50 : loss:  0.37042442 	 acc:  0.91
52 : loss:  0.82922673 	 acc:  0.73
54 : loss:  1.2675415 	 acc:  0.57
56 : loss:  0.5264988 	 acc:  0.86
58 : loss:  0.63269806 	 acc:  0.84
60 : loss:  0.8227498 	 acc:  0.74
62 : loss:  0.7986935 	 acc:  0.75
64 : loss:  0.79353637 	 acc:  0.76
66 : loss:  0.67537326 	 acc:  0.78
68 : loss:  0.44964275 	 acc:  0.88
70 : loss:  1.1314019 	 acc:  0.66
72 : loss:  0.23172167 	 acc:  0.96
74 : loss:  0.8422459 	 acc:  0.67
76 : loss:  0.8904177 	 acc:  0.63
78 : loss:  0.84579027 	 acc:  0.76
80 : loss:  1.3500302 	 acc:  0.67
82 : loss:  0.9060456 	 acc:  0.8
84 : loss:  0.95445985 	 acc:  0.75
86 : loss:  0.62254846 	 acc:  0.75
88 : loss:  0.9643185 	 acc:  0.66
90 : loss:  1.1752994 	 acc:  0.64
92 : loss:  0.91319185 	 acc:  0.72
94 : loss:  0.3084403 	 acc:  0.95
96 : loss:  0.6916886 	 acc:  0.76
98 : loss:  0.5366099 	 acc:  0.82
100 : loss:  0.8802825 	 acc:  0.68
102 : loss:  0.532156 	 acc:  0.83
104 : loss:  0.5122961 	 acc:  0.84
106 : loss:  0.34845203 	 acc:  0.97
108 : loss:  0.7711087 	 acc:  0.75
110 : loss:  1.1184478 	 acc:  0.61
112 : loss:  0.7980523 	 acc:  0.81
114 : loss:  0.646668 	 acc:  0.85
116 : loss:  1.2588155 	 acc:  0.68
118 : loss:  0.5417137 	 acc:  0.83
120 : loss:  0.72444725 	 acc:  0.65
122 : loss:  0.60643226 	 acc:  0.8
124 : loss:  1.2596612 	 acc:  0.47
126 : loss:  0.8615663 	 acc:  0.68
128 : loss:  0.79115844 	 acc:  0.78
130 : loss:  1.7125853 	 acc:  0.49
132 : loss:  1.2164041 	 acc:  0.5
134 : loss:  0.7930384 	 acc:  0.71
136 : loss:  0.7026261 	 acc:  0.8
138 : loss:  0.6341814 	 acc:  0.83
140 : loss:  1.0473021 	 acc:  0.77
142 : loss:  1.0730118 	 acc:  0.66
144 : loss:  0.8504187 	 acc:  0.67
146 : loss:  0.7794961 	 acc:  0.76
148 : loss:  0.59254295 	 acc:  0.85
150 : loss:  0.3010418 	 acc:  0.95
152 : loss:  0.95229614 	 acc:  0.64
154 : loss:  0.32849178 	 acc:  0.95
156 : loss:  0.6722896 	 acc:  0.85
158 : loss:  0.7810766 	 acc:  0.75
160 : loss:  0.43403193 	 acc:  0.91
162 : loss:  0.9006679 	 acc:  0.63
164 : loss:  0.5966077 	 acc:  0.85
166 : loss:  0.6486527 	 acc:  0.79
168 : loss:  0.20117542 	 acc:  0.96
170 : loss:  0.84818107 	 acc:  0.82
172 : loss:  0.88186055 	 acc:  0.7
174 : loss:  1.0548182 	 acc:  0.43
176 : loss:  0.6530316 	 acc:  0.84
178 : loss:  0.7144587 	 acc:  0.78
180 : loss:  0.8255722 	 acc:  0.76
182 : loss:  1.6186503 	 acc:  0.66
184 : loss:  0.8236651 	 acc:  0.72
186 : loss:  1.0185353 	 acc:  0.52
188 : loss:  0.7354927 	 acc:  0.76
190 : loss:  0.98420817 	 acc:  0.7
192 : loss:  0.81865466 	 acc:  0.82
194 : loss:  1.0162283 	 acc:  0.76
196 : loss:  0.27053273 	 acc:  0.92
198 : loss:  1.008419 	 acc:  0.49
200 : loss:  0.58286214 	 acc:  0.81
/home/yannik/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)

70 	 [ 8.7354779   0.33333333 58.2685396 ]
0 	val accuracy:  0.70514274 	 f_! score:  [0.12479254 0.0047619  0.83240771]

70 	 [ 8.7354779   0.33333333 58.2685396 ]
70 	 [ 6.63938606  1.         59.55344775]
70 	 [17.70431053  0.2        57.42857997]
70 	 [ 501.  664. 5835.]
---train_last_layer Test  Prime ---
0.70514274
f1:  [0.12479254 0.0047619  0.83240771]


  Twitter
delete old models
65730
Tensor("lstm_net/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("lstm_net_1/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   LSTM ---
2018-08-31 13:08:11.727767: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-31 13:08:11.727800: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-31 13:08:11.727806: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-31 13:08:11.727811: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-31 13:08:11.727892: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/LSTM/pretrained_model.ckpt-0

Saving...
saved to models/LSTM/pretrained_model.ckpt-372

100 	 [63.35939485 61.89159176 45.7724945 ]
0 	val accuracy:  0.58419997 	 f_! score:  [0.63359395 0.61891592 0.45772494]


Saving...
saved to models/LSTM/pretrained_model.ckpt-744

100 	 [66.05239761 61.32158723 52.69403142]
1 	val accuracy:  0.60580003 	 f_! score:  [0.66052398 0.61321587 0.52694031]


Saving...
saved to models/LSTM/pretrained_model.ckpt-1116

100 	 [66.21561971 63.46161503 48.02607838]
2 	val accuracy:  0.6067 	 f_! score:  [0.6621562  0.63461615 0.48026078]


Saving...
saved to models/LSTM/pretrained_model.ckpt-1488

100 	 [61.20656232 63.32437557 52.2898622 ]
3 	val accuracy:  0.59610003 	 f_! score:  [0.61206562 0.63324376 0.52289862]


Saving...
saved to models/LSTM/pretrained_model.ckpt-1860

100 	 [67.44161686 64.75417191 48.16451438]
4 	val accuracy:  0.616 	 f_! score:  [0.67441617 0.64754172 0.48164514]


Saving...
saved to models/LSTM/pretrained_model.ckpt-2232

100 	 [67.44805671 61.54276515 55.72725464]
5 	val accuracy:  0.62079996 	 f_! score:  [0.67448057 0.61542765 0.55727255]


Saving...
saved to models/LSTM/pretrained_model.ckpt-2604

100 	 [67.32811201 62.51522076 55.70976034]
6 	val accuracy:  0.62399995 	 f_! score:  [0.67328112 0.62515221 0.5570976 ]

100 	 [67.32811201 62.51522076 55.70976034]
100 	 [65.27184161 62.03166175 59.2813442 ]
100 	 [70.14179376 63.84931282 53.31186115]
100 	 [3323. 3339. 3338.]
--- Test   Twitter ---
0.62399995
f1:  [0.67328112 0.62515221 0.5570976 ]
/home/yannik/PycharmProjects/bachelorarbeit/TestHelper.py:72: DtypeWarning: Columns (1) have mixed types. Specify dtype option on import or set low_memory=False.
  input_data = self.decide_which_input(input_name)
509411
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-31 13:14:52.895570: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-31 13:14:52.895606: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-31 13:14:52.895613: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-31 13:14:52.895617: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-31 13:14:52.895717: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-31 13:14:56.074151: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-31 13:14:56.074193: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-31 13:14:56.074199: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-31 13:14:56.074204: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-31 13:14:56.074297: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
0 : loss:  2.3958073 	 acc:  0.3
100 : loss:  1.1257892 	 acc:  0.35
200 : loss:  1.0816803 	 acc:  0.44
300 : loss:  1.0200257 	 acc:  0.49
400 : loss:  1.0122294 	 acc:  0.48
500 : loss:  1.0492172 	 acc:  0.49
600 : loss:  1.122659 	 acc:  0.4
700 : loss:  1.0825452 	 acc:  0.48
800 : loss:  1.0228486 	 acc:  0.51
900 : loss:  1.2311394 	 acc:  0.35
1000 : loss:  1.0200676 	 acc:  0.5
1100 : loss:  1.1161048 	 acc:  0.38
1200 : loss:  1.2411996 	 acc:  0.37
1300 : loss:  1.1158626 	 acc:  0.42
1400 : loss:  1.2582881 	 acc:  0.38
1500 : loss:  1.1033297 	 acc:  0.47
1600 : loss:  0.92325246 	 acc:  0.59
1700 : loss:  1.0117435 	 acc:  0.53
1800 : loss:  1.01702 	 acc:  0.48
1900 : loss:  1.1425065 	 acc:  0.41
2000 : loss:  1.1098816 	 acc:  0.46
2100 : loss:  0.9543219 	 acc:  0.55
2200 : loss:  1.0410168 	 acc:  0.49
2300 : loss:  1.1828843 	 acc:  0.4
2400 : loss:  1.0095974 	 acc:  0.47
2500 : loss:  1.122103 	 acc:  0.47
2600 : loss:  1.0713941 	 acc:  0.47
2700 : loss:  1.100483 	 acc:  0.45
2800 : loss:  1.1552131 	 acc:  0.45
2900 : loss:  1.0591265 	 acc:  0.47
3000 : loss:  1.1805247 	 acc:  0.39
3100 : loss:  1.1529841 	 acc:  0.36
3200 : loss:  1.0153674 	 acc:  0.51
3300 : loss:  1.0221658 	 acc:  0.52
3400 : loss:  1.1439668 	 acc:  0.44
3500 : loss:  1.0634105 	 acc:  0.39
3600 : loss:  0.9922099 	 acc:  0.5
3700 : loss:  1.164795 	 acc:  0.41
3800 : loss:  0.9662042 	 acc:  0.53
3900 : loss:  1.1561826 	 acc:  0.45
4000 : loss:  1.1507671 	 acc:  0.37
4100 : loss:  1.0879798 	 acc:  0.47
4200 : loss:  1.0121046 	 acc:  0.52
4300 : loss:  1.0063454 	 acc:  0.49
4400 : loss:  1.0550458 	 acc:  0.46
4500 : loss:  0.99829245 	 acc:  0.55

500 	 [185.98435567 182.16988578 282.00738513]
0 	val accuracy:  0.45856 	 f_! score:  [0.37196871 0.36433977 0.56401477]

500 	 [185.98435567 182.16988578 282.00738513]
500 	 [302.72972136 180.19424423 232.61573241]
500 	 [136.44819831 188.26831215 362.18346005]
500 	 [18794. 13823. 17383.]
---train_last_layer Test  Review ---
0.45856
f1:  [0.37196871 0.36433977 0.56401477]
(7733, 3)
7733
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-31 13:22:34.139232: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-31 13:22:34.139269: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-31 13:22:34.139275: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-31 13:22:34.139279: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-31 13:22:34.139363: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-31 13:22:37.022728: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-31 13:22:37.022771: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-31 13:22:37.022777: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-31 13:22:37.022782: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-31 13:22:37.022873: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
0 : loss:  0.6892545 	 acc:  0.84
2 : loss:  1.0994577 	 acc:  0.69
4 : loss:  1.1691236 	 acc:  0.43
6 : loss:  1.0847781 	 acc:  0.58
8 : loss:  2.073466 	 acc:  0.38
10 : loss:  0.68185395 	 acc:  0.83
12 : loss:  1.4724665 	 acc:  0.48
14 : loss:  1.0558375 	 acc:  0.55
16 : loss:  1.3667307 	 acc:  0.4
18 : loss:  1.3281853 	 acc:  0.51
20 : loss:  1.0818357 	 acc:  0.62
22 : loss:  1.0011914 	 acc:  0.72
24 : loss:  0.581646 	 acc:  0.86
26 : loss:  0.95982444 	 acc:  0.59
28 : loss:  0.86508316 	 acc:  0.61
30 : loss:  0.5593496 	 acc:  0.88
32 : loss:  1.3668114 	 acc:  0.52
34 : loss:  0.48424152 	 acc:  0.85
36 : loss:  1.1892576 	 acc:  0.5
38 : loss:  0.8204078 	 acc:  0.66
40 : loss:  1.2475481 	 acc:  0.19

14 	 [ 0.          0.         12.77481666]
0 	val accuracy:  0.84214294 	 f_! score:  [0.        0.        0.9124869]

14 	 [ 0.          0.         12.77481666]
14 	 [ 0.    0.   11.79]
14 	 [ 0.  0. 14.]
14 	 [ 105.  116. 1179.]
---train_last_layer Test  Medical ---
0.84214294
f1:  [0.        0.        0.9124869]
(37126,)
(37126,)
37126
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-31 13:23:11.169577: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-31 13:23:11.169618: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-31 13:23:11.169626: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-31 13:23:11.169631: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-31 13:23:11.169724: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-31 13:23:13.998099: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-31 13:23:13.998141: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-31 13:23:13.998147: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-31 13:23:13.998152: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-31 13:23:13.998252: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3058 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
0 : loss:  0.51505774 	 acc:  0.88
2 : loss:  0.25707212 	 acc:  0.96
4 : loss:  1.3679258 	 acc:  0.6
6 : loss:  1.0037279 	 acc:  0.6
8 : loss:  0.7614667 	 acc:  0.82
10 : loss:  0.4274081 	 acc:  0.9
12 : loss:  0.29765794 	 acc:  0.93
14 : loss:  0.5074665 	 acc:  0.88
16 : loss:  1.3710178 	 acc:  0.56
18 : loss:  0.49122864 	 acc:  0.89
20 : loss:  0.8738844 	 acc:  0.65
22 : loss:  0.5820676 	 acc:  0.83
24 : loss:  1.2547047 	 acc:  0.6
26 : loss:  0.7490471 	 acc:  0.81
28 : loss:  0.8948454 	 acc:  0.67
30 : loss:  1.5708623 	 acc:  0.49
32 : loss:  0.34500486 	 acc:  0.92
34 : loss:  1.1391788 	 acc:  0.66
36 : loss:  1.0020374 	 acc:  0.64
38 : loss:  0.33967674 	 acc:  0.93
40 : loss:  0.4305041 	 acc:  0.89
42 : loss:  0.8606858 	 acc:  0.8
44 : loss:  1.3053888 	 acc:  0.54
46 : loss:  0.48759452 	 acc:  0.92
48 : loss:  0.8352417 	 acc:  0.7
50 : loss:  0.67824286 	 acc:  0.76
52 : loss:  0.8262082 	 acc:  0.66
54 : loss:  0.5936439 	 acc:  0.84
56 : loss:  0.39619306 	 acc:  0.92
58 : loss:  0.5279009 	 acc:  0.82
60 : loss:  1.2253305 	 acc:  0.57
62 : loss:  0.51052815 	 acc:  0.84
64 : loss:  0.67166525 	 acc:  0.78
66 : loss:  0.30010295 	 acc:  0.93
68 : loss:  0.5943694 	 acc:  0.81
70 : loss:  0.7728547 	 acc:  0.82
72 : loss:  0.67357683 	 acc:  0.8
74 : loss:  0.695685 	 acc:  0.8
76 : loss:  0.24494033 	 acc:  0.95
78 : loss:  0.5277162 	 acc:  0.96
80 : loss:  1.1791087 	 acc:  0.25
82 : loss:  0.87083894 	 acc:  0.74
84 : loss:  0.67678285 	 acc:  0.84
86 : loss:  1.4587399 	 acc:  0.62
88 : loss:  0.9195322 	 acc:  0.62
90 : loss:  0.8477527 	 acc:  0.71
92 : loss:  0.5814469 	 acc:  0.83
94 : loss:  0.942188 	 acc:  0.62
96 : loss:  0.72454447 	 acc:  0.69
98 : loss:  0.35367748 	 acc:  0.91
100 : loss:  0.8670338 	 acc:  0.67
102 : loss:  0.88266623 	 acc:  0.68
104 : loss:  0.47815686 	 acc:  0.87
106 : loss:  0.23755161 	 acc:  0.97
108 : loss:  0.6165497 	 acc:  0.82
110 : loss:  0.80298126 	 acc:  0.73
112 : loss:  0.6398866 	 acc:  0.8
114 : loss:  0.60669094 	 acc:  0.76
116 : loss:  1.0589118 	 acc:  0.72
118 : loss:  0.67955095 	 acc:  0.84
120 : loss:  0.2922474 	 acc:  0.93
122 : loss:  1.0504204 	 acc:  0.62
124 : loss:  0.651567 	 acc:  0.82
126 : loss:  0.7891967 	 acc:  0.68
128 : loss:  0.8252444 	 acc:  0.67
130 : loss:  0.582708 	 acc:  0.85
132 : loss:  0.6095601 	 acc:  0.84
134 : loss:  0.39547318 	 acc:  0.9
136 : loss:  0.4919438 	 acc:  0.87
138 : loss:  1.4321612 	 acc:  0.56
140 : loss:  0.47167167 	 acc:  0.84
142 : loss:  0.8875563 	 acc:  0.71
144 : loss:  0.52795726 	 acc:  0.84
146 : loss:  0.6883963 	 acc:  0.76
148 : loss:  0.5098393 	 acc:  0.84
150 : loss:  1.718809 	 acc:  0.41
152 : loss:  1.1516175 	 acc:  0.68
154 : loss:  1.0818539 	 acc:  0.63
156 : loss:  0.6272228 	 acc:  0.83
158 : loss:  0.85441077 	 acc:  0.7
160 : loss:  1.0495387 	 acc:  0.44
162 : loss:  0.9110489 	 acc:  0.74
164 : loss:  0.60628015 	 acc:  0.89
166 : loss:  1.3854958 	 acc:  0.78
168 : loss:  0.911771 	 acc:  0.81
170 : loss:  0.5713468 	 acc:  0.87
172 : loss:  0.9287095 	 acc:  0.75
174 : loss:  0.5935794 	 acc:  0.82
176 : loss:  0.68067956 	 acc:  0.8
178 : loss:  1.1461842 	 acc:  0.63
180 : loss:  0.2719182 	 acc:  0.95
182 : loss:  0.79403913 	 acc:  0.75
184 : loss:  0.63811153 	 acc:  0.89
186 : loss:  1.0372678 	 acc:  0.73
188 : loss:  1.0070226 	 acc:  0.72
190 : loss:  0.92451364 	 acc:  0.69
192 : loss:  0.86751777 	 acc:  0.65
194 : loss:  0.8089532 	 acc:  0.67
196 : loss:  0.5414872 	 acc:  0.81
198 : loss:  0.4987768 	 acc:  0.88
200 : loss:  0.82879835 	 acc:  0.77
(2,)
(2,)

68 	 [ 0.          2.45945867 61.12918395]
0 	val accuracy:  0.82239133 	 f_! score:  [0.         0.03616851 0.89895859]

68 	 [ 0.          2.45945867 61.12918395]
68 	 [ 0.          6.55       56.49894771]
68 	 [ 0.          1.54116332 66.95286055]
68 	 [ 501.  657. 5642.]
---train_last_layer Test  Prime ---
0.82239133
f1:  [0.         0.03616851 0.89895859]
(base) yannik@yannik-All-Series:~/PycharmProjects/bachelorarbeit$ ^C
(base) yannik@yannik-All-Series:~/PycharmProjects/bachelorarbeit$ 


"""