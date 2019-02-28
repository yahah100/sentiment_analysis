from TestHelper import TestHelper


net_name = "CNN"

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
/home/yannik/PycharmProjects/bachelorarbeit/TestHelper.py:57: DtypeWarning: Columns (1) have mixed types. Specify dtype option on import or set low_memory=False.
  input_data = self.decide_which_input(input_name)
509411
Tensor("ConvNet/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("ConvNet_1/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   CNN ---
2018-09-05 15:47:32.114781: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-09-05 15:47:32.204462: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-09-05 15:47:32.204823: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 970 major: 5 minor: 2 memoryClockRate(GHz): 1.253
pciBusID: 0000:01:00.0
totalMemory: 3.94GiB freeMemory: 3.25GiB
2018-09-05 15:47:32.204842: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 15:47:32.932879: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 15:47:32.932908: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-09-05 15:47:32.932913: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-09-05 15:47:32.933043: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2957 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/CNN/pretrained_model.ckpt-0
2018-09-05 15:47:54.471268: W tensorflow/core/common_runtime/bfc_allocator.cc:219] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.60GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.

Saving...
saved to models/CNN/pretrained_model.ckpt-4482

500 	 [356.32887708 164.99468135 368.75435049]
0 	val accuracy:  0.64732003 	 f_! score:  [0.71265775 0.32998936 0.7375087 ]


Saving...
saved to models/CNN/pretrained_model.ckpt-8965

500 	 [367.04164723 225.91758994 382.07574785]
1 	val accuracy:  0.68193996 	 f_! score:  [0.73408329 0.45183518 0.7641515 ]


Saving...
saved to models/CNN/pretrained_model.ckpt-13448

500 	 [369.55884908 238.68461603 378.69260017]
2 	val accuracy:  0.68233997 	 f_! score:  [0.7391177  0.47736923 0.7573852 ]


Saving...
saved to models/CNN/pretrained_model.ckpt-17931

500 	 [370.76399552 223.52409235 384.57475222]
3 	val accuracy:  0.6855 	 f_! score:  [0.74152799 0.44704818 0.7691495 ]


Saving...
saved to models/CNN/pretrained_model.ckpt-22414

500 	 [373.84580304 219.29808286 388.97786695]
4 	val accuracy:  0.693 	 f_! score:  [0.74769161 0.43859617 0.77795573]


Saving...
saved to models/CNN/pretrained_model.ckpt-26897

500 	 [371.03413503 254.02417163 386.7827316 ]
5 	val accuracy:  0.69376004 	 f_! score:  [0.74206827 0.50804834 0.77356546]


Saving...
saved to models/CNN/pretrained_model.ckpt-31380

500 	 [358.25586719 273.57696541 385.64783025]
6 	val accuracy:  0.68534 	 f_! score:  [0.71651173 0.54715393 0.77129566]

500 	 [358.25586719 273.57696541 385.64783025]
500 	 [383.90006377 249.0920544  399.80258241]
500 	 [338.66852281 308.43458583 374.92190785]
500 	 [18794. 13823. 17383.]
--- Test   Review ---
0.68534
f1:  [0.71651173 0.54715393 0.77129566]
65730
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-09-05 16:10:42.173394: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 16:10:42.174365: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 16:10:42.174377: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-09-05 16:10:42.174381: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-09-05 16:10:42.176065: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2957 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
models/CNN/pretrained_model.ckpt-31380
2018-09-05 16:10:45.899206: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 16:10:45.899247: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 16:10:45.899253: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-09-05 16:10:45.899258: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-09-05 16:10:45.899346: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2957 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
0 : loss:  1.1136374 	 acc:  0.34
2 : loss:  1.0882286 	 acc:  0.38
4 : loss:  1.0952308 	 acc:  0.42
6 : loss:  1.1068803 	 acc:  0.35
8 : loss:  1.1230295 	 acc:  0.32
10 : loss:  1.0954101 	 acc:  0.32
12 : loss:  1.0989969 	 acc:  0.33
14 : loss:  1.0787255 	 acc:  0.49
16 : loss:  1.1152451 	 acc:  0.43
18 : loss:  1.1336465 	 acc:  0.29
20 : loss:  1.088222 	 acc:  0.39
22 : loss:  1.0868248 	 acc:  0.36
24 : loss:  1.0803025 	 acc:  0.4
26 : loss:  1.0936397 	 acc:  0.41
28 : loss:  1.0874867 	 acc:  0.34
30 : loss:  1.0649176 	 acc:  0.42
32 : loss:  1.0820507 	 acc:  0.45
34 : loss:  1.1334834 	 acc:  0.31
36 : loss:  1.0831053 	 acc:  0.34
38 : loss:  1.1050085 	 acc:  0.36
40 : loss:  1.0759751 	 acc:  0.41
42 : loss:  1.0771433 	 acc:  0.38
44 : loss:  1.0726023 	 acc:  0.42
46 : loss:  1.0687318 	 acc:  0.51
48 : loss:  1.0658447 	 acc:  0.52
50 : loss:  1.0647671 	 acc:  0.48
52 : loss:  1.0857695 	 acc:  0.44
54 : loss:  1.0809883 	 acc:  0.38
56 : loss:  1.0707775 	 acc:  0.48
58 : loss:  1.0672516 	 acc:  0.44
60 : loss:  1.0727096 	 acc:  0.46
62 : loss:  1.0717053 	 acc:  0.43
64 : loss:  1.0629886 	 acc:  0.45
66 : loss:  1.061866 	 acc:  0.39
68 : loss:  1.0865486 	 acc:  0.41
70 : loss:  1.078125 	 acc:  0.36
72 : loss:  1.043598 	 acc:  0.49
74 : loss:  1.0802289 	 acc:  0.4
76 : loss:  1.0699726 	 acc:  0.38
78 : loss:  1.0662011 	 acc:  0.49
80 : loss:  1.0266765 	 acc:  0.45
82 : loss:  1.0603414 	 acc:  0.43
84 : loss:  1.0647116 	 acc:  0.39
86 : loss:  1.0503293 	 acc:  0.45
88 : loss:  1.0322721 	 acc:  0.47
90 : loss:  1.094861 	 acc:  0.31
92 : loss:  1.0840414 	 acc:  0.41
94 : loss:  1.0838976 	 acc:  0.38
96 : loss:  1.0713973 	 acc:  0.46
98 : loss:  1.064138 	 acc:  0.43
100 : loss:  1.094319 	 acc:  0.34
102 : loss:  1.0307313 	 acc:  0.5
104 : loss:  1.0287873 	 acc:  0.46
106 : loss:  1.0597558 	 acc:  0.49
108 : loss:  1.058192 	 acc:  0.44
110 : loss:  1.0456568 	 acc:  0.42
112 : loss:  1.076968 	 acc:  0.39
114 : loss:  1.0344656 	 acc:  0.43
116 : loss:  1.0475069 	 acc:  0.45
118 : loss:  1.0584188 	 acc:  0.48
2018-09-05 16:11:07.106396: W tensorflow/core/common_runtime/bfc_allocator.cc:219] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.31GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
120 : loss:  1.101929 	 acc:  0.36666667
122 : loss:  1.0505942 	 acc:  0.36
124 : loss:  1.0856905 	 acc:  0.35
126 : loss:  1.0905968 	 acc:  0.34
128 : loss:  1.0433354 	 acc:  0.43
130 : loss:  1.0578159 	 acc:  0.46
132 : loss:  1.0591094 	 acc:  0.42
134 : loss:  1.0412594 	 acc:  0.5
136 : loss:  1.0568881 	 acc:  0.43
138 : loss:  1.1055334 	 acc:  0.39
140 : loss:  1.0560299 	 acc:  0.42
142 : loss:  1.0296035 	 acc:  0.52
144 : loss:  1.0912395 	 acc:  0.34
146 : loss:  1.0173924 	 acc:  0.52
148 : loss:  1.0564879 	 acc:  0.43
150 : loss:  1.0436947 	 acc:  0.4
152 : loss:  1.1049972 	 acc:  0.34
154 : loss:  1.0850967 	 acc:  0.35
156 : loss:  1.0367397 	 acc:  0.44
158 : loss:  1.0472174 	 acc:  0.45
160 : loss:  1.0603391 	 acc:  0.48
162 : loss:  1.0544963 	 acc:  0.4
164 : loss:  1.0798848 	 acc:  0.44
166 : loss:  1.0574962 	 acc:  0.45
168 : loss:  1.058441 	 acc:  0.43
170 : loss:  1.1071503 	 acc:  0.3
172 : loss:  1.0872487 	 acc:  0.4
174 : loss:  1.0877706 	 acc:  0.36
176 : loss:  1.0834879 	 acc:  0.4
178 : loss:  1.0563595 	 acc:  0.44
180 : loss:  1.046126 	 acc:  0.41
182 : loss:  1.0402529 	 acc:  0.42
184 : loss:  1.0769643 	 acc:  0.41
186 : loss:  1.077336 	 acc:  0.39
188 : loss:  1.0146853 	 acc:  0.47
190 : loss:  1.0301604 	 acc:  0.49
192 : loss:  1.0611614 	 acc:  0.43
194 : loss:  1.0718117 	 acc:  0.41
196 : loss:  1.045007 	 acc:  0.45
198 : loss:  1.0689982 	 acc:  0.45
200 : loss:  1.0615451 	 acc:  0.44
202 : loss:  1.0907619 	 acc:  0.41
204 : loss:  1.0888765 	 acc:  0.38
206 : loss:  1.0735685 	 acc:  0.43
208 : loss:  1.0748442 	 acc:  0.39
210 : loss:  1.0149785 	 acc:  0.54
212 : loss:  1.0732796 	 acc:  0.45
214 : loss:  1.064205 	 acc:  0.41
216 : loss:  1.045437 	 acc:  0.49
218 : loss:  1.0733474 	 acc:  0.44
220 : loss:  1.0622916 	 acc:  0.45
222 : loss:  1.0506074 	 acc:  0.45
224 : loss:  1.0109816 	 acc:  0.53
226 : loss:  1.0774044 	 acc:  0.38
228 : loss:  1.0903074 	 acc:  0.37
230 : loss:  1.0391457 	 acc:  0.4
232 : loss:  1.0550547 	 acc:  0.48
234 : loss:  1.054945 	 acc:  0.46
236 : loss:  1.0368024 	 acc:  0.46
238 : loss:  1.0265046 	 acc:  0.46
240 : loss:  1.045948 	 acc:  0.48
242 : loss:  1.0536346 	 acc:  0.42
244 : loss:  1.0324706 	 acc:  0.4
246 : loss:  1.0448874 	 acc:  0.38
248 : loss:  1.0932099 	 acc:  0.37
250 : loss:  1.0337552 	 acc:  0.48
252 : loss:  1.0008551 	 acc:  0.5
254 : loss:  1.0657157 	 acc:  0.39
256 : loss:  1.0659913 	 acc:  0.39
258 : loss:  1.1064464 	 acc:  0.34
260 : loss:  1.0666666 	 acc:  0.35
262 : loss:  1.0270554 	 acc:  0.5
264 : loss:  1.0412569 	 acc:  0.42
266 : loss:  1.0784576 	 acc:  0.41
268 : loss:  1.046592 	 acc:  0.42
270 : loss:  1.0478339 	 acc:  0.41
272 : loss:  1.0583392 	 acc:  0.39
274 : loss:  1.0595657 	 acc:  0.45
276 : loss:  1.045576 	 acc:  0.48
278 : loss:  1.0758433 	 acc:  0.41
280 : loss:  1.0542078 	 acc:  0.49
282 : loss:  1.0113826 	 acc:  0.48
284 : loss:  1.0491817 	 acc:  0.39
286 : loss:  0.9982839 	 acc:  0.54
288 : loss:  1.0809528 	 acc:  0.36
290 : loss:  1.0584581 	 acc:  0.44
292 : loss:  1.0754151 	 acc:  0.43
294 : loss:  1.0536339 	 acc:  0.45
296 : loss:  1.0604719 	 acc:  0.44
298 : loss:  1.0675606 	 acc:  0.39
300 : loss:  1.0399528 	 acc:  0.41
302 : loss:  1.0338302 	 acc:  0.47
304 : loss:  1.0734504 	 acc:  0.44
306 : loss:  1.0578072 	 acc:  0.42
308 : loss:  1.0486405 	 acc:  0.4
310 : loss:  1.0602534 	 acc:  0.4
312 : loss:  1.059215 	 acc:  0.38
314 : loss:  1.042396 	 acc:  0.4
316 : loss:  1.0535512 	 acc:  0.45
318 : loss:  1.0322118 	 acc:  0.39
320 : loss:  0.9718964 	 acc:  0.51
322 : loss:  0.9944643 	 acc:  0.48
324 : loss:  1.0319474 	 acc:  0.46
326 : loss:  1.0075314 	 acc:  0.45
328 : loss:  0.99854165 	 acc:  0.53
330 : loss:  1.0563626 	 acc:  0.39
332 : loss:  1.0166445 	 acc:  0.47
334 : loss:  1.0078381 	 acc:  0.48
336 : loss:  1.035478 	 acc:  0.48
338 : loss:  1.0704546 	 acc:  0.41
340 : loss:  1.0055712 	 acc:  0.52
342 : loss:  1.0441877 	 acc:  0.45
344 : loss:  1.0453974 	 acc:  0.41
346 : loss:  1.0696113 	 acc:  0.36
348 : loss:  1.0583675 	 acc:  0.48
350 : loss:  1.0139023 	 acc:  0.45
352 : loss:  1.0545641 	 acc:  0.43
354 : loss:  1.003021 	 acc:  0.49
356 : loss:  1.0774701 	 acc:  0.38
358 : loss:  0.98280805 	 acc:  0.56
360 : loss:  1.0409455 	 acc:  0.45
362 : loss:  1.0458859 	 acc:  0.38
364 : loss:  1.0704612 	 acc:  0.4
366 : loss:  1.0678791 	 acc:  0.4
368 : loss:  1.0400312 	 acc:  0.44
370 : loss:  1.0537788 	 acc:  0.37
/home/yannik/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)

100 	 [50.40703689 50.83498132  0.82404161]
0 	val accuracy:  0.42360002 	 f_! score:  [0.50407037 0.50834981 0.00824042]

100 	 [50.40703689 50.83498132  0.82404161]
100 	 [46.04993984 39.77457039 14.5       ]
100 	 [56.3574389  71.36861345  0.42451415]
100 	 [3317. 3289. 3394.]
---train_last_layer Test  Twitter ---
0.42360002
f1:  [0.50407037 0.50834981 0.00824042]
(7733, 3)
7733
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-09-05 16:11:26.426255: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 16:11:26.426291: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 16:11:26.426298: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-09-05 16:11:26.426306: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-09-05 16:11:26.426388: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2957 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
models/CNN/pretrained_model.ckpt-31380
2018-09-05 16:11:29.303462: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 16:11:29.303504: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 16:11:29.303510: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-09-05 16:11:29.303514: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-09-05 16:11:29.303603: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2957 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
0 : loss:  0.9665732 	 acc:  0.65
2 : loss:  1.051209 	 acc:  0.48
4 : loss:  0.9157106 	 acc:  0.66
6 : loss:  0.7191437 	 acc:  0.76
8 : loss:  1.2641932 	 acc:  0.52
10 : loss:  1.2129931 	 acc:  0.55
12 : loss:  1.002443 	 acc:  0.68
14 : loss:  0.905644 	 acc:  0.7
16 : loss:  1.3985972 	 acc:  0.44
18 : loss:  0.9364524 	 acc:  0.59
2018-09-05 16:11:45.751742: W tensorflow/core/common_runtime/bfc_allocator.cc:219] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.39GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
20 : loss:  0.869412 	 acc:  0.67
22 : loss:  0.72245485 	 acc:  0.82
24 : loss:  0.94448996 	 acc:  0.61
26 : loss:  0.8279095 	 acc:  0.71
28 : loss:  0.78158045 	 acc:  0.71
30 : loss:  0.8158514 	 acc:  0.7
32 : loss:  0.63323146 	 acc:  0.83
34 : loss:  0.66364944 	 acc:  0.81
36 : loss:  1.06161 	 acc:  0.5
38 : loss:  0.9817657 	 acc:  0.56
40 : loss:  0.958646 	 acc:  0.61

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
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-09-05 16:12:01.107881: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 16:12:01.107925: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 16:12:01.107934: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-09-05 16:12:01.107941: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-09-05 16:12:01.108039: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2957 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
models/CNN/pretrained_model.ckpt-31380
2018-09-05 16:12:04.849224: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 16:12:04.849270: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 16:12:04.849279: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-09-05 16:12:04.849286: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-09-05 16:12:04.849378: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2957 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
0 : loss:  0.9026464 	 acc:  0.89
2 : loss:  0.6966409 	 acc:  0.88
4 : loss:  0.60655963 	 acc:  0.86
6 : loss:  1.525804 	 acc:  0.4
8 : loss:  0.4345617 	 acc:  0.91
10 : loss:  0.9571215 	 acc:  0.72
12 : loss:  0.77179044 	 acc:  0.76
14 : loss:  1.4103668 	 acc:  0.52
16 : loss:  0.5616497 	 acc:  0.85
18 : loss:  0.6702127 	 acc:  0.8
20 : loss:  0.87018937 	 acc:  0.71
22 : loss:  0.44738194 	 acc:  0.92
24 : loss:  0.8905152 	 acc:  0.66
26 : loss:  0.7297379 	 acc:  0.76
28 : loss:  0.56512934 	 acc:  0.87
30 : loss:  0.7762583 	 acc:  0.74
32 : loss:  0.76505584 	 acc:  0.73
34 : loss:  0.7335447 	 acc:  0.76
36 : loss:  0.6068779 	 acc:  0.82
38 : loss:  1.1801121 	 acc:  0.61
40 : loss:  0.8794681 	 acc:  0.71
42 : loss:  0.633041 	 acc:  0.82
44 : loss:  0.3003211 	 acc:  0.96
46 : loss:  0.93420625 	 acc:  0.67
48 : loss:  0.8365407 	 acc:  0.69
50 : loss:  0.4488403 	 acc:  0.9
52 : loss:  0.77370125 	 acc:  0.73
54 : loss:  0.52571577 	 acc:  0.86
56 : loss:  0.6850367 	 acc:  0.77
58 : loss:  0.48850226 	 acc:  0.87
60 : loss:  0.41422775 	 acc:  0.91
62 : loss:  0.38374123 	 acc:  0.91
64 : loss:  0.4583483 	 acc:  0.88
66 : loss:  1.3055334 	 acc:  0.53
68 : loss:  0.6996027 	 acc:  0.78
70 : loss:  0.62540126 	 acc:  0.81
72 : loss:  0.7065625 	 acc:  0.77
74 : loss:  0.32630837 	 acc:  0.96
76 : loss:  1.1042377 	 acc:  0.56
78 : loss:  1.0587447 	 acc:  0.49
80 : loss:  0.54818034 	 acc:  0.87
82 : loss:  0.54797214 	 acc:  0.87
84 : loss:  0.91684806 	 acc:  0.6
86 : loss:  0.9240343 	 acc:  0.67
88 : loss:  0.5621736 	 acc:  0.84
90 : loss:  0.45463666 	 acc:  0.9
92 : loss:  1.0428624 	 acc:  0.61
94 : loss:  0.68618554 	 acc:  0.78
96 : loss:  0.5199103 	 acc:  0.85
98 : loss:  0.92914426 	 acc:  0.69
100 : loss:  0.5690194 	 acc:  0.81
102 : loss:  0.46914372 	 acc:  0.86
104 : loss:  0.58151203 	 acc:  0.84
106 : loss:  1.0326853 	 acc:  0.61
108 : loss:  0.49351683 	 acc:  0.86
110 : loss:  0.4623692 	 acc:  0.87
112 : loss:  0.74854493 	 acc:  0.73
114 : loss:  0.64705205 	 acc:  0.78
116 : loss:  0.69021624 	 acc:  0.75
118 : loss:  0.4286427 	 acc:  0.9
120 : loss:  0.87930167 	 acc:  0.7
122 : loss:  0.862677 	 acc:  0.65384614
124 : loss:  0.5926274 	 acc:  0.84
126 : loss:  0.44684058 	 acc:  0.9
128 : loss:  0.36748773 	 acc:  0.92
130 : loss:  0.54716796 	 acc:  0.83
132 : loss:  0.57517207 	 acc:  0.84
134 : loss:  0.8032072 	 acc:  0.72
136 : loss:  0.5791835 	 acc:  0.8
138 : loss:  0.84965056 	 acc:  0.7
140 : loss:  0.81416535 	 acc:  0.71
142 : loss:  0.85888296 	 acc:  0.63
144 : loss:  0.5622322 	 acc:  0.84
146 : loss:  0.63495016 	 acc:  0.8
148 : loss:  0.55133694 	 acc:  0.84
150 : loss:  0.51151824 	 acc:  0.85
152 : loss:  0.55155694 	 acc:  0.84
154 : loss:  0.6656372 	 acc:  0.76
156 : loss:  0.5121506 	 acc:  0.86
158 : loss:  0.5628576 	 acc:  0.84
160 : loss:  0.5198136 	 acc:  0.84
162 : loss:  0.3080004 	 acc:  0.96
164 : loss:  0.7799419 	 acc:  0.72
166 : loss:  0.9070225 	 acc:  0.68
168 : loss:  0.4233915 	 acc:  0.92
170 : loss:  0.691558 	 acc:  0.78
172 : loss:  0.74277174 	 acc:  0.73
174 : loss:  0.65389425 	 acc:  0.78
176 : loss:  0.40770447 	 acc:  0.92
178 : loss:  0.6536137 	 acc:  0.8
180 : loss:  0.6270018 	 acc:  0.8
182 : loss:  0.7440314 	 acc:  0.76
184 : loss:  1.0704424 	 acc:  0.59
186 : loss:  0.6842133 	 acc:  0.78
188 : loss:  0.71196884 	 acc:  0.74
190 : loss:  0.87232 	 acc:  0.65
192 : loss:  0.4308059 	 acc:  0.9
194 : loss:  0.4451963 	 acc:  0.88
196 : loss:  0.8556624 	 acc:  0.69
198 : loss:  0.58396155 	 acc:  0.83
200 : loss:  0.5576896 	 acc:  0.84
(2,)
(2,)

68 	 [ 0.          0.         61.49417912]
0 	val accuracy:  0.8316667 	 f_! score:  [0.         0.         0.90432616]

68 	 [ 0.          0.         61.49417912]
68 	 [ 0.    0.   56.42]
68 	 [ 0.  0. 68.]
68 	 [ 501.  657. 5642.]
---train_last_layer Test  Prime ---
0.8316667
f1:  [0.         0.         0.90432616]


  Twitter
delete old models
65730
Tensor("ConvNet/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("ConvNet_1/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   CNN ---
2018-09-05 16:12:44.812549: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 16:12:44.812586: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 16:12:44.812593: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-09-05 16:12:44.812599: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-09-05 16:12:44.812679: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2957 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/CNN/pretrained_model.ckpt-0

Saving...
saved to models/CNN/pretrained_model.ckpt-372

100 	 [63.50078078 59.20260691 21.64615365]
0 	val accuracy:  0.52199996 	 f_! score:  [0.63500781 0.59202607 0.21646154]


Saving...
saved to models/CNN/pretrained_model.ckpt-744

100 	 [66.89621409 62.73540662 44.29035724]
1 	val accuracy:  0.5995 	 f_! score:  [0.66896214 0.62735407 0.44290357]


Saving...
saved to models/CNN/pretrained_model.ckpt-1116

100 	 [65.75995438 63.64771898 52.70119843]
2 	val accuracy:  0.6143 	 f_! score:  [0.65759954 0.63647719 0.52701198]


Saving...
saved to models/CNN/pretrained_model.ckpt-1488

100 	 [68.25356397 63.66872783 54.52215722]
3 	val accuracy:  0.6282 	 f_! score:  [0.68253564 0.63668728 0.54522157]


Saving...
saved to models/CNN/pretrained_model.ckpt-1860

100 	 [69.50757601 61.08940273 55.55450254]
4 	val accuracy:  0.62740004 	 f_! score:  [0.69507576 0.61089403 0.55554503]


Saving...
saved to models/CNN/pretrained_model.ckpt-2232

100 	 [67.67314932 62.65639944 56.44424433]
5 	val accuracy:  0.626 	 f_! score:  [0.67673149 0.62656399 0.56444244]


Saving...
saved to models/CNN/pretrained_model.ckpt-2604

100 	 [67.5697069  65.25469615 54.54967825]
6 	val accuracy:  0.6316 	 f_! score:  [0.67569707 0.65254696 0.54549678]

100 	 [67.5697069  65.25469615 54.54967825]
100 	 [71.29989567 57.26705143 64.17680834]
100 	 [64.65047735 76.62266677 48.1413027 ]
100 	 [3339. 3341. 3320.]
--- Test   Twitter ---
0.6316
f1:  [0.67569707 0.65254696 0.54549678]
/home/yannik/PycharmProjects/bachelorarbeit/TestHelper.py:112: DtypeWarning: Columns (1) have mixed types. Specify dtype option on import or set low_memory=False.
  input_data = self.decide_which_input(input_name)
509411
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-09-05 16:16:34.434876: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 16:16:34.434917: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 16:16:34.434924: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-09-05 16:16:34.434928: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-09-05 16:16:34.435023: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2957 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
models/CNN/pretrained_model.ckpt-2604
2018-09-05 16:16:38.133309: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 16:16:38.133352: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 16:16:38.133360: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-09-05 16:16:38.133364: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-09-05 16:16:38.133457: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2957 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
0 : loss:  1.1123284 	 acc:  0.28
100 : loss:  1.0677576 	 acc:  0.42
200 : loss:  1.0856547 	 acc:  0.42
300 : loss:  1.0910704 	 acc:  0.43
400 : loss:  1.0706341 	 acc:  0.44
500 : loss:  1.1145902 	 acc:  0.34
600 : loss:  1.1192783 	 acc:  0.36
700 : loss:  1.0690887 	 acc:  0.47
800 : loss:  1.0790243 	 acc:  0.43
900 : loss:  1.09012 	 acc:  0.4
1000 : loss:  1.087841 	 acc:  0.37
1100 : loss:  1.0871806 	 acc:  0.43
1200 : loss:  1.079141 	 acc:  0.39
1300 : loss:  1.0820851 	 acc:  0.38
1400 : loss:  1.0570239 	 acc:  0.46
1500 : loss:  1.0695783 	 acc:  0.41
1600 : loss:  1.0968593 	 acc:  0.37
1700 : loss:  1.1100452 	 acc:  0.37
1800 : loss:  1.0840164 	 acc:  0.42
1900 : loss:  1.0873302 	 acc:  0.4
2000 : loss:  1.0766748 	 acc:  0.4
2100 : loss:  1.0661905 	 acc:  0.45
2200 : loss:  1.0874556 	 acc:  0.45
2300 : loss:  1.0861567 	 acc:  0.43
2400 : loss:  1.0697646 	 acc:  0.42
2500 : loss:  1.0858938 	 acc:  0.39
2600 : loss:  1.0530921 	 acc:  0.47
2700 : loss:  1.0844014 	 acc:  0.41
2800 : loss:  1.0818633 	 acc:  0.41
2900 : loss:  1.0774326 	 acc:  0.42
3000 : loss:  1.070092 	 acc:  0.41
3100 : loss:  1.0246147 	 acc:  0.53
3200 : loss:  1.1130711 	 acc:  0.34
3300 : loss:  1.1069212 	 acc:  0.39
3400 : loss:  1.096354 	 acc:  0.37
3500 : loss:  1.0593663 	 acc:  0.46
3600 : loss:  1.0414422 	 acc:  0.45
3700 : loss:  1.0502748 	 acc:  0.43
3800 : loss:  1.0852654 	 acc:  0.42
3900 : loss:  1.0632745 	 acc:  0.45
4000 : loss:  1.0721551 	 acc:  0.52
4100 : loss:  1.0245863 	 acc:  0.51
4200 : loss:  1.0871923 	 acc:  0.45
4300 : loss:  1.07587 	 acc:  0.42
4400 : loss:  1.095188 	 acc:  0.42
4500 : loss:  1.0582188 	 acc:  0.48

500 	 [269.16145395   0.81438169 207.79535605]
0 	val accuracy:  0.4283 	 f_! score:  [0.53832291 0.00162876 0.41559071]

500 	 [269.16145395   0.81438169 207.79535605]
500 	 [204.78548667  12.         239.91183012]
500 	 [397.36951354   0.42166834 186.43045352]
500 	 [18794. 13823. 17383.]
---train_last_layer Test  Review ---
0.4283
f1:  [0.53832291 0.00162876 0.41559071]
(7733, 3)
7733
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-09-05 16:20:14.085045: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 16:20:14.085419: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 16:20:14.085427: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-09-05 16:20:14.085434: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-09-05 16:20:14.085527: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2957 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
models/CNN/pretrained_model.ckpt-2604
2018-09-05 16:20:17.962293: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 16:20:17.962338: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 16:20:17.962345: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-09-05 16:20:17.962353: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-09-05 16:20:17.962463: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2957 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
0 : loss:  1.1198797 	 acc:  0.37
2 : loss:  0.9979722 	 acc:  0.55
4 : loss:  0.5331305 	 acc:  0.88
6 : loss:  0.6011443 	 acc:  0.84
8 : loss:  1.1013422 	 acc:  0.59
10 : loss:  0.978566 	 acc:  0.67
12 : loss:  1.0748838 	 acc:  0.61
14 : loss:  0.5414706 	 acc:  0.86
16 : loss:  0.7158633 	 acc:  0.82
18 : loss:  0.9166792 	 acc:  0.64
20 : loss:  0.7258603 	 acc:  0.83
22 : loss:  0.9591727 	 acc:  0.57
24 : loss:  0.88262314 	 acc:  0.64
26 : loss:  1.0935161 	 acc:  0.46
28 : loss:  1.2047704 	 acc:  0.4
30 : loss:  0.88429314 	 acc:  0.65
32 : loss:  0.8325719 	 acc:  0.69
34 : loss:  0.58282846 	 acc:  0.88
36 : loss:  0.9632584 	 acc:  0.61
38 : loss:  0.59267896 	 acc:  0.85
40 : loss:  1.02378 	 acc:  0.51

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
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-09-05 16:20:49.424844: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 16:20:49.424885: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 16:20:49.424893: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-09-05 16:20:49.424899: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-09-05 16:20:49.424991: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2957 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
models/CNN/pretrained_model.ckpt-2604
2018-09-05 16:20:52.438754: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-09-05 16:20:52.438795: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-09-05 16:20:52.438802: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-09-05 16:20:52.438808: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-09-05 16:20:52.438897: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2957 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
0 : loss:  0.8817533 	 acc:  0.87
2 : loss:  0.59742564 	 acc:  0.92
4 : loss:  1.0659401 	 acc:  0.58
6 : loss:  0.99335045 	 acc:  0.72
8 : loss:  0.38642472 	 acc:  0.93
10 : loss:  0.45730042 	 acc:  0.9
12 : loss:  0.4332334 	 acc:  0.9
14 : loss:  1.0131259 	 acc:  0.69
16 : loss:  0.7539569 	 acc:  0.76
18 : loss:  0.9039644 	 acc:  0.67
20 : loss:  0.80322206 	 acc:  0.71
22 : loss:  0.67421913 	 acc:  0.79
24 : loss:  0.71943724 	 acc:  0.76
26 : loss:  0.7442223 	 acc:  0.76
28 : loss:  0.8849948 	 acc:  0.66
30 : loss:  0.58439606 	 acc:  0.87
32 : loss:  0.48433506 	 acc:  0.92
34 : loss:  0.46310714 	 acc:  0.9
36 : loss:  0.3668709 	 acc:  0.93
38 : loss:  0.5629347 	 acc:  0.84
40 : loss:  1.0468078 	 acc:  0.6
42 : loss:  0.56588596 	 acc:  0.84
44 : loss:  0.9604746 	 acc:  0.67
46 : loss:  0.28961137 	 acc:  0.95
48 : loss:  0.54902005 	 acc:  0.84
50 : loss:  0.49168023 	 acc:  0.86
52 : loss:  0.57617784 	 acc:  0.82
54 : loss:  0.4031427 	 acc:  0.9
56 : loss:  0.42579007 	 acc:  0.89
58 : loss:  0.5930471 	 acc:  0.81
60 : loss:  0.39638412 	 acc:  0.9
62 : loss:  0.79483724 	 acc:  0.73
64 : loss:  0.52948135 	 acc:  0.85
66 : loss:  0.7365744 	 acc:  0.74
68 : loss:  0.79410625 	 acc:  0.67
70 : loss:  0.8373966 	 acc:  0.68
72 : loss:  1.0160061 	 acc:  0.6
74 : loss:  0.7057573 	 acc:  0.76
76 : loss:  0.49001026 	 acc:  0.87
78 : loss:  0.5777435 	 acc:  0.84
80 : loss:  0.5901779 	 acc:  0.82
82 : loss:  0.5026907 	 acc:  0.86
84 : loss:  0.7460331 	 acc:  0.76
86 : loss:  0.96692634 	 acc:  0.63
88 : loss:  0.6874379 	 acc:  0.78
90 : loss:  0.93730026 	 acc:  0.68
92 : loss:  0.8408807 	 acc:  0.65384614
94 : loss:  0.29696083 	 acc:  0.96
96 : loss:  0.62627715 	 acc:  0.8
98 : loss:  0.37771815 	 acc:  0.92
100 : loss:  0.5189465 	 acc:  0.84
102 : loss:  0.38986504 	 acc:  0.95
104 : loss:  0.8115456 	 acc:  0.72
106 : loss:  0.8798096 	 acc:  0.65
108 : loss:  0.5849297 	 acc:  0.83
110 : loss:  0.48174998 	 acc:  0.89
112 : loss:  0.738144 	 acc:  0.75
114 : loss:  0.63342005 	 acc:  0.8
116 : loss:  0.57702196 	 acc:  0.83
118 : loss:  0.7145292 	 acc:  0.76
120 : loss:  0.44202414 	 acc:  0.89
122 : loss:  1.0558791 	 acc:  0.59
124 : loss:  0.45905566 	 acc:  0.88
126 : loss:  0.6734774 	 acc:  0.78
128 : loss:  0.76988715 	 acc:  0.73
130 : loss:  0.686085 	 acc:  0.76
132 : loss:  0.8862638 	 acc:  0.66
134 : loss:  0.7949252 	 acc:  0.69
136 : loss:  0.6046255 	 acc:  0.8
138 : loss:  0.92648256 	 acc:  0.66
140 : loss:  0.6130038 	 acc:  0.81
142 : loss:  0.9439094 	 acc:  0.67
144 : loss:  0.593352 	 acc:  0.81
146 : loss:  0.56122327 	 acc:  0.83
148 : loss:  0.68713266 	 acc:  0.78
150 : loss:  0.8121984 	 acc:  0.7
152 : loss:  0.5956754 	 acc:  0.83
154 : loss:  1.039613 	 acc:  0.57
156 : loss:  0.41204202 	 acc:  0.9
158 : loss:  0.5353375 	 acc:  0.84
160 : loss:  0.4679391 	 acc:  0.87
162 : loss:  0.5732428 	 acc:  0.84
164 : loss:  0.5433869 	 acc:  0.85
166 : loss:  0.5119314 	 acc:  0.83
168 : loss:  0.4223951 	 acc:  0.92
170 : loss:  1.2451086 	 acc:  0.48
172 : loss:  0.81727785 	 acc:  0.71
174 : loss:  0.5836803 	 acc:  0.84
176 : loss:  0.82187164 	 acc:  0.67
178 : loss:  0.97589874 	 acc:  0.63
180 : loss:  0.43162754 	 acc:  0.91
182 : loss:  0.765887 	 acc:  0.73
184 : loss:  0.49370262 	 acc:  0.86
186 : loss:  0.6239296 	 acc:  0.8
188 : loss:  0.5540993 	 acc:  0.83
190 : loss:  1.2727522 	 acc:  0.48
192 : loss:  0.5156015 	 acc:  0.85
194 : loss:  0.7153849 	 acc:  0.78
196 : loss:  0.6544373 	 acc:  0.77
198 : loss:  0.56308603 	 acc:  0.86
200 : loss:  0.4865578 	 acc:  0.88
(2,)
(2,)

68 	 [ 0.          0.         61.49417912]
0 	val accuracy:  0.8316667 	 f_! score:  [0.         0.         0.90432616]

68 	 [ 0.          0.         61.49417912]
68 	 [ 0.    0.   56.42]
68 	 [ 0.  0. 68.]
68 	 [ 501.  657. 5642.]
---train_last_layer Test  Prime ---
0.8316667
f1:  [0.         0.         0.90432616]


"""