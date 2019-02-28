import matplotlib.pyplot as plt

from TestHelper import TestHelper


net_name = "WITHOUT_EMBED"

testhelper = TestHelper()
review_loss, review_acc = testhelper.train_input("Review", net_name)

testhelper.just_test("Twitter", net_name)
testhelper.just_test("Medical", net_name)
testhelper.just_test("Prime", net_name)

tw_loss, tw_acc = testhelper.train_input("Twitter", net_name)

testhelper.just_test("Review", net_name)
testhelper.just_test("Medical", net_name)
testhelper.just_test("Prime", net_name)

med_loss, med_acc = testhelper.train_input("Medical", net_name)

testhelper.just_test("Review", net_name)
testhelper.just_test("Twitter", net_name)
testhelper.just_test("Prime", net_name)

prime_loss, prime_acc = testhelper.train_input("Prime", net_name)

testhelper.just_test("Review", net_name)
testhelper.just_test("Twitter", net_name)
testhelper.just_test("Medical", net_name)

def plot_result(a, b, c, d):
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(a)
    axarr[0, 0].set_title('Twitter')
    axarr[0, 1].plot(b)
    axarr[0, 1].set_title('Prime Video')
    axarr[1, 0].plot(c)
    axarr[1, 0].set_title('Amazon Review')
    axarr[1, 1].plot(d)
    axarr[1, 1].set_title('Medikamente')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                        wspace=0.35)
    plt.show()


plot_result(tw_loss, prime_loss, review_loss, med_loss)
plot_result(tw_acc, prime_acc, review_acc, med_acc)

"""
Output:

Review
delete old models
/home/yannik/PycharmProjects/bachelorarbeit/TestHelper.py:39: DtypeWarning: Columns (1) have mixed types. Specify dtype option on import or set low_memory=False.
  input_data = self.decide_which_input(input_name)
509411
Tensor("ConvNet/dense_1/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("ConvNet_1/dense_1/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   WITHOUT_EMBED ---
2018-08-30 16:52:18.807953: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-08-30 16:52:18.904479: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-08-30 16:52:18.904836: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 970 major: 5 minor: 2 memoryClockRate(GHz): 1.253
pciBusID: 0000:01:00.0
totalMemory: 3.94GiB freeMemory: 3.34GiB
2018-08-30 16:52:18.904853: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 16:52:19.617452: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 16:52:19.617491: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 16:52:19.617499: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 16:52:19.617815: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-0

Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-4482
/home/yannik/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)

500 	 [2.72329079e+02 0.00000000e+00 1.09609610e-01]
0 	val accuracy:  0.37590003 	 f_! score:  [5.44658159e-01 0.00000000e+00 2.19219219e-04]


Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-8965

500 	 [272.32631982   0.           0.        ]
1 	val accuracy:  0.37588 	 f_! score:  [0.54465264 0.         0.        ]


Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-13448

500 	 [272.32631982   0.           0.        ]
2 	val accuracy:  0.37588 	 f_! score:  [0.54465264 0.         0.        ]


Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-17931

500 	 [272.32631982   0.           0.        ]
3 	val accuracy:  0.37588 	 f_! score:  [0.54465264 0.         0.        ]

500 	 [272.32631982   0.           0.        ]
500 	 [187.94   0.     0.  ]
500 	 [500.   0.   0.]
500 	 [18794. 13823. 17383.]
--- Test   Review ---
0.37588
f1:  [0.54465264 0.         0.        ]
65730
Vorsicht! es muss vorher ein  WITHOUT_EMBED  Netz abgespeichert worden sein
2018-08-30 17:05:30.402228: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:05:30.402283: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:05:30.402290: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:05:30.402294: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:05:30.402375: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:05:30.562378: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:05:30.562427: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:05:30.562438: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:05:30.562446: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:05:30.562565: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

101 	 [ 0.09756098  0.05714286 49.73881901]
val accuracy:  0.32871285 	 f_! score:  [0.00096595 0.00056577 0.49246355]

101 	 [ 0.09756098  0.05714286 49.73881901]
101 	 [ 0.4         1.         33.18851064]
101 	 [5.55555556e-02 2.94117647e-02 1.00966667e+02]
101 	 [3386. 3396. 3318.]
---just Test  Twitter ---
0.32871285
f1:  [0.00096595 0.00056577 0.49246355]
(7733, 3)
7733
Vorsicht! es muss vorher ein  WITHOUT_EMBED  Netz abgespeichert worden sein
2018-08-30 17:05:33.084209: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:05:33.084268: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:05:33.084278: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:05:33.084286: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:05:33.084369: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:05:33.242494: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:05:33.242541: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:05:33.242551: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:05:33.242558: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:05:33.242660: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

15 	 [0.125      2.44726433 0.02857143]
val accuracy:  0.091333315 	 f_! score:  [0.00833333 0.16315096 0.00190476]

15 	 [0.125      2.44726433 0.02857143]
15 	 [1.         1.35892677 0.5       ]
15 	 [6.66666667e-02 1.49500000e+01 1.47058824e-02]
15 	 [ 117.  136. 1247.]
---just Test  Medical ---
0.091333315
f1:  [0.00833333 0.16315096 0.00190476]
(37126,)
(37126,)
37126
Vorsicht! es muss vorher ein  WITHOUT_EMBED  Netz abgespeichert worden sein
2018-08-30 17:05:41.381874: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:05:41.381919: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:05:41.381930: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:05:41.381937: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:05:41.382050: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:05:41.544434: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:05:41.544477: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:05:41.544487: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:05:41.544493: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:05:41.544593: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
/home/yannik/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)

71 	 [ 5.75726207 12.05736037  0.50685336]
val accuracy:  0.09802817 	 f_! score:  [0.0810882  0.16982198 0.00713878]

71 	 [ 5.75726207 12.05736037  0.50685336]
71 	 [6.15922035 6.84996749 2.92307692]
71 	 [ 7.28063079 64.60593054  0.33061007]
71 	 [ 506.  681. 5913.]
---just Test  Prime ---
0.09802817
f1:  [0.0810882  0.16982198 0.00713878]


  Twitter
delete old models
65730
Tensor("ConvNet/dense_1/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("ConvNet_1/dense_1/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   WITHOUT_EMBED ---
2018-08-30 17:05:48.219124: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:05:48.219158: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:05:48.219164: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:05:48.219168: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:05:48.219244: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-0

Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-372

100 	 [ 3.96191611 49.70138895  2.45591279]
0 	val accuracy:  0.338 	 f_! score:  [0.03961916 0.49701389 0.02455913]


Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-744

100 	 [ 0.72434549 49.80370802  0.86604878]
1 	val accuracy:  0.3341 	 f_! score:  [0.00724345 0.49803708 0.00866049]


Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-1116

100 	 [ 1.43191466  1.26148065 49.22593163]
2 	val accuracy:  0.33019996 	 f_! score:  [0.01431915 0.01261481 0.49225932]


Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-1488

100 	 [ 0.16451969  0.86527065 49.14975023]
3 	val accuracy:  0.32770002 	 f_! score:  [0.0016452  0.00865271 0.4914975 ]

100 	 [ 0.16451969  0.86527065 49.14975023]
100 	 [ 3.         10.5        32.79687459]
100 	 [8.45928165e-02 4.56324020e-01 9.91857921e+01]
100 	 [3376. 3338. 3286.]
--- Test   Twitter ---
0.32770002
f1:  [0.0016452  0.00865271 0.4914975 ]
/home/yannik/PycharmProjects/bachelorarbeit/TestHelper.py:59: DtypeWarning: Columns (1) have mixed types. Specify dtype option on import or set low_memory=False.
  input_data = self.decide_which_input(input_name)
509411
Vorsicht! es muss vorher ein  WITHOUT_EMBED  Netz abgespeichert worden sein
2018-08-30 17:07:40.297911: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:07:40.297947: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:07:40.297953: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:07:40.297957: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:07:40.298071: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:07:40.455987: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:07:40.456034: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:07:40.456042: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:07:40.456048: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:07:40.456151: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

501 	 [  3.67582193 213.53645523  37.50894838]
val accuracy:  0.2793014 	 f_! score:  [0.00733697 0.42622047 0.07486816]

501 	 [  3.67582193 213.53645523  37.50894838]
501 	 [ 59.95833333 138.4341645  167.58990023]
501 	 [  1.96153606 476.26483464  21.74994522]
501 	 [18834. 13850. 17416.]
---just Test  Review ---
0.2793014
f1:  [0.00733697 0.42622047 0.07486816]
(7733, 3)
7733
Vorsicht! es muss vorher ein  WITHOUT_EMBED  Netz abgespeichert worden sein
2018-08-30 17:07:47.777878: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:2707:47.777913: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:07:47.777919: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:07:47.777924: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:07:47.778003: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:07:47.933914: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:07:47.933958: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:07:47.933971: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:07:47.933981: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:07:47.934102: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

15 	 [2.00588348 0.06666667 0.12987013]
val accuracy:  0.07733332 	 f_! score:  [0.13372557 0.00444444 0.00865801]

15 	 [2.00588348 0.06666667 0.12987013]
15 	 [1.10681818 0.2        0.71428571]
15 	 [15.          0.04        0.07142857]
15 	 [ 110.  141. 1249.]
---just Test  Medical ---
0.07733332
f1:  [0.13372557 0.00444444 0.00865801]
(37126,)
(37126,)
37126
Vorsicht! es muss vorher ein  WITHOUT_EMBED  Netz abgespeichert worden sein
2018-08-30 17:07:55.940050: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:07:55.940094: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:07:55.940103: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:07:55.940111: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:07:55.940208: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:07:56.094641: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:07:56.094687: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:07:56.094695: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:07:56.094701: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:07:56.094803: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

71 	 [9.15602595 0.26666667 2.50717428]
val accuracy:  0.08591548 	 f_! score:  [0.12895811 0.00375587 0.03531231]

71 	 [9.15602595 0.26666667 2.50717428]
71 	 [ 5.0560885   1.         48.93482143]
71 	 [67.09615385  0.15384615  1.36497594]
71 	 [ 511.  677. 5912.]
---just Test  Prime ---
0.08591548
f1:  [0.12895811 0.00375587 0.03531231]


  Medical
delete old models
(7733, 3)
7733
Tensor("ConvNet/dense_1/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("ConvNet_1/dense_1/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   WITHOUT_EMBED ---
2018-08-30 17:07:58.380086: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:07:58.380117: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:07:58.380124: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:07:58.380129: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:07:58.380209: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-0

Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-42

14 	 [ 1.24858037  0.71880342 11.39505128]
0 	val accuracy:  0.68714285 	 f_! score:  [0.08918431 0.0513431  0.81393223]


Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-84

14 	 [ 0.          0.         12.68912671]
1 	val accuracy:  0.83 	 f_! score:  [0.         0.         0.90636619]


Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-126

14 	 [ 0.          0.13333333 12.73068026]
2 	val accuracy:  0.8371428 	 f_! score:  [0.         0.00952381 0.9093343 ]


Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-168

14 	 [ 0.          0.         12.74451851]
3 	val accuracy:  0.83857155 	 f_! score:  [0.         0.         0.91032275]

14 	 [ 0.          0.         12.74451851]
14 	 [ 0.          0.         11.78128221]
14 	 [ 0.          0.         13.93914665]
14 	 [ 105.  116. 1179.]
--- Test   Medical ---
0.83857155
f1:  [0.         0.         0.91032275]
509411
Vorsicht! es muss vorher ein  WITHOUT_EMBED  Netz abgespeichert worden sein
2018-08-30 17:08:43.321725: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:08:43.321763: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:08:43.321771: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:08:43.321777: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:08:43.321872: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:08:43.480721: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:08:43.480768: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:08:43.480778: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:08:43.480785: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:08:43.480924: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

501 	 [1.95833333e-01 0.00000000e+00 2.57649861e+02]
val accuracy:  0.34762472 	 f_! score:  [3.90884897e-04 0.00000000e+00 5.14271179e-01]

501 	 [1.95833333e-01 0.00000000e+00 2.57649861e+02]
501 	 [  1.5          0.         174.14224801]
501 	 [1.09181141e-01 0.00000000e+00 5.00935484e+02]
501 	 [18833. 13853. 17414.]
---just Test  Review ---
0.34762472
f1:  [3.90884897e-04 0.00000000e+00 5.14271179e-01]
65730
Vorsicht! es muss vorher ein  WITHOUT_EMBED  Netz abgespeichert worden sein
2018-08-30 17:08:55.133214: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:08:55.133251: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:08:55.133258: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:08:55.133263: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:08:55.133345: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:08:55.357822: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:08:55.357870: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:08:55.357879: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:08:55.357886: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:08:55.357992: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

101 	 [50.44613477  0.05714286  0.23076923]
val accuracy:  0.3350495 	 f_! score:  [0.49946668 0.00056577 0.00228484]

101 	 [50.44613477  0.05714286  0.23076923]
101 	 [33.83911392  0.16666667  0.4       ]
101 	 [1.00764706e+02 3.44827586e-02 1.62162162e-01]
101 	 [3385. 3398. 3317.]
---just Test  Twitter ---
0.3350495
f1:  [0.49946668 0.00056577 0.00228484]
(37126,)
(37126,)
37126
Vorsicht! es muss vorher ein  WITHOUT_EMBED  Netz abgespeichert worden sein
2018-08-30 17:09:04.253877: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:09:04.253913: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:09:04.253919: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:09:04.253924: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:09:04.254007: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:09:04.462624: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:09:04.462679: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:09:04.462693: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:09:04.462704: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:09:04.462834: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

71 	 [9.43181201 0.11111111 5.71621818]
val accuracy:  0.10577465 	 f_! score:  [0.13284242 0.00156495 0.08051012]

71 	 [9.43181201 0.11111111 5.71621818]
71 	 [ 5.26292384  0.16666667 59.68771777]
71 	 [65.8655202   0.08333333  3.15530584]
71 	 [ 521.  676. 5903.]
---just Test  Prime ---
0.10577465
f1:  [0.13284242 0.00156495 0.08051012]


  Prime
delete old models
(37126,)
(37126,)
37126
Tensor("ConvNet/dense_1/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("ConvNet_1/dense_1/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   WITHOUT_EMBED ---
2018-08-30 17:09:13.642468: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:09:13.642503: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:09:13.642510: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:09:13.642517: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:09:13.642597: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-0

Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-201
(2,)
(2,)

68 	 [ 0.          0.         61.49417912]
0 	val accuracy:  0.83166665 	 f_! score:  [0.         0.         0.90432616]


Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-402
(2,)

69 	 [ 0.          0.         62.46302693]
1 	val accuracy:  0.83230215 	 f_! score:  [0.         0.         0.90526126]


Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-603
(2,)

69 	 [ 0.          0.4        62.46616226]
2 	val accuracy:  0.832446 	 f_! score:  [0.        0.0057971 0.9053067]


Saving...
saved to models/WITHOUT_EMBED/pretrained_model.ckpt-804
(2,)

69 	 [ 0.          0.         62.46853809]
3 	val accuracy:  0.83244604 	 f_! score:  [0.         0.         0.90534113]

69 	 [ 0.          0.         62.46853809]
69 	 [ 0.          0.         57.37959596]
69 	 [ 0.          0.         68.98958333]
69 	 [ 501.  661. 5738.]
--- Test   Prime ---
0.83244604
f1:  [0.         0.         0.90534113]
509411
Vorsicht! es muss vorher ein  WITHOUT_EMBED  Netz abgespeichert worden sein
2018-08-30 17:10:30.649729: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:10:30.649768: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:10:30.649774: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:10:30.649779: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:10:30.649886: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:10:30.868577: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:10:30.868628: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:10:30.868637: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:10:30.868643: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:10:30.868754: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

501 	 [2.26393205e+02 6.06060606e-02 1.89229735e+02]
val accuracy:  0.36339322 	 f_! score:  [4.51882646e-01 1.20970181e-04 3.77704062e-01]

501 	 [2.26393205e+02 6.06060606e-02 1.89229735e+02]
501 	 [187.06698428   0.25       174.95608575]
501 	 [2.90636978e+02 3.44827586e-02 2.09329195e+02]
501 	 [18825. 13852. 17423.]
---just Test  Review ---
0.36339322
f1:  [4.51882646e-01 1.20970181e-04 3.77704062e-01]
65730
Vorsicht! es muss vorher ein  WITHOUT_EMBED  Netz abgespeichert worden sein
2018-08-30 17:10:42.601637: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:10:42.601675: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:10:42.601682: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:10:42.601687: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:10:42.601780: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:10:42.760877: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:10:42.760930: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:10:42.760940: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:10:42.760947: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:10:42.761064: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

101 	 [49.76691605  0.16326531  0.1754386 ]
val accuracy:  0.32881188 	 f_! score:  [0.49274174 0.00161649 0.00173702]

101 	 [49.76691605  0.16326531  0.1754386 ]
101 	 [33.25333333  0.28571429  0.19230769]
101 	 [100.58823529   0.11428571   0.16129032]
101 	 [3326. 3361. 3413.]
---just Test  Twitter ---
0.32881188
f1:  [0.49274174 0.00161649 0.00173702]
(7733, 3)
7733
Vorsicht! es muss vorher ein  WITHOUT_EMBED  Netz abgespeichert worden sein
2018-08-30 17:10:44.902198: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:10:44.902231: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:10:44.902237: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:10:44.902243: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:10:44.902322: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:10:45.059005: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:10:45.059049: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:10:45.059058: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:10:45.059065: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:10:45.059164: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3050 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

15 	 [ 0.225       0.         13.55752055]
val accuracy:  0.82933336 	 f_! score:  [0.015     0.        0.9038347]

15 	 [ 0.225       0.         13.55752055]
15 	 [ 2.          0.         12.44706029]
15 	 [ 0.11929825  0.         14.96923077]
15 	 [ 124.  132. 1244.]
---just Test  Medical ---
0.82933336
f1:  [0.015     0.        0.9038347]

"""