import matplotlib.pyplot as plt

from TestHelper import TestHelper


net_name = "CNN"

testhelper = TestHelper()
# review_loss, review_acc = testhelper.train_input("Review", net_name)

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
Tensor("ConvNet/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("ConvNet_1/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   CNN ---
2018-08-30 17:15:25.828566: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-08-30 17:15:25.908114: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-08-30 17:15:25.908509: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 970 major: 5 minor: 2 memoryClockRate(GHz): 1.253
pciBusID: 0000:01:00.0
totalMemory: 3.94GiB freeMemory: 3.33GiB
2018-08-30 17:15:25.908527: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:15:26.058315: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:15:26.058346: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:15:26.058351: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:15:26.058471: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
2018-08-30 17:15:33.258325: W tensorflow/core/framework/allocator.cc:108] Allocation of 367528800 exceeds 10% of system memory.

Saving...
saved to models/CNN/pretrained_model.ckpt-0
2018-08-30 17:15:46.961902: W tensorflow/core/common_runtime/bfc_allocator.cc:219] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.60GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.

Saving...
saved to models/CNN/pretrained_model.ckpt-4482

500 	 [346.72231987 248.81036615 373.66697718]
0 	val accuracy:  0.66080004 	 f_! score:  [0.69344464 0.49762073 0.74733395]

2018-08-30 17:19:10.690163: W tensorflow/core/framework/allocator.cc:108] Allocation of 367528800 exceeds 10% of system memory.

Saving...
saved to models/CNN/pretrained_model.ckpt-8965

500 	 [356.67992293 261.52982351 379.93036079]
1 	val accuracy:  0.6783 	 f_! score:  [0.71335985 0.52305965 0.75986072]

2018-08-30 17:22:26.355608: W tensorflow/core/framework/allocator.cc:108] Allocation of 367528800 exceeds 10% of system memory.

Saving...
saved to models/CNN/pretrained_model.ckpt-13448

500 	 [369.91587486 233.93415172 379.94035667]
2 	val accuracy:  0.6825 	 f_! score:  [0.73983175 0.4678683  0.75988071]

2018-08-30 17:25:40.801536: W tensorflow/core/framework/allocator.cc:108] Allocation of 367528800 exceeds 10% of system memory.









Saving...
saved to models/CNN/pretrained_model.ckpt-17931

500 	 [370.83726084 244.01883574 387.28227527]
3 	val accuracy:  0.69292 	 f_! score:  [0.74167452 0.48803767 0.77456455]

2018-08-30 17:28:53.282646: W tensorflow/core/framework/allocator.cc:108] Allocation of 367528800 exceeds 10% of system memory.

Saving...
saved to models/CNN/pretrained_model.ckpt-22414

500 	 [373.41201223 208.80719528 388.47805814]
4 	val accuracy:  0.6897 	 f_! score:  [0.74682402 0.41761439 0.77695612]


Saving...
saved to models/CNN/pretrained_model.ckpt-26897

500 	 [369.80644522 248.49226975 389.46930604]
5 	val accuracy:  0.69583994 	 f_! score:  [0.73961289 0.49698454 0.77893861]


Saving...
saved to models/CNN/pretrained_model.ckpt-31380

500 	 [366.17716112 261.69963599 388.27927772]
6 	val accuracy:  0.69409996 	 f_! score:  [0.73235432 0.52339927 0.77655856]

500 	 [366.17716112 261.69963599 388.27927772]
500 	 [380.30778774 267.96320166 373.24632793]
500 	 [355.77226319 260.21759398 407.13547597]
500 	 [18794. 13823. 17383.]
--- Test   Review ---
0.69409996
f1:  [0.73235432 0.52339927 0.77655856]
65730
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-08-30 17:38:38.724724: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:38:38.725408: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:38:38.725420: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:38:38.725425: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:38:38.726184: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:38:42.444643: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:38:42.444686: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:38:42.444692: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:38:42.444697: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:38:42.444789: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
/home/yannik/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)

101 	 [50.24601018  0.21818182  0.22641509]
val accuracy:  0.33376238 	 f_! score:  [0.49748525 0.00216022 0.00224173]

101 	 [50.24601018  0.21818182  0.22641509]
101 	 [33.70118644  0.25        0.35294118]
101 	 [100.48484848   0.19354839   0.16666667]
101 	 [3376. 3333. 3391.]
---just Test  Twitter ---
0.33376238
f1:  [0.49748525 0.00216022 0.00224173]
(7733, 3)
7733
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-08-30 17:38:57.948164: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:38:57.948198: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:38:57.948204: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:38:57.948208: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:38:57.948285: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:39:00.790166: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:39:00.790207: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:39:00.790214: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:39:00.790219: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:39:00.790305: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

15 	 [0.         2.27892961 0.32323232]
val accuracy:  0.092666656 	 f_! score:  [0.         0.15192864 0.02154882]

15 	 [0.         2.27892961 0.32323232]
15 	 [0.         1.24974359 0.8       ]
15 	 [ 0.         14.77777778  0.20253165]
15 	 [ 117.  125. 1258.]
---just Test  Medical ---
0.092666656
f1:  [0.         0.15192864 0.02154882]
(37126,)
(37126,)
37126
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-08-30 17:39:22.224058: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:39:22.224101: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:39:22.224109: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:39:22.224115: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:39:22.224208: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:39:25.071289: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:39:25.071332: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:39:25.071340: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:39:25.071346: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:39:25.071438: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
(2,)
(2,)

69 	 [ 0.         11.8921049   0.45397708]
val accuracy:  0.098071426 	 f_! score:  [0.         0.17234935 0.00657938]

69 	 [ 0.         11.8921049   0.45397708]
69 	 [0.         6.67142857 8.13157895]
69 	 [ 0.         68.54207071  0.26308064]
69 	 [ 518.  668. 5714.]
---just Test  Prime ---
0.098071426
f1:  [0.         0.17234935 0.00657938]


  Twitter
delete old models
65730
Tensor("ConvNet/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("ConvNet_1/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   CNN ---
2018-08-30 17:39:47.315443: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:39:47.315476: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:39:47.315482: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:39:47.315486: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:39:47.315564: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/CNN/pretrained_model.ckpt-0

Saving...
saved to models/CNN/pretrained_model.ckpt-372

100 	 [64.50459806 54.98030286 21.37180865]
0 	val accuracy:  0.5104 	 f_! score:  [0.64504598 0.54980303 0.21371809]


Saving...
saved to models/CNN/pretrained_model.ckpt-744

100 	 [65.80130576 59.85027187 42.71620599]
1 	val accuracy:  0.5765 	 f_! score:  [0.65801306 0.59850272 0.42716206]


Saving...
saved to models/CNN/pretrained_model.ckpt-1116

100 	 [67.52973095 63.58475505 46.17846546]
2 	val accuracy:  0.6056 	 f_! score:  [0.67529731 0.63584755 0.46178465]


Saving...
saved to models/CNN/pretrained_model.ckpt-1488

100 	 [68.5949841  60.62806127 57.91268167]
3 	val accuracy:  0.626 	 f_! score:  [0.68594984 0.60628061 0.57912682]


Saving...
saved to models/CNN/pretrained_model.ckpt-1860

100 	 [68.54647612 64.14093479 55.09838812]
4 	val accuracy:  0.6323 	 f_! score:  [0.68546476 0.64140935 0.55098388]


Saving...
saved to models/CNN/pretrained_model.ckpt-2232

100 	 [69.32732866 57.57573438 59.47320857]
5 	val accuracy:  0.6274 	 f_! score:  [0.69327329 0.57575734 0.59473209]


Saving...
saved to models/CNN/pretrained_model.ckpt-2604

100 	 [68.77075492 63.70829813 49.05954002]
6 	val accuracy:  0.61960006 	 f_! score:  [0.68770755 0.63708298 0.4905954 ]

100 	 [68.77075492 63.70829813 49.05954002]
100 	 [65.5600188  55.02921585 72.18658692]
100 	 [73.09027423 76.50020865 37.74666113]
100 	 [3302. 3251. 3447.]
--- Test   Twitter ---
0.61960006
f1:  [0.68770755 0.63708298 0.4905954 ]
/home/yannik/PycharmProjects/bachelorarbeit/TestHelper.py:59: DtypeWarning: Columns (1) have mixed types. Specify dtype option on import or set low_memory=False.
  input_data = self.decide_which_input(input_name)
509411
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-08-30 17:43:30.070459: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:43:30.070502: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:43:30.070510: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:43:30.070516: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:43:30.070614: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:43:33.153850: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:43:33.153892: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:43:33.153899: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:43:33.153903: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:43:33.153994: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

501 	 [3.83064516e-01 2.10526316e-01 2.57593855e+02]
val accuracy:  0.34764472 	 f_! score:  [7.64599833e-04 4.20212207e-04 5.14159392e-01]

501 	 [3.83064516e-01 2.10526316e-01 2.57593855e+02]
501 	 [  2.42105263   0.4        174.12310428]
501 	 [2.50562641e-01 1.42857143e-01 5.00689655e+02]
501 	 [18837. 13851. 17412.]
---just Test  Review ---
0.34764472
f1:  [7.64599833e-04 4.20212207e-04 5.14159392e-01]
(7733, 3)
7733
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-08-30 17:43:53.595143: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:43:53.596151: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:43:53.596159: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:43:53.596165: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:43:53.596248: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:43:56.987410: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:43:56.987453: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:43:56.987460: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:43:56.987464: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:43:56.987555: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

15 	 [2.24260697 0.43822844 0.08108108]
val accuracy:  0.084000014 	 f_! score:  [0.14950713 0.02921523 0.00540541]

15 	 [2.24260697 0.43822844 0.08108108]
15 	 [1.28111143 2.0625     1.        ]
15 	 [14.61515152  0.46904762  0.04225352]
15 	 [ 127.  123. 1250.]
---just Test  Medical ---
0.084000014
f1:  [0.14950713 0.02921523 0.00540541]
(37126,)
(37126,)
37126
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-08-30 17:44:17.439834: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:44:17.439875: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:44:17.439882: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:44:17.439887: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:44:17.439979: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:44:20.366884: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:44:20.366927: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:44:20.366933: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:44:20.366938: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:44:20.367024: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
(2,)
(2,)

69 	 [ 0.          0.6100713  62.28929652]
val accuracy:  0.82914287 	 f_! score:  [0.         0.00884161 0.90274343]

69 	 [ 0.          0.6100713  62.28929652]
69 	 [ 0.          2.15384615 57.38450154]
69 	 [ 0.          0.77678571 68.64819379]
69 	 [ 502.  664. 5734.]
---just Test  Prime ---
0.82914287
f1:  [0.         0.00884161 0.90274343]


  Medical
delete old models
(7733, 3)
7733
Tensor("ConvNet/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("ConvNet_1/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   CNN ---
2018-08-30 17:44:37.513190: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:44:37.513225: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:44:37.513231: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:44:37.513236: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:44:37.513313: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/CNN/pretrained_model.ckpt-0

Saving...
saved to models/CNN/pretrained_model.ckpt-42

14 	 [ 0.          0.         12.77481666]
0 	val accuracy:  0.8421429 	 f_! score:  [0.        0.        0.9124869]


Saving...
saved to models/CNN/pretrained_model.ckpt-84

14 	 [ 0.          0.         12.77481666]
1 	val accuracy:  0.84214294 	 f_! score:  [0.        0.        0.9124869]


Saving...
saved to models/CNN/pretrained_model.ckpt-126

14 	 [ 0.          0.         12.77481666]
2 	val accuracy:  0.84214294 	 f_! score:  [0.        0.        0.9124869]


Saving...
saved to models/CNN/pretrained_model.ckpt-168

14 	 [ 0.          0.         12.77481666]
3 	val accuracy:  0.84214294 	 f_! score:  [0.        0.        0.9124869]


Saving...
saved to models/CNN/pretrained_model.ckpt-210

14 	 [ 0.          0.         12.77481666]
4 	val accuracy:  0.84214294 	 f_! score:  [0.        0.        0.9124869]


Saving...
saved to models/CNN/pretrained_model.ckpt-252

14 	 [ 0.          0.         12.77481666]
5 	val accuracy:  0.84214294 	 f_! score:  [0.        0.        0.9124869]


Saving...
saved to models/CNN/pretrained_model.ckpt-294

14 	 [ 0.          0.         12.77481666]
6 	val accuracy:  0.84214294 	 f_! score:  [0.        0.        0.9124869]

14 	 [ 0.          0.         12.77481666]
14 	 [ 0.    0.   11.79]
14 	 [ 0.  0. 14.]
14 	 [ 105.  116. 1179.]
--- Test   Medical ---
0.84214294
f1:  [0.        0.        0.9124869]
509411
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-08-30 17:46:24.483072: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:46:24.483115: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:46:24.483123: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:46:24.483128: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:46:24.483220: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:46:27.963633: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:46:27.963676: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:46:27.963682: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:46:27.963686: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:46:27.963772: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

501 	 [2.64288983e+02 2.17391304e-01 7.25401393e+01]
val accuracy:  0.3720559 	 f_! score:  [5.27522920e-01 4.33914779e-04 1.44790697e-01]

501 	 [2.64288983e+02 2.17391304e-01 7.25401393e+01]
501 	 [187.63684604   0.29411765 175.1746954 ]
501 	 [4.52290824e+02 1.72413793e-01 4.68410784e+01]
501 	 [18831. 13852. 17417.]
---just Test  Review ---
0.3720559
f1:  [5.27522920e-01 4.33914779e-04 1.44790697e-01]
65730
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-08-30 17:46:52.863305: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:46:52.863347: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:46:52.863354: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:46:52.863359: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:46:52.863450: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:46:56.582330: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:46:56.582372: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:46:56.582378: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:46:56.582382: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:46:56.582465: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

101 	 [50.51937328  0.05405405  0.1       ]
val accuracy:  0.33534652 	 f_! score:  [0.50019181 0.00053519 0.0009901 ]

101 	 [50.51937328  0.05405405  0.1       ]
101 	 [33.87333333  0.25        0.33333333]
101 	 [1.00909091e+02 3.03030303e-02 5.88235294e-02]
101 	 [3387. 3289. 3424.]
---just Test  Twitter ---
0.33534652
f1:  [0.50019181 0.00053519 0.0009901 ]
(37126,)
(37126,)
37126
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-08-30 17:47:18.726349: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:47:18.726390: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:47:18.726398: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:47:18.726403: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:47:18.726498: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:47:21.587502: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:47:21.587544: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:47:21.587550: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:47:21.587555: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:47:21.587649: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
/home/yannik/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)

71 	 [9.64433735 7.96864542 0.09090909]
val accuracy:  0.07915493 	 f_! score:  [0.13583574 0.11223444 0.00128041]

71 	 [9.64433735 7.96864542 0.09090909]
71 	 [5.68939058 5.74655658 0.5       ]
71 	 [5.26485910e+01 1.79217356e+01 5.00000000e-02]
71 	 [ 546.  679. 5875.]
---just Test  Prime ---
0.07915493
f1:  [0.13583574 0.11223444 0.00128041]


  Prime
delete old models
(37126,)
(37126,)
37126
Tensor("ConvNet/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("ConvNet_1/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   CNN ---
2018-08-30 17:47:46.190688: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:47:46.190723: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:47:46.190729: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:47:46.190733: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:47:46.190813: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/CNN/pretrained_model.ckpt-0

Saving...
saved to models/CNN/pretrained_model.ckpt-201
(2,)
(2,)

68 	 [ 0.          0.         61.49417912]
0 	val accuracy:  0.83166665 	 f_! score:  [0.         0.         0.90432616]


Saving...
saved to models/CNN/pretrained_model.ckpt-402
(2,)
(2,)

68 	 [ 0.          0.         61.49417912]
1 	val accuracy:  0.8316667 	 f_! score:  [0.         0.         0.90432616]


Saving...
saved to models/CNN/pretrained_model.ckpt-603
(2,)
(2,)

68 	 [ 1.00555556  0.         61.51031059]
2 	val accuracy:  0.83152175 	 f_! score:  [0.01478758 0.         0.90456339]


Saving...
saved to models/CNN/pretrained_model.ckpt-804
(2,)

69 	 [12.60039336  0.         62.75045715]
3 	val accuracy:  0.8359712 	 f_! score:  [0.1826144  0.         0.90942692]


Saving...
saved to models/CNN/pretrained_model.ckpt-1005
(2,)

69 	 [11.6559057  0.        62.7597856]
4 	val accuracy:  0.8381295 	 f_! score:  [0.16892617 0.         0.90956211]


Saving...
saved to models/CNN/pretrained_model.ckpt-1206

70 	 [23.81956561  0.         63.24522176]
5 	val accuracy:  0.81357145 	 f_! score:  [0.34027951 0.         0.90350317]


Saving...
saved to models/CNN/pretrained_model.ckpt-1407
(2,)

69 	 [18.19671146  0.         62.85707821]
6 	val accuracy:  0.83870506 	 f_! score:  [0.26372046 0.         0.91097215]

69 	 [18.19671146  0.         62.85707821]
69 	 [29.4827381   0.         58.56753423]
69 	 [14.14210512  0.         68.12512634]
69 	 [ 501.  660. 5739.]
--- Test   Prime ---
0.83870506
f1:  [0.26372046 0.         0.91097215]
509411
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-08-30 17:50:30.244070: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:50:30.244111: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:50:30.244119: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:50:30.244124: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:50:30.244213: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:50:33.744317: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:50:33.744361: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:50:33.744369: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:50:33.744377: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:50:33.744476: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

501 	 [ 88.51443763 187.07860104 188.43842124]
val accuracy:  0.32499 	 f_! score:  [0.17667552 0.37341038 0.37612459]

501 	 [ 88.51443763 187.07860104 188.43842124]
501 	 [154.31840235 150.34868872 182.02175063]
501 	 [ 63.47361505 253.34572524 198.63435586]
501 	 [18833. 13850. 17417.]
---just Test  Review ---
0.32499
f1:  [0.17667552 0.37341038 0.37612459]
65730
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-08-30 17:50:58.804114: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:50:58.806547: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:50:58.806556: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:50:58.806561: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:50:58.806661: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:51:02.517209: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:51:02.517252: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:51:02.517260: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:51:02.517265: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:51:02.517357: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

101 	 [48.59635366  0.34375     0.15      ]
val accuracy:  0.3189109 	 f_! score:  [0.48115202 0.00340347 0.00148515]

101 	 [48.59635366  0.34375     0.15      ]
101 	 [32.16884058  0.42307692  0.6       ]
101 	 [1.00814815e+02 2.89473684e-01 8.57142857e-02]
101 	 [3212. 3413. 3475.]
---just Test  Twitter ---
0.3189109
f1:  [0.48115202 0.00340347 0.00148515]
(7733, 3)
7733
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-08-30 17:51:17.824268: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:51:17.824303: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:51:17.824309: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:51:17.824314: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:51:17.824394: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 17:51:20.815089: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 17:51:20.815133: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 17:51:20.815140: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 17:51:20.815144: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 17:51:20.815238: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3046 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

15 	 [1.50994876 2.35368449 0.42709908]
val accuracy:  0.08999999 	 f_! score:  [0.10066325 0.1569123  0.02847327]

15 	 [1.50994876 2.35368449 0.42709908]
15 	 [0.9438394  1.39840827 5.76470588]
15 	 [6.40549451 8.36707459 0.25351975]
15 	 [ 125.  129. 1246.]
---just Test  Medical ---
0.08999999
f1:  [0.10066325 0.1569123  0.02847327]

"""