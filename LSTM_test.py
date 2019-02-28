import matplotlib.pyplot as plt

from TestHelper import TestHelper


net_name = "LSTM"

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
Tensor("lstm_net/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("lstm_net_1/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   LSTM ---
2018-08-30 18:04:52.762249: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-08-30 18:04:52.853364: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-08-30 18:04:52.853715: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 970 major: 5 minor: 2 memoryClockRate(GHz): 1.253
pciBusID: 0000:01:00.0
totalMemory: 3.94GiB freeMemory: 3.33GiB
2018-08-30 18:04:52.853729: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 18:04:53.603537: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 18:04:53.603566: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 18:04:53.603571: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 18:04:53.603690: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/LSTM/pretrained_model.ckpt-0

Saving...
saved to models/LSTM/pretrained_model.ckpt-4482

500 	 [348.88332082 188.74142899 358.67926865]
0 	val accuracy:  0.63886 	 f_! score:  [0.69776664 0.37748286 0.71735854]


Saving...
saved to models/LSTM/pretrained_model.ckpt-8965

500 	 [362.09622278 228.30880261 366.9764323 ]
1 	val accuracy:  0.66334 	 f_! score:  [0.72419245 0.45661761 0.73395286]


Saving...
saved to models/LSTM/pretrained_model.ckpt-13448

500 	 [362.2252901  230.74595536 379.15774008]
2 	val accuracy:  0.67454004 	 f_! score:  [0.72445058 0.46149191 0.75831548]


Saving...
saved to models/LSTM/pretrained_model.ckpt-17931

500 	 [365.36592689 243.81129029 384.16422784]
3 	val accuracy:  0.68509996 	 f_! score:  [0.73073185 0.48762258 0.76832846]















Saving...
saved to models/LSTM/pretrained_model.ckpt-22414

500 	 [371.79835559 233.36416332 387.843716  ]
4 	val accuracy:  0.6908199 	 f_! score:  [0.74359671 0.46672833 0.77568743]


Saving...
saved to models/LSTM/pretrained_model.ckpt-26897

500 	 [367.6719609  259.1648761  381.98661836]
5 	val accuracy:  0.68719995 	 f_! score:  [0.73534392 0.51832975 0.76397324]


Saving...
saved to models/LSTM/pretrained_model.ckpt-31380

500 	 [375.36940575 226.60918748 390.08658456]
6 	val accuracy:  0.69513994 	 f_! score:  [0.75073881 0.45321837 0.78017317]

500 	 [375.36940575 226.60918748 390.08658456]
500 	 [356.81312863 282.07918949 370.21447783]
500 	 [398.89573892 193.52158573 414.76400292]
500 	 [18794. 13823. 17383.]
--- Test   Review ---
0.69513994
f1:  [0.75073881 0.45321837 0.78017317]
65730
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-30 18:57:26.475752: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 18:57:26.475801: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 18:57:26.475809: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 18:57:26.475815: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 18:57:26.475912: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 18:57:29.961125: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 18:57:29.961169: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 18:57:29.961177: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 18:57:29.961183: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 18:57:29.961278: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
/home/yannik/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)

101 	 [ 0.         29.75014961 44.56253272]
val accuracy:  0.32643566 	 f_! score:  [0.         0.29455594 0.4412132 ]

101 	 [ 0.         29.75014961 44.56253272]
101 	 [ 0.         35.34037626 32.14578205]
101 	 [ 0.         26.13894932 73.45110414]
101 	 [3374. 3417. 3309.]
---just Test  Twitter ---
0.32643566
f1:  [0.         0.29455594 0.4412132 ]
(7733, 3)
7733
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-30 18:57:48.066239: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 18:57:48.066273: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 18:57:48.066279: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 18:57:48.066283: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 18:57:48.066361: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 18:57:50.890723: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 18:57:50.890766: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 18:57:50.890773: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 18:57:50.890777: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 18:57:50.890864: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

15 	 [2.13155761 2.44790068 0.88642544]
val accuracy:  0.11733334 	 f_! score:  [0.14210384 0.16319338 0.05909503]

15 	 [2.13155761 2.44790068 0.88642544]
15 	 [ 1.28720752  1.54623115 13.44047619]
15 	 [8.45526557 7.21882284 0.46485226]
15 	 [ 121.  134. 1245.]
---just Test  Medical ---
0.11733334
f1:  [0.14210384 0.16319338 0.05909503]
(37126,)
(37126,)
37126
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-30 18:58:12.665900: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 18:58:12.665943: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 18:58:12.665952: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 18:58:12.665959: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 18:58:12.666057: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 18:58:15.470740: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 18:58:15.470783: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 18:58:15.470789: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 18:58:15.470794: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 18:58:15.470894: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
/home/yannik/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)

71 	 [ 7.93612504  2.94414964 49.63828025]
val accuracy:  0.5335211 	 f_! score:  [0.11177641 0.0414669  0.69913071]

71 	 [ 7.93612504  2.94414964 49.63828025]
71 	 [ 5.01904379  5.16309524 59.44650644]
71 	 [25.45144955  2.38023818 42.83932233]
71 	 [ 513.  673. 5914.]
---just Test  Prime ---
0.5335211
f1:  [0.11177641 0.0414669  0.69913071]


  Twitter
delete old models
65730
Tensor("lstm_net/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("lstm_net_1/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   LSTM ---
2018-08-30 18:58:39.712046: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 18:58:39.712081: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 18:58:39.712087: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 18:58:39.712091: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 18:58:39.712170: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/LSTM/pretrained_model.ckpt-0

Saving...
saved to models/LSTM/pretrained_model.ckpt-372

100 	 [63.55283214 47.57813201 53.28775304]
0 	val accuracy:  0.5597 	 f_! score:  [0.63552832 0.47578132 0.53287753]


Saving...
saved to models/LSTM/pretrained_model.ckpt-744

100 	 [65.01273284 59.97842881 48.74519419]
1 	val accuracy:  0.58959997 	 f_! score:  [0.65012733 0.59978429 0.48745194]


Saving...
saved to models/LSTM/pretrained_model.ckpt-1116

100 	 [63.96150716 60.53954561 51.03749958]
2 	val accuracy:  0.59220004 	 f_! score:  [0.63961507 0.60539546 0.510375  ]


Saving...
saved to models/LSTM/pretrained_model.ckpt-1488

100 	 [66.11511675 61.38789544 52.86794663]
3 	val accuracy:  0.6078 	 f_! score:  [0.66115117 0.61387895 0.52867947]


Saving...
saved to models/LSTM/pretrained_model.ckpt-1860

100 	 [66.58836682 59.71117049 54.2204413 ]
4 	val accuracy:  0.6065 	 f_! score:  [0.66588367 0.5971117  0.54220441]


Saving...
saved to models/LSTM/pretrained_model.ckpt-2232

100 	 [66.89114281 61.30548547 54.95006689]
5 	val accuracy:  0.615 	 f_! score:  [0.66891143 0.61305485 0.54950067]


Saving...
saved to models/LSTM/pretrained_model.ckpt-2604

100 	 [67.31844069 61.90741087 55.51120872]
6 	val accuracy:  0.6213 	 f_! score:  [0.67318441 0.61907411 0.55511209]

100 	 [67.31844069 61.90741087 55.51120872]
100 	 [65.97505535 59.58520063 60.54180158]
100 	 [69.32914433 65.08610971 51.87778355]
100 	 [3286. 3336. 3378.]
--- Test   Twitter ---
0.6213
f1:  [0.67318441 0.61907411 0.55511209]
/home/yannik/PycharmProjects/bachelorarbeit/TestHelper.py:59: DtypeWarning: Columns (1) have mixed types. Specify dtype option on import or set low_memory=False.
  input_data = self.decide_which_input(input_name)
509411
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-30 19:05:30.184807: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:05:30.184857: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:05:30.184884: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:05:30.184892: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:05:30.184996: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 19:05:33.768272: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:05:33.768315: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:05:33.768323: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:05:33.768329: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:05:33.768422: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

501 	 [  7.67336291 183.7359401  183.75899046]
val accuracy:  0.30145708 	 f_! score:  [0.01531609 0.3667384  0.36678441]

501 	 [  7.67336291 183.7359401  183.75899046]
501 	 [106.11666667 136.87468194 170.29678744]
501 	 [  4.02655836 285.43911014 202.82235645]
501 	 [18825. 13862. 17413.]
---just Test  Review ---
0.30145708
f1:  [0.01531609 0.3667384  0.36678441]
(7733, 3)
7733
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-30 19:06:07.782031: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:06:07.782066: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:06:07.782072: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:06:07.782076: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:06:07.782158: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 19:06:10.664339: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:06:10.664382: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:06:10.664389: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:06:10.664393: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:06:10.664476: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

15 	 [ 1.18434343  1.80962727 10.06366246]
val accuracy:  0.5086667 	 f_! score:  [0.07895623 0.12064182 0.67091083]

15 	 [ 1.18434343  1.80962727 10.06366246]
15 	 [ 1.21666667  1.14144281 12.28391041]
15 	 [1.20952381 5.07804973 8.59957058]
15 	 [ 122.  129. 1249.]
---just Test  Medical ---
0.5086667
f1:  [0.07895623 0.12064182 0.67091083]
(37126,)
(37126,)
37126
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-30 19:06:32.430297: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:06:32.430344: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:06:32.430350: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:06:32.430355: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:06:32.430451: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 19:06:35.270579: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:06:35.270623: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:06:35.270630: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:06:35.270637: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:06:35.270723: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

71 	 [ 8.87520906  5.19288683 33.43018777]
val accuracy:  0.3256338 	 f_! score:  [0.12500294 0.07313925 0.47084772]

71 	 [ 8.87520906  5.19288683 33.43018777]
71 	 [ 5.1750646   8.47222222 59.60972694]
71 	 [43.89095321  4.31802488 23.43065396]
71 	 [ 515.  679. 5906.]
---just Test  Prime ---
0.3256338
f1:  [0.12500294 0.07313925 0.47084772]


  Medical
delete old models
(7733, 3)
7733
Tensor("lstm_net/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("lstm_net_1/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   LSTM ---
2018-08-30 19:06:55.100106: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:06:55.100138: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:06:55.100144: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:06:55.100148: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:06:55.100224: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/LSTM/pretrained_model.ckpt-0

Saving...
saved to models/LSTM/pretrained_model.ckpt-42

14 	 [ 0.          0.         12.77481666]
0 	val accuracy:  0.84214294 	 f_! score:  [0.        0.        0.9124869]


Saving...
saved to models/LSTM/pretrained_model.ckpt-84

14 	 [ 0.          0.         12.77481666]
1 	val accuracy:  0.84214276 	 f_! score:  [0.        0.        0.9124869]


Saving...
saved to models/LSTM/pretrained_model.ckpt-126

14 	 [ 0.          0.         12.77481666]
2 	val accuracy:  0.84214294 	 f_! score:  [0.        0.        0.9124869]


Saving...
saved to models/LSTM/pretrained_model.ckpt-168

14 	 [ 1.8453306   0.18181818 12.56279162]
3 	val accuracy:  0.8142857 	 f_! score:  [0.13180933 0.01298701 0.89734226]


Saving...
saved to models/LSTM/pretrained_model.ckpt-210

14 	 [ 1.04640523  0.         12.71859378]
4 	val accuracy:  0.83357143 	 f_! score:  [0.07474323 0.         0.90847098]


Saving...
saved to models/LSTM/pretrained_model.ckpt-252

14 	 [ 1.13584357  1.11575092 12.53595445]
5 	val accuracy:  0.81214285 	 f_! score:  [0.08113168 0.07969649 0.89542532]


Saving...
saved to models/LSTM/pretrained_model.ckpt-294

14 	 [ 1.95542232  0.         12.62702609]
6 	val accuracy:  0.8157143 	 f_! score:  [0.13967302 0.         0.90193043]

14 	 [ 1.95542232  0.         12.62702609]
14 	 [ 2.1497114   0.         12.01736261]
14 	 [ 1.91575092  0.         13.33329886]
14 	 [ 105.  116. 1179.]
--- Test   Medical ---
0.8157143
f1:  [0.13967302 0.         0.90193043]
509411
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-30 19:09:03.269706: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:09:03.269750: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:09:03.269760: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:09:03.269767: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:09:03.269887: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 19:09:06.721817: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:09:06.721860: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:09:06.721866: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:09:06.721871: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:09:06.721954: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

501 	 [250.90569514  32.44079788 139.52979997]
val accuracy:  0.37203595 	 f_! score:  [0.50080977 0.06475209 0.27850259]

501 	 [250.90569514  32.44079788 139.52979997]
501 	 [189.9195811  154.82712843 178.73480661]
501 	 [374.12469597  18.70824794 116.68797467]
501 	 [18837. 13848. 17415.]
---just Test  Review ---
0.37203595
f1:  [0.50080977 0.06475209 0.27850259]
65730
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-30 19:09:45.406063: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:09:45.406100: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:09:45.406107: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:09:45.406114: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:09:45.406204: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 19:09:48.429736: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:09:48.429777: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:09:48.429783: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:09:48.429787: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:09:48.429877: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

101 	 [ 0.17200277  0.11111111 50.60217669]
val accuracy:  0.3362376 	 f_! score:  [0.001703   0.00110011 0.50101165]

101 	 [ 0.17200277  0.11111111 50.60217669]
101 	 [ 3.          1.         33.95128633]
101 	 [8.85763259e-02 5.88235294e-02 1.00965517e+02]
101 	 [3394. 3314. 3392.]
---just Test  Twitter ---
0.3362376
f1:  [0.001703   0.00110011 0.50101165]
(37126,)
(37126,)
37126
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-30 19:10:13.598603: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:10:13.598645: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:10:13.598653: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:10:13.598660: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:10:13.598759: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 19:10:16.424138: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:10:16.424178: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:10:16.424184: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:10:16.424188: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:10:16.424273: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

71 	 [ 8.78178276  9.64904863 23.82013761]
val accuracy:  0.24197184 	 f_! score:  [0.12368708 0.13590209 0.3354949 ]

71 	 [ 8.78178276  9.64904863 23.82013761]
71 	 [ 5.31000728  6.59537188 58.31178624]
71 	 [34.57239329 22.85340963 15.11156498]
71 	 [ 504.  671. 5925.]
---just Test  Prime ---
0.24197184
f1:  [0.12368708 0.13590209 0.3354949 ]


  Prime
delete old models
(37126,)
(37126,)
37126
Tensor("lstm_net/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("lstm_net_1/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   LSTM ---
2018-08-30 19:10:43.158106: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:10:43.158140: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:10:43.158147: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:10:43.158154: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:10:43.158234: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/LSTM/pretrained_model.ckpt-0

Saving...
saved to models/LSTM/pretrained_model.ckpt-201
(2,)

69 	 [ 2.26575092  4.82608853 61.48399279]
0 	val accuracy:  0.8067626 	 f_! score:  [0.03283697 0.06994331 0.89107236]


Saving...
saved to models/LSTM/pretrained_model.ckpt-402
(2,)

69 	 [10.45955039  0.89616854 61.98316844]
1 	val accuracy:  0.8151798 	 f_! score:  [0.15158769 0.01298795 0.89830679]


Saving...
saved to models/LSTM/pretrained_model.ckpt-603
(2,)

69 	 [10.50856345  3.85954384 62.0515291 ]
2 	val accuracy:  0.81676257 	 f_! score:  [0.15229802 0.05593542 0.89929752]


Saving...
saved to models/LSTM/pretrained_model.ckpt-804
(2,)

69 	 [ 8.92571331  1.4588221  62.43591896]
3 	val accuracy:  0.8282734 	 f_! score:  [0.12935816 0.02114235 0.90486839]


Saving...
saved to models/LSTM/pretrained_model.ckpt-1005
(2,)

69 	 [13.1814333   2.8890931  62.35056057]
4 	val accuracy:  0.825036 	 f_! score:  [0.19103527 0.04187091 0.90363131]


Saving...
saved to models/LSTM/pretrained_model.ckpt-1206
(2,)
(2,)

68 	 [10.94265564  3.74214539 61.45224866]
5 	val accuracy:  0.82804346 	 f_! score:  [0.16092141 0.05503155 0.90370954]


Saving...
saved to models/LSTM/pretrained_model.ckpt-1407
(2,)

69 	 [15.70555242  4.16163594 62.34534161]
6 	val accuracy:  0.8259712 	 f_! score:  [0.2276167  0.06031356 0.90355568]

69 	 [15.70555242  4.16163594 62.34534161]
69 	 [27.81078644  9.16904762 58.66029506]
69 	 [11.74100999  2.86886345 66.76623987]
69 	 [ 501.  660. 5739.]
--- Test   Prime ---
0.8259712
f1:  [0.2276167  0.06031356 0.90355568]
509411
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-30 19:15:10.905260: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:15:10.905298: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:15:10.905304: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:15:10.905308: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:15:10.905402: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 19:15:14.347958: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:15:14.348004: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:15:14.348011: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:15:14.348018: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:15:14.348106: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

501 	 [195.18294728 146.56157588 117.06623697]
val accuracy:  0.31792414 	 f_! score:  [0.38958672 0.29253808 0.23366514]

501 	 [195.18294728 146.56157588 117.06623697]
501 	 [183.86781792 133.46776032 151.14068743]
501 	 [211.18717164 166.59897598  97.31094643]
501 	 [18833. 13854. 17413.]
---just Test  Review ---
0.31792414
f1:  [0.38958672 0.29253808 0.23366514]
65730
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-30 19:15:52.725933: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:15:52.725968: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:15:52.725975: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:15:52.725979: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:15:52.726060: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 19:15:55.838697: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:15:55.838739: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:15:55.838746: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:15:55.838751: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:15:55.838842: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

101 	 [40.5687703  42.6408676   3.66066204]
val accuracy:  0.34891087 	 f_! score:  [0.40167099 0.42218681 0.03624418]

101 	 [40.5687703  42.6408676   3.66066204]
101 	 [36.57499383 34.50212462 24.26666667]
101 	 [46.26544856 56.77264249  2.00982269]
101 	 [3439. 3346. 3315.]
---just Test  Twitter ---
0.34891087
f1:  [0.40167099 0.42218681 0.03624418]
(7733, 3)
7733
Vorsicht! es muss vorher ein  LSTM  Netz abgespeichert worden sein
2018-08-30 19:16:14.061663: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:16:14.061696: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:16:14.061705: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:16:14.061713: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:16:14.061798: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-30 19:16:17.023437: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-30 19:16:17.023482: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-30 19:16:17.023489: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-30 19:16:17.023496: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-30 19:16:17.023589: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3044 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

15 	 [2.04364707 0.83653846 2.00058036]
val accuracy:  0.13333334 	 f_! score:  [0.13624314 0.05576923 0.13337202]

15 	 [2.04364707 0.83653846 2.00058036]
15 	 [ 1.15463811  1.5        11.8527417 ]
15 	 [12.6782967   0.59444444  1.10315374]
15 	 [ 121.  124. 1255.]
---just Test  Medical ---
0.13333334
f1:  [0.13624314 0.05576923 0.13337202]


"""