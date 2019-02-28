import matplotlib.pyplot as plt

from TestHelper import TestHelper


net_name = "ONLY_EMBED"

testhelper = TestHelper()
review_loss, review_acc = testhelper.train_input("Review", net_name, epochs=4)

testhelper.just_test("Twitter", net_name)
testhelper.just_test("Medical", net_name)
testhelper.just_test("Prime", net_name)

tw_loss, tw_acc = testhelper.train_input("Twitter", net_name, epochs=4)

testhelper.just_test("Review", net_name)
testhelper.just_test("Medical", net_name)
testhelper.just_test("Prime", net_name)

med_loss, med_acc = testhelper.train_input("Medical", net_name, epochs=4)

testhelper.just_test("Review", net_name)
testhelper.just_test("Twitter", net_name)
testhelper.just_test("Prime", net_name)

prime_loss, prime_acc = testhelper.train_input("Prime", net_name, epochs=4)

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
---init ready   ONLY_EMBED ---
2018-08-29 16:34:52.151132: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-08-29 16:34:52.245279: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-08-29 16:34:52.245625: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 970 major: 5 minor: 2 memoryClockRate(GHz): 1.253
pciBusID: 0000:01:00.0
totalMemory: 3.94GiB freeMemory: 3.12GiB
2018-08-29 16:34:52.245639: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:34:52.959218: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:34:52.959245: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:34:52.959250: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:34:52.959367: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-0

Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-4482

500 	 [291.68567867 145.81735013 231.79757617]
0 	val accuracy:  0.48965997 	 f_! score:  [0.58337136 0.2916347  0.46359515]


Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-8965

500 	 [288.65353018 120.42691677 290.24651401]
1 	val accuracy:  0.51538 	 f_! score:  [0.57730706 0.24085383 0.58049303]


Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-13448

500 	 [259.25077209 188.32647304 284.01091483]
2 	val accuracy:  0.50012 	 f_! score:  [0.51850154 0.37665295 0.56802183]


Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-17931

500 	 [281.62201448 192.59964678 262.56147212]
3 	val accuracy:  0.50317997 	 f_! score:  [0.56324403 0.38519929 0.52512294]

500 	 [281.62201448 192.59964678 262.56147212]
500 	 [266.27535021 184.89728915 305.28794303]
500 	 [302.33993803 205.61186286 233.49757669]
500 	 [18794. 13823. 17383.]
--- Test   Review ---
0.50317997
f1:  [0.56324403 0.38519929 0.52512294]
65730
Vorsicht! es muss vorher ein  ONLY_EMBED  Netz abgespeichert worden sein
2018-08-29 16:36:07.876412: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:36:07.876456: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:36:07.876463: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:36:07.876467: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:36:07.876574: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-29 16:36:11.329567: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:36:11.329608: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:36:11.329614: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:36:11.329619: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:36:11.329708: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
/home/yannik/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)

101 	 [ 3.48476154 48.12568527 10.24458843]
val accuracy:  0.3262376 	 f_! score:  [0.03450259 0.47649193 0.10143157]

101 	 [ 3.48476154 48.12568527 10.24458843]
101 	 [24.89166667 32.94323726 35.30396825]
101 	 [ 1.9235922  90.42156604  6.13194084]
101 	 [3310. 3377. 3413.]
---just Test  Twitter ---
0.3262376
f1:  [0.03450259 0.47649193 0.10143157]
(7733, 3)
7733
Vorsicht! es muss vorher ein  ONLY_EMBED  Netz abgespeichert worden sein
2018-08-29 16:36:25.036641: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:36:25.036675: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:36:25.036681: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:36:25.036685: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:36:25.036761: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-29 16:36:27.911497: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:36:27.911540: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:36:27.911547: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:36:27.911552: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:36:27.911644: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

15 	 [2.10917023 1.9399265  2.78250354]
val accuracy:  0.16266666 	 f_! score:  [0.14061135 0.12932843 0.18550024]

15 	 [2.10917023 1.9399265  2.78250354]
15 	 [ 1.2133485   1.60467398 12.67458097]
15 	 [11.76373626  2.8705711   1.57029595]
15 	 [ 120.  122. 1258.]
---just Test  Medical ---
0.16266666
f1:  [0.14061135 0.12932843 0.18550024]
(37126,)
(37126,)
37126
Vorsicht! es muss vorher ein  ONLY_EMBED  Netz abgespeichert worden sein
2018-08-29 16:36:48.354839: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:36:48.354879: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:36:48.354887: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:36:48.354893: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:36:48.354990: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-29 16:36:51.099551: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:36:51.099591: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:36:51.099597: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:36:51.099602: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:36:51.099689: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
/home/yannik/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)

71 	 [ 5.74684488  0.81267507 60.8187293 ]
val accuracy:  0.7494366 	 f_! score:  [0.08094148 0.01144613 0.85660182]

71 	 [ 5.74684488  0.81267507 60.8187293 ]
71 	 [ 5.26367244  3.5        59.15219808]
71 	 [ 8.11869194  0.46658009 63.01342322]
71 	 [ 505.  669. 5926.]
---just Test  Prime ---
0.7494366
f1:  [0.08094148 0.01144613 0.85660182]


  Twitter
delete old models
65730
---init ready   ONLY_EMBED ---
2018-08-29 16:37:11.891213: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:37:11.891247: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:37:11.891253: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:37:11.891258: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:37:11.891338: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-0

Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-372

100 	 [61.30755572 30.16002928 53.98804773]
0 	val accuracy:  0.5183 	 f_! score:  [0.61307556 0.30160029 0.53988048]


Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-744

100 	 [59.65908977 44.26395918 54.53576391]
1 	val accuracy:  0.5397 	 f_! score:  [0.5965909  0.44263959 0.54535764]


Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-1116

100 	 [60.55491591 55.62946182 17.25955357]
2 	val accuracy:  0.5107 	 f_! score:  [0.60554916 0.55629462 0.17259554]


Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-1488

100 	 [62.83672642 57.49285448 39.870166  ]
3 	val accuracy:  0.55619997 	 f_! score:  [0.62836726 0.57492854 0.39870166]

100 	 [62.83672642 57.49285448 39.870166  ]
100 	 [53.61122391 55.87009801 60.65493925]
100 	 [76.75188319 60.07274896 30.22506143]
100 	 [3312. 3337. 3351.]
--- Test   Twitter ---
0.55619997
f1:  [0.62836726 0.57492854 0.39870166]
/home/yannik/PycharmProjects/bachelorarbeit/TestHelper.py:59: DtypeWarning: Columns (1) have mixed types. Specify dtype option on import or set low_memory=False.
  input_data = self.decide_which_input(input_name)
509411
Vorsicht! es muss vorher ein  ONLY_EMBED  Netz abgespeichert worden sein
2018-08-29 16:38:33.822006: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:38:33.822049: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:38:33.822060: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:38:33.822068: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:38:33.822178: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-29 16:38:37.212224: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:38:37.212275: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:38:37.212282: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:38:37.212286: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:38:37.212369: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

501 	 [218.12368466 139.9553984  179.17651357]
val accuracy:  0.37167662 	 f_! score:  [0.43537662 0.27935209 0.35763775]

501 	 [218.12368466 139.9553984  179.17651357]
501 	 [199.77058806 142.01617867 208.52614202]
501 	 [243.86752629 141.51634255 159.88142173]
501 	 [18836. 13848. 17416.]
---just Test  Review ---
0.37167662
f1:  [0.43537662 0.27935209 0.35763775]
(7733, 3)
7733
Vorsicht! es muss vorher ein  ONLY_EMBED  Netz abgespeichert worden sein
2018-08-29 16:38:52.066979: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:38:52.067012: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:38:52.067019: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:38:52.067025: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:38:52.067109: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-29 16:38:54.949995: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:38:54.950037: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:38:54.950044: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:38:54.950051: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:38:54.950139: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

15 	 [2.09879327 0.76066434 6.64258096]
val accuracy:  0.3133334 	 f_! score:  [0.13991955 0.05071096 0.44283873]

15 	 [2.09879327 0.76066434 6.64258096]
15 	 [ 1.22335618  2.66666667 12.24400304]
15 	 [10.26502795  0.45519105  4.63175344]
15 	 [ 124.  133. 1243.]
---just Test  Medical ---
0.3133334
f1:  [0.13991955 0.05071096 0.44283873]
(37126,)
(37126,)
37126
Vorsicht! es muss vorher ein  ONLY_EMBED  Netz abgespeichert worden sein
2018-08-29 16:39:15.412683: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:39:15.412720: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:39:15.412727: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:39:15.412732: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:39:15.412816: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-29 16:39:18.180362: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:39:18.180404: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:39:18.180411: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:39:18.180418: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:39:18.180504: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

71 	 [ 8.71114208 11.31451927  1.03670655]
val accuracy:  0.093239434 	 f_! score:  [0.12269214 0.15935943 0.0146015 ]

71 	 [ 8.71114208 11.31451927  1.03670655]
71 	 [ 5.34300516  6.91359886 27.58333333]
71 	 [32.50943433 38.6630812   0.53089916]
71 	 [ 510.  676. 5914.]
---just Test  Prime ---
0.093239434
f1:  [0.12269214 0.15935943 0.0146015 ]


  Medical
delete old models
(7733, 3)
7733
---init ready   ONLY_EMBED ---
2018-08-29 16:39:34.292158: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:39:34.292193: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:39:34.292199: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:39:34.292203: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:39:34.292282: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-0

Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-42

14 	 [ 0.          0.44551282 12.62459752]
0 	val accuracy:  0.8221429 	 f_! score:  [0.         0.03182234 0.90175697]


Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-84

14 	 [ 1.95075707  0.81495726 10.78310919]
1 	val accuracy:  0.6214285 	 f_! score:  [0.13933979 0.05821123 0.77022209]


Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-126

14 	 [ 0.11764706  0.66666667 12.66888156]
2 	val accuracy:  0.8292856 	 f_! score:  [0.00840336 0.04761905 0.90492011]


Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-168

14 	 [ 0.14285714  0.66666667 12.71769035]
3 	val accuracy:  0.83357143 	 f_! score:  [0.01020408 0.04761905 0.90840645]

14 	 [ 0.14285714  0.66666667 12.71769035]
14 	 [ 0.25        2.         11.82157178]
14 	 [ 0.1         0.42424242 13.81583559]
14 	 [ 105.  116. 1179.]
--- Test   Medical ---
0.83357143
f1:  [0.01020408 0.04761905 0.90840645]
509411
Vorsicht! es muss vorher ein  ONLY_EMBED  Netz abgespeichert worden sein
2018-08-29 16:40:53.950056: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:40:53.950092: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:40:53.950098: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:40:53.950103: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:40:53.950205: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-29 16:40:57.340495: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:40:57.340535: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:40:57.340542: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:40:57.340547: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:40:57.340638: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

501 	 [ 30.58022436 191.30104189 223.77258081]
val accuracy:  0.34473053 	 f_! score:  [0.06103837 0.38183841 0.44665186]

501 	 [ 30.58022436 191.30104189 223.77258081]
501 	 [186.4515873  148.63173097 197.83929004]
501 	 [ 17.04699532 274.27870102 261.54499525]
501 	 [18840. 13850. 17410.]
---just Test  Review ---
0.34473053
f1:  [0.06103837 0.38183841 0.44665186]
65730
Vorsicht! es muss vorher ein  ONLY_EMBED  Netz abgespeichert worden sein
2018-08-29 16:41:16.703802: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:41:16.703837: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:41:16.703845: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:41:16.703855: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:41:16.703949: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-29 16:41:19.776543: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:41:19.776583: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:41:19.776589: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:41:19.776593: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:41:19.776682: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

101 	 [5.02820132e+01 2.06755770e+00 4.87804878e-02]
val accuracy:  0.3346535 	 f_! score:  [4.97841715e-01 2.04708683e-02 4.82975127e-04]

101 	 [5.02820132e+01 2.06755770e+00 4.87804878e-02]
101 	 [33.69541996 24.83333333  1.        ]
101 	 [1.00391561e+02 1.08700084e+00 2.50000000e-02]
101 	 [3364. 3381. 3355.]
---just Test  Twitter ---
0.3346535
f1:  [4.97841715e-01 2.04708683e-02 4.82975127e-04]
(37126,)
(37126,)
37126
Vorsicht! es muss vorher ein  ONLY_EMBED  Netz abgespeichert worden sein
2018-08-29 16:41:40.325652: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:41:40.325686: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:41:40.325693: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:41:40.325697: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:41:40.325786: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-29 16:41:43.087073: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:41:43.087113: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:41:43.087119: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:41:43.087124: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:41:43.087212: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

71 	 [ 6.4707182   9.80706368 58.56247837]
val accuracy:  0.7035212 	 f_! score:  [0.09113688 0.13812766 0.82482364]

71 	 [ 6.4707182   9.80706368 58.56247837]
71 	 [ 7.0619912   9.38178909 59.92164985]
71 	 [ 7.86230801 11.99033969 57.61564424]
71 	 [ 504.  671. 5925.]
---just Test  Prime ---
0.7035212
f1:  [0.09113688 0.13812766 0.82482364]


  Prime
delete old models
(37126,)
(37126,)
37126
---init ready   ONLY_EMBED ---
2018-08-29 16:42:06.141126: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:42:06.141161: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:42:06.141167: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:42:06.141172: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:42:06.141252: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-0

Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-201
(2,)
(2,)

68 	 [ 3.40055862  3.23779872 60.80785052]
0 	val accuracy:  0.8126087 	 f_! score:  [0.05000822 0.04761469 0.8942331 ]


Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-402

70 	 [ 3.6347279   3.86455748 62.61972089]
1 	val accuracy:  0.8101429 	 f_! score:  [0.05192468 0.05520796 0.89456744]


Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-603
(2,)

69 	 [15.34824154  8.07739184 60.14644912]
2 	val accuracy:  0.7682014 	 f_! score:  [0.22243828 0.11706365 0.87168767]


Saving...
saved to models/ONLY_EMBED/pretrained_model.ckpt-804
(2,)

69 	 [ 9.55535072  3.54305402 61.62818175]
3 	val accuracy:  0.80841726 	 f_! score:  [0.13848334 0.05134861 0.89316205]

69 	 [ 9.55535072  3.54305402 61.62818175]
69 	 [15.7215368  10.61666667 58.18289589]
69 	 [ 7.84527155  2.45139069 65.81085048]
69 	 [ 501.  661. 5738.]
--- Test   Prime ---
0.80841726
f1:  [0.13848334 0.05134861 0.89316205]
509411
Vorsicht! es muss vorher ein  ONLY_EMBED  Netz abgespeichert worden sein
2018-08-29 16:43:27.039923: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:43:27.039962: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:43:27.039971: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:43:27.039980: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:43:27.040092: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-29 16:43:30.042391: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:43:30.042433: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:43:30.042439: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:43:30.042443: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:43:30.042537: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

501 	 [190.5114475  181.16283561  56.86179319]
val accuracy:  0.31762472 	 f_! score:  [0.38026237 0.36160247 0.11349659]

501 	 [190.5114475  181.16283561  56.86179319]
501 	 [197.01549458 136.49231618 145.18245749]
501 	 [187.36631581 275.53820311  36.13640294]
501 	 [18833. 13849. 17418.]
---just Test  Review ---
0.31762472
f1:  [0.38026237 0.36160247 0.11349659]
65730
Vorsicht! es muss vorher ein  ONLY_EMBED  Netz abgespeichert worden sein
2018-08-29 16:43:49.031484: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:43:49.031527: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:43:49.031535: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:43:49.031539: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:43:49.031618: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-29 16:43:52.201568: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:43:52.201610: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:43:52.201617: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:43:52.201622: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:43:52.201707: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

101 	 [ 0.20342523 50.7260242   0.        ]
val accuracy:  0.33732674 	 f_! score:  [0.00201411 0.50223786 0.        ]

101 	 [ 0.20342523 50.7260242   0.        ]
101 	 [ 3.        34.0569697  0.       ]
101 	 [  0.10559846 100.93325918   0.        ]
101 	 [3383. 3406. 3311.]
---just Test  Twitter ---
0.33732674
f1:  [0.00201411 0.50223786 0.        ]
(7733, 3)
7733
Vorsicht! es muss vorher ein  ONLY_EMBED  Netz abgespeichert worden sein
2018-08-29 16:44:05.869955: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:44:05.869986: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:44:05.869992: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:44:05.869996: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:44:05.870074: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
2018-08-29 16:44:08.614470: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-08-29 16:44:08.614512: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-08-29 16:44:08.614518: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-08-29 16:44:08.614523: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-08-29 16:44:08.614606: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2826 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

15 	 [ 0.30634921  2.05991941 10.40745673]
val accuracy:  0.534 	 f_! score:  [0.02042328 0.13732796 0.69383045]

15 	 [ 0.30634921  2.05991941 10.40745673]
15 	 [ 0.61666667  1.28123874 12.57660102]
15 	 [0.2047619  6.18335276 8.90777722]
15 	 [ 119.  126. 1255.]
---just Test  Medical ---
0.534
f1:  [0.02042328 0.13732796 0.69383045]

"""
