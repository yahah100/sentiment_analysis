import matplotlib.pyplot as plt

from TestHelper import TestHelper


net_name = "CNN"

testhelper = TestHelper()
review_loss, review_acc = testhelper.train_input("Review", net_name, epochs=4)

testhelper.just_test("Twitter", net_name, print_time=True)
testhelper.train_last_layer("Twitter", net_name, epochs=1, print_time=True)
testhelper.train_input("Twitter", net_name, epochs=1, print_time=True, restore=True)


"""
Review
delete old models
/home/yannik/ba/TestHelper.py:64: DtypeWarning: Columns (1) have mixed types. Specify dtype option on import or set low_memory=False.
  input_data = self.decide_which_input(input_name)
509411
Tensor("ConvNet/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("ConvNet_1/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   CNN ---
2018-10-14 18:36:12.353702: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-10-14 18:36:12.446957: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-10-14 18:36:12.447294: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 970 major: 5 minor: 2 memoryClockRate(GHz): 1.253
pciBusID: 0000:01:00.0
totalMemory: 3.94GiB freeMemory: 3.19GiB
2018-10-14 18:36:12.447308: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-10-14 18:36:13.156585: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-14 18:36:13.156616: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-10-14 18:36:13.156622: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-10-14 18:36:13.156939: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2905 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
2018-10-14 18:36:20.533319: W tensorflow/core/framework/allocator.cc:108] Allocation of 367528800 exceeds 10% of system memory.

Saving...
saved to models/CNN/pretrained_model.ckpt-0
2018-10-14 18:36:25.930470: W tensorflow/core/framework/allocator.cc:108] Allocation of 238702800 exceeds 10% of system memory.
2018-10-14 18:36:26.202722: W tensorflow/core/framework/allocator.cc:108] Allocation of 238702800 exceeds 10% of system memory.
2018-10-14 18:36:26.471267: W tensorflow/core/framework/allocator.cc:108] Allocation of 238702800 exceeds 10% of system memory.
2018-10-14 18:36:26.740125: W tensorflow/core/framework/allocator.cc:108] Allocation of 238702800 exceeds 10% of system memory.
2018-10-14 18:36:34.943547: W tensorflow/core/common_runtime/bfc_allocator.cc:219] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.60GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
0 : loss:  2.4797688 	 acc:  0.4
40 : loss:  1.093603 	 acc:  0.38
80 : loss:  1.1027913 	 acc:  0.33
120 : loss:  1.0802163 	 acc:  0.4
160 : loss:  1.0884652 	 acc:  0.36
200 : loss:  1.062772 	 acc:  0.44
240 : loss:  1.0715008 	 acc:  0.47
280 : loss:  1.0282133 	 acc:  0.48
320 : loss:  0.8561518 	 acc:  0.63
360 : loss:  0.9267742 	 acc:  0.59
400 : loss:  1.003554 	 acc:  0.49
440 : loss:  1.0180616 	 acc:  0.51
480 : loss:  0.8899496 	 acc:  0.64
520 : loss:  0.92326134 	 acc:  0.55
560 : loss:  0.92482114 	 acc:  0.58
600 : loss:  0.926982 	 acc:  0.57
640 : loss:  0.92252994 	 acc:  0.53
680 : loss:  0.8991607 	 acc:  0.56
720 : loss:  0.96533686 	 acc:  0.48
760 : loss:  0.92055655 	 acc:  0.49
800 : loss:  0.87321943 	 acc:  0.63
840 : loss:  0.8129223 	 acc:  0.64
880 : loss:  0.84957397 	 acc:  0.63
920 : loss:  0.9859088 	 acc:  0.51
960 : loss:  0.95257264 	 acc:  0.46
1000 : loss:  0.88103515 	 acc:  0.61
1040 : loss:  0.84326327 	 acc:  0.64
1080 : loss:  0.8478107 	 acc:  0.59
1120 : loss:  0.78921366 	 acc:  0.69
1160 : loss:  1.0367681 	 acc:  0.43
1200 : loss:  0.7860175 	 acc:  0.61
1240 : loss:  0.8163732 	 acc:  0.56
1280 : loss:  0.872316 	 acc:  0.57
1320 : loss:  0.9869692 	 acc:  0.52
1360 : loss:  0.7845598 	 acc:  0.59
1400 : loss:  0.7231506 	 acc:  0.66
1440 : loss:  0.8228634 	 acc:  0.62
1480 : loss:  0.82857126 	 acc:  0.59
1520 : loss:  0.88318557 	 acc:  0.57
1560 : loss:  0.9502977 	 acc:  0.57
1600 : loss:  0.8660157 	 acc:  0.55
1640 : loss:  0.8493011 	 acc:  0.59
1680 : loss:  0.86314154 	 acc:  0.61
1720 : loss:  0.7820186 	 acc:  0.61
1760 : loss:  0.8126828 	 acc:  0.62
1800 : loss:  0.82946414 	 acc:  0.62
1840 : loss:  0.8358545 	 acc:  0.6
1880 : loss:  0.8361415 	 acc:  0.59
1920 : loss:  0.81181264 	 acc:  0.66
1960 : loss:  0.85598165 	 acc:  0.6
2000 : loss:  0.7216547 	 acc:  0.69
2040 : loss:  0.78318864 	 acc:  0.71
2080 : loss:  0.7892379 	 acc:  0.59
2120 : loss:  0.8600609 	 acc:  0.61
2160 : loss:  0.8572759 	 acc:  0.63
2200 : loss:  0.7784517 	 acc:  0.64
2240 : loss:  0.76300603 	 acc:  0.68
2280 : loss:  0.7850357 	 acc:  0.67
2320 : loss:  0.7584855 	 acc:  0.66
2360 : loss:  0.8240091 	 acc:  0.6
2400 : loss:  0.7043909 	 acc:  0.67
2440 : loss:  0.62898016 	 acc:  0.74
2480 : loss:  0.8928418 	 acc:  0.59
2520 : loss:  0.7556621 	 acc:  0.63
2560 : loss:  0.7610743 	 acc:  0.68
2600 : loss:  0.7897142 	 acc:  0.67
2640 : loss:  0.828328 	 acc:  0.61
2680 : loss:  0.73768693 	 acc:  0.64
2720 : loss:  0.7669307 	 acc:  0.68
2760 : loss:  0.81583166 	 acc:  0.62
2800 : loss:  0.6737679 	 acc:  0.66
2840 : loss:  0.8271807 	 acc:  0.62
2880 : loss:  0.8004205 	 acc:  0.69
2920 : loss:  0.76022667 	 acc:  0.68
2960 : loss:  0.8441777 	 acc:  0.61
3000 : loss:  0.7843673 	 acc:  0.67
3040 : loss:  0.7499182 	 acc:  0.7
3080 : loss:  0.72903746 	 acc:  0.74
3120 : loss:  0.7650535 	 acc:  0.67
3160 : loss:  0.75309354 	 acc:  0.65
3200 : loss:  0.8833067 	 acc:  0.58
3240 : loss:  0.85484374 	 acc:  0.57
3280 : loss:  0.7769854 	 acc:  0.63
3320 : loss:  0.7989823 	 acc:  0.6
3360 : loss:  0.80901426 	 acc:  0.6
3400 : loss:  0.8040652 	 acc:  0.6
3440 : loss:  0.6430542 	 acc:  0.68
3480 : loss:  0.8587034 	 acc:  0.58
3520 : loss:  0.74994 	 acc:  0.67
3560 : loss:  0.6916355 	 acc:  0.73
3600 : loss:  0.7649128 	 acc:  0.66
3640 : loss:  0.8436241 	 acc:  0.58
3680 : loss:  0.7034776 	 acc:  0.7
3720 : loss:  0.87960345 	 acc:  0.57
3760 : loss:  0.8248123 	 acc:  0.62
3800 : loss:  0.68729806 	 acc:  0.69
3840 : loss:  0.75160646 	 acc:  0.66
3880 : loss:  0.7925146 	 acc:  0.6
3920 : loss:  0.7226889 	 acc:  0.69
3960 : loss:  0.7277622 	 acc:  0.63
4000 : loss:  0.7854263 	 acc:  0.69
4040 : loss:  0.7970897 	 acc:  0.62
4080 : loss:  0.68166643 	 acc:  0.74
4120 : loss:  0.79792273 	 acc:  0.64
4160 : loss:  0.72857076 	 acc:  0.67
4200 : loss:  0.77761286 	 acc:  0.64
4240 : loss:  0.74228525 	 acc:  0.65
4280 : loss:  0.64677763 	 acc:  0.71
4320 : loss:  0.9955212 	 acc:  0.56
4360 : loss:  0.67018247 	 acc:  0.67
4400 : loss:  0.84090453 	 acc:  0.62
4440 : loss:  0.8283593 	 acc:  0.63
4480 : loss:  0.80442244 	 acc:  0.64

Saving...
saved to models/CNN/pretrained_model.ckpt-4482

500 	 [363.96625877 225.51532696 371.41215487]
0 	val accuracy:  0.66811997 	 f_! score:  [0.72793252 0.45103065 0.74282431]

4520 : loss:  0.7853273 	 acc:  0.65
4560 : loss:  0.81140107 	 acc:  0.6
4600 : loss:  0.89808977 	 acc:  0.59
4640 : loss:  0.7449685 	 acc:  0.65
4680 : loss:  0.72202855 	 acc:  0.69
4720 : loss:  0.7393556 	 acc:  0.65
4760 : loss:  0.7015544 	 acc:  0.66
4800 : loss:  0.81609535 	 acc:  0.63
4840 : loss:  0.6787191 	 acc:  0.69
4880 : loss:  0.8301974 	 acc:  0.6
4920 : loss:  0.7728116 	 acc:  0.66
4960 : loss:  0.79850084 	 acc:  0.65
5000 : loss:  0.6858141 	 acc:  0.7
5040 : loss:  0.716529 	 acc:  0.7
5080 : loss:  0.7633607 	 acc:  0.64
5120 : loss:  0.75601023 	 acc:  0.6
5160 : loss:  0.8011946 	 acc:  0.65
5200 : loss:  0.7292663 	 acc:  0.67
5240 : loss:  0.69520396 	 acc:  0.7
5280 : loss:  0.7273378 	 acc:  0.64
5320 : loss:  0.85358864 	 acc:  0.57
5360 : loss:  0.8324869 	 acc:  0.63
5400 : loss:  0.7643379 	 acc:  0.64
5440 : loss:  0.6350802 	 acc:  0.75
5480 : loss:  0.72552437 	 acc:  0.71
5520 : loss:  0.67277986 	 acc:  0.7
5560 : loss:  0.77389586 	 acc:  0.66
5600 : loss:  0.74211484 	 acc:  0.7
5640 : loss:  0.7151214 	 acc:  0.71
5680 : loss:  0.7278541 	 acc:  0.69
5720 : loss:  0.6907576 	 acc:  0.7
5760 : loss:  0.766999 	 acc:  0.66
5800 : loss:  0.76620126 	 acc:  0.6
5840 : loss:  0.6297245 	 acc:  0.76
5880 : loss:  0.8267025 	 acc:  0.68
5920 : loss:  0.63393724 	 acc:  0.71
5960 : loss:  0.81212914 	 acc:  0.61
6000 : loss:  0.70356786 	 acc:  0.72
6040 : loss:  0.843694 	 acc:  0.62
6080 : loss:  0.75269943 	 acc:  0.65
6120 : loss:  0.7070372 	 acc:  0.67
6160 : loss:  0.8162444 	 acc:  0.61
6200 : loss:  0.8928086 	 acc:  0.63
6240 : loss:  0.59374005 	 acc:  0.77
6280 : loss:  0.6773817 	 acc:  0.65
6320 : loss:  0.7487183 	 acc:  0.62
6360 : loss:  0.80446434 	 acc:  0.64
6400 : loss:  0.66839385 	 acc:  0.63
6440 : loss:  0.6753023 	 acc:  0.7
6480 : loss:  0.7765242 	 acc:  0.66
6520 : loss:  0.71827996 	 acc:  0.69
6560 : loss:  0.70641595 	 acc:  0.68
6600 : loss:  0.8464723 	 acc:  0.66
6640 : loss:  0.68839735 	 acc:  0.77
6680 : loss:  0.7519533 	 acc:  0.65
6720 : loss:  0.8315971 	 acc:  0.67
6760 : loss:  0.8246868 	 acc:  0.67
6800 : loss:  0.73495245 	 acc:  0.67
6840 : loss:  0.8278367 	 acc:  0.67
6880 : loss:  0.84371793 	 acc:  0.63
6920 : loss:  0.6666295 	 acc:  0.68
6960 : loss:  0.70533895 	 acc:  0.7
7000 : loss:  0.74010676 	 acc:  0.63
7040 : loss:  0.7288485 	 acc:  0.65
7080 : loss:  0.6891174 	 acc:  0.69
7120 : loss:  0.77515197 	 acc:  0.6
7160 : loss:  0.7042672 	 acc:  0.62
7200 : loss:  0.8090943 	 acc:  0.6
7240 : loss:  0.8136068 	 acc:  0.66
7280 : loss:  0.6490072 	 acc:  0.73
7320 : loss:  0.7963398 	 acc:  0.6
7360 : loss:  0.803618 	 acc:  0.65
7400 : loss:  0.630982 	 acc:  0.76
7440 : loss:  0.6527047 	 acc:  0.75
7480 : loss:  0.835662 	 acc:  0.61
7520 : loss:  0.6511781 	 acc:  0.74
7560 : loss:  0.6235168 	 acc:  0.79
7600 : loss:  0.76868993 	 acc:  0.65
7640 : loss:  0.7710054 	 acc:  0.64
7680 : loss:  0.7467438 	 acc:  0.64
7720 : loss:  0.6727815 	 acc:  0.7
7760 : loss:  0.78401023 	 acc:  0.66
7800 : loss:  0.74049157 	 acc:  0.65
7840 : loss:  0.7980774 	 acc:  0.73
7880 : loss:  0.75377107 	 acc:  0.63
7920 : loss:  0.6429005 	 acc:  0.71
7960 : loss:  0.7217175 	 acc:  0.63
8000 : loss:  0.7460027 	 acc:  0.68
8040 : loss:  0.8725344 	 acc:  0.63
8080 : loss:  0.6854741 	 acc:  0.73
8120 : loss:  0.6532977 	 acc:  0.7
8160 : loss:  0.6349529 	 acc:  0.71
8200 : loss:  0.82808614 	 acc:  0.61
8240 : loss:  0.7452675 	 acc:  0.65
8280 : loss:  0.8134716 	 acc:  0.66
8320 : loss:  0.7201172 	 acc:  0.67
8360 : loss:  0.7755468 	 acc:  0.65
8400 : loss:  0.75136495 	 acc:  0.66
8440 : loss:  0.8115663 	 acc:  0.58
8480 : loss:  0.75126624 	 acc:  0.64
8520 : loss:  0.77198815 	 acc:  0.62
8560 : loss:  0.83610153 	 acc:  0.65
8600 : loss:  0.75345916 	 acc:  0.72
8640 : loss:  0.80142206 	 acc:  0.66
8680 : loss:  0.7357925 	 acc:  0.7
8720 : loss:  0.68279 	 acc:  0.73
8760 : loss:  0.75958043 	 acc:  0.62
8800 : loss:  0.86584616 	 acc:  0.56
8840 : loss:  0.69305 	 acc:  0.65
8880 : loss:  0.7017448 	 acc:  0.71
8920 : loss:  0.68457913 	 acc:  0.69
8960 : loss:  0.76831746 	 acc:  0.65

Saving...
saved to models/CNN/pretrained_model.ckpt-8965

500 	 [362.93939704 256.42539061 378.67434135]
1 	val accuracy:  0.67916006 	 f_! score:  [0.72587879 0.51285078 0.75734868]

9000 : loss:  0.8502497 	 acc:  0.65
9040 : loss:  0.79695266 	 acc:  0.65
9080 : loss:  0.5392089 	 acc:  0.79
9120 : loss:  0.7189104 	 acc:  0.6
9160 : loss:  0.96212196 	 acc:  0.5
9200 : loss:  0.7673613 	 acc:  0.62
9240 : loss:  0.6920204 	 acc:  0.64
9280 : loss:  0.6149714 	 acc:  0.7
9320 : loss:  0.68612754 	 acc:  0.65
9360 : loss:  0.71151155 	 acc:  0.63
9400 : loss:  0.6684398 	 acc:  0.71
9440 : loss:  0.65828097 	 acc:  0.68
9480 : loss:  0.54279286 	 acc:  0.79
9520 : loss:  0.741229 	 acc:  0.65
9560 : loss:  0.77193964 	 acc:  0.65
9600 : loss:  0.5951675 	 acc:  0.74
9640 : loss:  0.6613612 	 acc:  0.72
9680 : loss:  0.7683294 	 acc:  0.63
9720 : loss:  0.6304476 	 acc:  0.69
9760 : loss:  0.68737227 	 acc:  0.78
9800 : loss:  0.66880625 	 acc:  0.68
9840 : loss:  0.8255656 	 acc:  0.61
9880 : loss:  0.5644685 	 acc:  0.76
9920 : loss:  0.600386 	 acc:  0.79
9960 : loss:  0.68611723 	 acc:  0.7
10000 : loss:  0.8110565 	 acc:  0.64
10040 : loss:  0.6672671 	 acc:  0.7
10080 : loss:  0.58850837 	 acc:  0.78
10120 : loss:  0.668124 	 acc:  0.73
10160 : loss:  0.70666414 	 acc:  0.69
10200 : loss:  0.7300679 	 acc:  0.69
10240 : loss:  0.67510736 	 acc:  0.69
10280 : loss:  0.68183565 	 acc:  0.68
10320 : loss:  0.6858229 	 acc:  0.75
10360 : loss:  0.7159269 	 acc:  0.72
10400 : loss:  0.7609903 	 acc:  0.67
10440 : loss:  0.73854756 	 acc:  0.63
10480 : loss:  0.6859143 	 acc:  0.67
10520 : loss:  0.6825945 	 acc:  0.67
10560 : loss:  0.7648872 	 acc:  0.62
10600 : loss:  0.7168813 	 acc:  0.67
10640 : loss:  0.63486075 	 acc:  0.73
10680 : loss:  0.7366459 	 acc:  0.71
10720 : loss:  0.6082033 	 acc:  0.73
10760 : loss:  0.81941336 	 acc:  0.65
10800 : loss:  0.68571997 	 acc:  0.71
10840 : loss:  0.71694285 	 acc:  0.63
10880 : loss:  0.66610396 	 acc:  0.71
10920 : loss:  0.8717646 	 acc:  0.61
10960 : loss:  0.7811718 	 acc:  0.65
11000 : loss:  0.66401947 	 acc:  0.73
11040 : loss:  0.7285159 	 acc:  0.68
11080 : loss:  0.7241301 	 acc:  0.75
11120 : loss:  0.71160614 	 acc:  0.67
11160 : loss:  0.6316498 	 acc:  0.67
11200 : loss:  0.65036374 	 acc:  0.75
11240 : loss:  0.662901 	 acc:  0.71
11280 : loss:  0.5384289 	 acc:  0.77
11320 : loss:  0.64610946 	 acc:  0.72
11360 : loss:  0.6119229 	 acc:  0.73
11400 : loss:  0.7334657 	 acc:  0.62
11440 : loss:  0.63861257 	 acc:  0.68
11480 : loss:  0.8746033 	 acc:  0.57
11520 : loss:  0.65631837 	 acc:  0.68
11560 : loss:  0.71231794 	 acc:  0.65
11600 : loss:  0.6542152 	 acc:  0.77
11640 : loss:  0.78232163 	 acc:  0.66
11680 : loss:  0.5758058 	 acc:  0.79
11720 : loss:  0.69888014 	 acc:  0.74
11760 : loss:  0.73343897 	 acc:  0.62
11800 : loss:  0.6657107 	 acc:  0.77
11840 : loss:  0.61950034 	 acc:  0.73
11880 : loss:  0.71411324 	 acc:  0.71
11920 : loss:  0.7178051 	 acc:  0.71
11960 : loss:  0.6978921 	 acc:  0.77
12000 : loss:  0.6236509 	 acc:  0.76
12040 : loss:  0.6472891 	 acc:  0.73
12080 : loss:  0.7273715 	 acc:  0.69
12120 : loss:  0.781269 	 acc:  0.69
12160 : loss:  0.6194025 	 acc:  0.7
12200 : loss:  0.7580373 	 acc:  0.65
12240 : loss:  0.76618004 	 acc:  0.62
12280 : loss:  0.64598 	 acc:  0.7
12320 : loss:  0.6349781 	 acc:  0.73
12360 : loss:  0.64487404 	 acc:  0.71
12400 : loss:  0.66801524 	 acc:  0.73
12440 : loss:  0.701582 	 acc:  0.71
12480 : loss:  0.8694924 	 acc:  0.64
12520 : loss:  0.8262241 	 acc:  0.6
12560 : loss:  0.69042426 	 acc:  0.64
12600 : loss:  0.69543666 	 acc:  0.72
12640 : loss:  0.7295744 	 acc:  0.67
12680 : loss:  0.66775274 	 acc:  0.71
12720 : loss:  0.6271679 	 acc:  0.69
12760 : loss:  0.70094216 	 acc:  0.7
12800 : loss:  0.71779877 	 acc:  0.71
12840 : loss:  0.71865004 	 acc:  0.65
12880 : loss:  0.6664951 	 acc:  0.7
12920 : loss:  0.69045305 	 acc:  0.65
12960 : loss:  0.8438873 	 acc:  0.62
13000 : loss:  0.7267311 	 acc:  0.71
13040 : loss:  0.62424046 	 acc:  0.74
13080 : loss:  0.7150155 	 acc:  0.72
13120 : loss:  0.7841623 	 acc:  0.57
13160 : loss:  0.65949136 	 acc:  0.7
13200 : loss:  0.5968481 	 acc:  0.77
13240 : loss:  0.6225105 	 acc:  0.68
13280 : loss:  0.667306 	 acc:  0.74
13320 : loss:  0.6865007 	 acc:  0.71
13360 : loss:  0.7342568 	 acc:  0.68
13400 : loss:  0.6415119 	 acc:  0.72
13440 : loss:  0.7330581 	 acc:  0.66

Saving...
saved to models/CNN/pretrained_model.ckpt-13448

500 	 [369.38701461 241.20296439 383.70110017]
2 	val accuracy:  0.68666 	 f_! score:  [0.73877403 0.48240593 0.7674022 ]

13480 : loss:  0.6702469 	 acc:  0.7
13520 : loss:  0.7057858 	 acc:  0.68
13560 : loss:  0.71955186 	 acc:  0.7
13600 : loss:  0.7522343 	 acc:  0.63
13640 : loss:  0.765158 	 acc:  0.69
13680 : loss:  0.663126 	 acc:  0.69
13720 : loss:  0.69955117 	 acc:  0.7
13760 : loss:  0.73046494 	 acc:  0.68
13800 : loss:  0.6379083 	 acc:  0.68
13840 : loss:  0.62713116 	 acc:  0.76
13880 : loss:  0.7375948 	 acc:  0.73
13920 : loss:  0.7257559 	 acc:  0.71
13960 : loss:  0.66064805 	 acc:  0.73
14000 : loss:  0.6693313 	 acc:  0.69
14040 : loss:  0.6808344 	 acc:  0.73
14080 : loss:  0.6625527 	 acc:  0.69
14120 : loss:  0.76657575 	 acc:  0.65
14160 : loss:  0.68057823 	 acc:  0.69
14200 : loss:  0.6171175 	 acc:  0.74
14240 : loss:  0.7382917 	 acc:  0.71
14280 : loss:  0.7099176 	 acc:  0.72
14320 : loss:  0.80121094 	 acc:  0.69
14360 : loss:  0.6645505 	 acc:  0.72
14400 : loss:  0.68519855 	 acc:  0.64
14440 : loss:  0.5979234 	 acc:  0.76
14480 : loss:  0.5758228 	 acc:  0.78
14520 : loss:  0.78751016 	 acc:  0.6
14560 : loss:  0.60175204 	 acc:  0.78
14600 : loss:  0.7474907 	 acc:  0.65
14640 : loss:  0.6323261 	 acc:  0.75
14680 : loss:  0.6222836 	 acc:  0.74
14720 : loss:  0.6378632 	 acc:  0.72
14760 : loss:  0.7867991 	 acc:  0.68
14800 : loss:  0.6213179 	 acc:  0.74
14840 : loss:  0.5971955 	 acc:  0.8
14880 : loss:  0.5481383 	 acc:  0.78
14920 : loss:  0.6099503 	 acc:  0.77
14960 : loss:  0.6449038 	 acc:  0.7
15000 : loss:  0.66342425 	 acc:  0.68
15040 : loss:  0.6534762 	 acc:  0.74
15080 : loss:  0.6214906 	 acc:  0.69
15120 : loss:  0.754406 	 acc:  0.71
15160 : loss:  0.8133651 	 acc:  0.64
15200 : loss:  0.6559637 	 acc:  0.75
15240 : loss:  0.85299486 	 acc:  0.65
15280 : loss:  0.6427929 	 acc:  0.68
15320 : loss:  0.66882056 	 acc:  0.65
15360 : loss:  0.7847363 	 acc:  0.67
15400 : loss:  0.7082646 	 acc:  0.7
15440 : loss:  0.694555 	 acc:  0.72
15480 : loss:  0.68944705 	 acc:  0.73
15520 : loss:  0.7743069 	 acc:  0.63
15560 : loss:  0.6901763 	 acc:  0.67
15600 : loss:  0.78328264 	 acc:  0.69
15640 : loss:  0.68083495 	 acc:  0.75
15680 : loss:  0.58648014 	 acc:  0.75
15720 : loss:  0.6594007 	 acc:  0.71
15760 : loss:  0.705413 	 acc:  0.71
15800 : loss:  0.5895143 	 acc:  0.75
15840 : loss:  0.6523247 	 acc:  0.74
15880 : loss:  0.6502103 	 acc:  0.69
15920 : loss:  0.7308481 	 acc:  0.65
15960 : loss:  0.6198189 	 acc:  0.69
16000 : loss:  0.70385957 	 acc:  0.66
16040 : loss:  0.68318045 	 acc:  0.72
16080 : loss:  0.7881034 	 acc:  0.62
16120 : loss:  0.810887 	 acc:  0.68
16160 : loss:  0.70603585 	 acc:  0.7
16200 : loss:  0.71167105 	 acc:  0.66
16240 : loss:  0.7905859 	 acc:  0.63
16280 : loss:  0.64050096 	 acc:  0.76
16320 : loss:  0.7031214 	 acc:  0.66
16360 : loss:  0.69370043 	 acc:  0.69
16400 : loss:  0.64203525 	 acc:  0.71
16440 : loss:  0.6799618 	 acc:  0.71
16480 : loss:  0.76303625 	 acc:  0.61
16520 : loss:  0.6528572 	 acc:  0.71
16560 : loss:  0.65627396 	 acc:  0.68
16600 : loss:  0.7132322 	 acc:  0.68
16640 : loss:  0.6942699 	 acc:  0.7
16680 : loss:  0.6668344 	 acc:  0.71
16720 : loss:  0.58365166 	 acc:  0.75
16760 : loss:  0.71195054 	 acc:  0.69
16800 : loss:  0.6842815 	 acc:  0.62
16840 : loss:  0.6840183 	 acc:  0.73
16880 : loss:  0.5916026 	 acc:  0.77
16920 : loss:  0.73870397 	 acc:  0.61
16960 : loss:  0.6902118 	 acc:  0.69
17000 : loss:  0.72186834 	 acc:  0.63
17040 : loss:  0.6082921 	 acc:  0.72
17080 : loss:  0.59879625 	 acc:  0.72
17120 : loss:  0.69100523 	 acc:  0.65
17160 : loss:  0.6709623 	 acc:  0.74
17200 : loss:  0.7064428 	 acc:  0.75
17240 : loss:  0.6132849 	 acc:  0.75
17280 : loss:  0.6553938 	 acc:  0.7
17320 : loss:  0.71204597 	 acc:  0.69
17360 : loss:  0.6587175 	 acc:  0.71
17400 : loss:  0.9264676 	 acc:  0.54
17440 : loss:  0.67234004 	 acc:  0.68
17480 : loss:  0.63182014 	 acc:  0.69
17520 : loss:  0.60966074 	 acc:  0.68
17560 : loss:  0.73791224 	 acc:  0.69
17600 : loss:  0.7490537 	 acc:  0.66
17640 : loss:  0.69215846 	 acc:  0.67
17680 : loss:  0.7192184 	 acc:  0.72
17720 : loss:  0.6627325 	 acc:  0.64
17760 : loss:  0.7780563 	 acc:  0.56
17800 : loss:  0.7569254 	 acc:  0.59
17840 : loss:  0.59671485 	 acc:  0.73
17880 : loss:  0.79344255 	 acc:  0.65
17920 : loss:  0.69057983 	 acc:  0.68

Saving...
saved to models/CNN/pretrained_model.ckpt-17931

500 	 [369.94274234 241.56508878 387.03832598]
3 	val accuracy:  0.69071996 	 f_! score:  [0.73988548 0.48313018 0.77407665]

500 	 [369.94274234 241.56508878 387.03832598]
500 	 [360.65442698 272.53817041 373.44005936]
500 	 [382.69532646 221.18286793 404.30099891]
500 	 [18794. 13823. 17383.]
--- Test   Review ---
0.69071996
f1:  [0.73988548 0.48313018 0.77407665]
65730
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-10-14 18:49:40.561502: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-10-14 18:49:40.561548: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-14 18:49:40.561556: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-10-14 18:49:40.561560: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-10-14 18:49:40.561659: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2905 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
models/CNN/pretrained_model.ckpt-17931
2018-10-14 18:49:44.367763: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-10-14 18:49:44.367804: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-14 18:49:44.367811: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-10-14 18:49:44.367815: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-10-14 18:49:44.367907: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2905 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
acc:  0.38
/home/yannik/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
acc:  0.34
acc:  0.33
acc:  0.3
acc:  0.29
acc:  0.35
acc:  0.35
acc:  0.36
acc:  0.39
acc:  0.32
acc:  0.38
acc:  0.23
acc:  0.36
acc:  0.3
acc:  0.39
acc:  0.38
acc:  0.41
acc:  0.31
acc:  0.24
acc:  0.39
acc:  0.33
acc:  0.33
acc:  0.34
acc:  0.33
acc:  0.37
acc:  0.31
acc:  0.34
acc:  0.33
acc:  0.35
acc:  0.36
acc:  0.39
acc:  0.31
acc:  0.42
acc:  0.29
acc:  0.26
acc:  0.33
acc:  0.31
acc:  0.43
acc:  0.34
acc:  0.31
acc:  0.32
acc:  0.36
acc:  0.35
acc:  0.29
acc:  0.31
acc:  0.38
acc:  0.24
acc:  0.27
acc:  0.4
acc:  0.37
acc:  0.35
acc:  0.27
acc:  0.36
acc:  0.35
acc:  0.34
acc:  0.37
acc:  0.3
acc:  0.21
acc:  0.41
acc:  0.33
acc:  0.31
acc:  0.37
acc:  0.35
acc:  0.28
acc:  0.33
acc:  0.36
acc:  0.36
acc:  0.44
acc:  0.31
acc:  0.35
acc:  0.35
acc:  0.25
acc:  0.43
acc:  0.32
acc:  0.31
acc:  0.26
acc:  0.35
acc:  0.33
acc:  0.32
acc:  0.32
acc:  0.29
acc:  0.32
acc:  0.35
acc:  0.34
acc:  0.36
acc:  0.37
acc:  0.22
acc:  0.34
acc:  0.31
acc:  0.37
acc:  0.36
acc:  0.32
acc:  0.33
acc:  0.32
acc:  0.29
acc:  0.37
acc:  0.28
acc:  0.35
acc:  0.32
acc:  0.4

100 	 [ 0.          0.         49.97740219]
val accuracy:  0.3347 	 f_! score:  [0.         0.         0.49977402]

100 	 [ 0.          0.         49.97740219]
100 	 [ 0.    0.   33.47]
100 	 [  0.   0. 100.]
100 	 [3307. 3346. 3347.]
---just Test  Twitter ---
0.3347
f1:  [0.         0.         0.49977402]
--- 8.400914192199707 seconds ---
65730
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-10-14 18:49:57.559811: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-10-14 18:49:57.559847: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-14 18:49:57.559853: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-10-14 18:49:57.559857: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-10-14 18:49:57.559938: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2905 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
models/CNN/pretrained_model.ckpt-17931
2018-10-14 18:50:00.383503: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-10-14 18:50:00.383554: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-14 18:50:00.383560: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-10-14 18:50:00.383564: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-10-14 18:50:00.383650: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2905 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
0 : loss:  1.0948321 	 acc:  0.28
2 : loss:  1.0989789 	 acc:  0.35
4 : loss:  1.1038288 	 acc:  0.36
6 : loss:  1.0983198 	 acc:  0.32
8 : loss:  1.088247 	 acc:  0.42
10 : loss:  1.1370988 	 acc:  0.26
12 : loss:  1.1065779 	 acc:  0.36
14 : loss:  1.110108 	 acc:  0.35
16 : loss:  1.0954727 	 acc:  0.28
18 : loss:  1.0915065 	 acc:  0.34
20 : loss:  1.08088 	 acc:  0.41
22 : loss:  1.0700202 	 acc:  0.45
24 : loss:  1.0620105 	 acc:  0.41
26 : loss:  1.1083549 	 acc:  0.3
28 : loss:  1.1148007 	 acc:  0.3
30 : loss:  1.1150308 	 acc:  0.32
32 : loss:  1.0944242 	 acc:  0.38
34 : loss:  1.1023128 	 acc:  0.3
36 : loss:  1.0707493 	 acc:  0.45
38 : loss:  1.0794549 	 acc:  0.36
2018-10-14 18:50:17.877432: W tensorflow/core/common_runtime/bfc_allocator.cc:219] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.31GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
40 : loss:  1.0560962 	 acc:  0.43333334
42 : loss:  1.0831141 	 acc:  0.41
44 : loss:  1.0615761 	 acc:  0.47
46 : loss:  1.0837172 	 acc:  0.41
48 : loss:  1.0713626 	 acc:  0.4
50 : loss:  1.0580403 	 acc:  0.48
52 : loss:  1.0540135 	 acc:  0.52
54 : loss:  1.0715388 	 acc:  0.42
56 : loss:  1.0869983 	 acc:  0.4
58 : loss:  1.0771891 	 acc:  0.43
60 : loss:  1.0900095 	 acc:  0.34
62 : loss:  1.0582091 	 acc:  0.45
64 : loss:  1.0878448 	 acc:  0.4
66 : loss:  1.0480678 	 acc:  0.48
68 : loss:  1.0448147 	 acc:  0.41
70 : loss:  1.0940661 	 acc:  0.35
72 : loss:  1.0403107 	 acc:  0.5
74 : loss:  1.0665532 	 acc:  0.35
76 : loss:  1.0569997 	 acc:  0.44
78 : loss:  1.0681797 	 acc:  0.34
80 : loss:  1.0430342 	 acc:  0.48
82 : loss:  1.0899178 	 acc:  0.32
84 : loss:  1.0756291 	 acc:  0.44
86 : loss:  1.0324416 	 acc:  0.49
88 : loss:  1.046692 	 acc:  0.43
90 : loss:  1.0529002 	 acc:  0.46
92 : loss:  1.0789226 	 acc:  0.37
94 : loss:  1.0341637 	 acc:  0.48
96 : loss:  1.0568829 	 acc:  0.45
98 : loss:  1.0759429 	 acc:  0.34
100 : loss:  1.0638317 	 acc:  0.4
102 : loss:  1.0567579 	 acc:  0.44
104 : loss:  1.0714062 	 acc:  0.37
106 : loss:  1.1063983 	 acc:  0.36
108 : loss:  1.0311776 	 acc:  0.47
110 : loss:  0.99898386 	 acc:  0.48
112 : loss:  1.0653399 	 acc:  0.41
114 : loss:  1.0520985 	 acc:  0.43
116 : loss:  1.0212224 	 acc:  0.49
118 : loss:  1.0639355 	 acc:  0.42
120 : loss:  1.0897225 	 acc:  0.36
122 : loss:  1.0209404 	 acc:  0.48
124 : loss:  1.0640091 	 acc:  0.39
126 : loss:  1.0224923 	 acc:  0.49
128 : loss:  1.0527264 	 acc:  0.39
130 : loss:  1.0274906 	 acc:  0.48
132 : loss:  1.0333003 	 acc:  0.45
134 : loss:  1.0652236 	 acc:  0.37
136 : loss:  1.0475302 	 acc:  0.48
138 : loss:  1.0714985 	 acc:  0.39
140 : loss:  1.031298 	 acc:  0.47
142 : loss:  1.0688854 	 acc:  0.39
144 : loss:  1.0708966 	 acc:  0.38
146 : loss:  1.1030512 	 acc:  0.38
148 : loss:  1.0409305 	 acc:  0.41
150 : loss:  1.1084807 	 acc:  0.39
152 : loss:  1.0408483 	 acc:  0.44
154 : loss:  1.0261772 	 acc:  0.49
156 : loss:  0.9962378 	 acc:  0.55
158 : loss:  1.0510643 	 acc:  0.45
160 : loss:  1.0896858 	 acc:  0.4
162 : loss:  1.0501381 	 acc:  0.43
164 : loss:  1.0925499 	 acc:  0.36
166 : loss:  1.0358256 	 acc:  0.39
168 : loss:  1.0623399 	 acc:  0.34
170 : loss:  1.0750153 	 acc:  0.41
172 : loss:  1.0415378 	 acc:  0.43
174 : loss:  1.0844946 	 acc:  0.41
176 : loss:  1.00775 	 acc:  0.4
178 : loss:  1.0055147 	 acc:  0.46
180 : loss:  1.0266192 	 acc:  0.48
182 : loss:  1.0381016 	 acc:  0.42
184 : loss:  1.0461903 	 acc:  0.44
186 : loss:  1.0974025 	 acc:  0.39
188 : loss:  1.0100608 	 acc:  0.44
190 : loss:  1.0550306 	 acc:  0.35
192 : loss:  1.0680325 	 acc:  0.41
194 : loss:  1.0379052 	 acc:  0.45
196 : loss:  1.016565 	 acc:  0.43
198 : loss:  1.0548085 	 acc:  0.39
200 : loss:  1.0731989 	 acc:  0.42
202 : loss:  1.037695 	 acc:  0.49
204 : loss:  1.0746243 	 acc:  0.41
206 : loss:  1.0757339 	 acc:  0.39
208 : loss:  1.058778 	 acc:  0.38
210 : loss:  1.0570279 	 acc:  0.44
212 : loss:  1.0289855 	 acc:  0.39
214 : loss:  1.0510249 	 acc:  0.44
216 : loss:  1.0186055 	 acc:  0.44
218 : loss:  1.0884836 	 acc:  0.39
220 : loss:  1.0293776 	 acc:  0.47
222 : loss:  0.99747187 	 acc:  0.48
224 : loss:  1.0380979 	 acc:  0.5
226 : loss:  1.0696105 	 acc:  0.4
228 : loss:  1.0609906 	 acc:  0.41
230 : loss:  1.0790464 	 acc:  0.42
232 : loss:  1.0476688 	 acc:  0.45
234 : loss:  1.0522528 	 acc:  0.39
236 : loss:  1.0345308 	 acc:  0.42
238 : loss:  1.0437676 	 acc:  0.45
240 : loss:  1.085715 	 acc:  0.41
242 : loss:  1.0228744 	 acc:  0.42
244 : loss:  1.0595073 	 acc:  0.42
246 : loss:  1.1024858 	 acc:  0.42
248 : loss:  1.0962925 	 acc:  0.37
250 : loss:  1.0795662 	 acc:  0.34
252 : loss:  1.0316014 	 acc:  0.45
254 : loss:  1.0313193 	 acc:  0.48
256 : loss:  1.0350758 	 acc:  0.44
258 : loss:  1.0644755 	 acc:  0.38
260 : loss:  1.0533081 	 acc:  0.41
262 : loss:  1.0516435 	 acc:  0.35
264 : loss:  1.0636889 	 acc:  0.43
266 : loss:  1.0277052 	 acc:  0.46
268 : loss:  1.0685257 	 acc:  0.44
270 : loss:  1.0730908 	 acc:  0.37
272 : loss:  0.98443115 	 acc:  0.56
274 : loss:  1.0153482 	 acc:  0.53
276 : loss:  1.0588015 	 acc:  0.41
278 : loss:  1.0350497 	 acc:  0.49
280 : loss:  1.029672 	 acc:  0.53
282 : loss:  1.0162741 	 acc:  0.54
284 : loss:  1.0279076 	 acc:  0.5
286 : loss:  1.0808319 	 acc:  0.4
288 : loss:  1.0189967 	 acc:  0.46
290 : loss:  1.0563513 	 acc:  0.44
292 : loss:  1.041039 	 acc:  0.43
294 : loss:  1.0729895 	 acc:  0.41
296 : loss:  1.0063305 	 acc:  0.49
298 : loss:  1.0444443 	 acc:  0.44
300 : loss:  1.0089293 	 acc:  0.52
302 : loss:  1.0459539 	 acc:  0.41
304 : loss:  1.0477597 	 acc:  0.43
306 : loss:  1.0961931 	 acc:  0.36
308 : loss:  1.0766912 	 acc:  0.36
310 : loss:  1.0412209 	 acc:  0.48
312 : loss:  1.0063707 	 acc:  0.41
314 : loss:  1.05 	 acc:  0.35
316 : loss:  1.038661 	 acc:  0.47
318 : loss:  1.061969 	 acc:  0.4
320 : loss:  1.0612887 	 acc:  0.42
322 : loss:  1.0562514 	 acc:  0.47
324 : loss:  1.063551 	 acc:  0.44
326 : loss:  1.0098803 	 acc:  0.52
328 : loss:  1.0774727 	 acc:  0.38
330 : loss:  1.0736238 	 acc:  0.38
332 : loss:  1.0192255 	 acc:  0.5
334 : loss:  1.0444942 	 acc:  0.44
336 : loss:  1.0596751 	 acc:  0.4
338 : loss:  1.0824099 	 acc:  0.39
340 : loss:  1.0623871 	 acc:  0.44
342 : loss:  1.0484762 	 acc:  0.44
344 : loss:  1.004569 	 acc:  0.47
346 : loss:  1.0520595 	 acc:  0.39
348 : loss:  1.0007195 	 acc:  0.44
350 : loss:  1.0683084 	 acc:  0.39
352 : loss:  1.0589837 	 acc:  0.4
354 : loss:  1.0672653 	 acc:  0.45
356 : loss:  1.0409787 	 acc:  0.43
358 : loss:  1.0167385 	 acc:  0.58
360 : loss:  1.0138801 	 acc:  0.45
362 : loss:  1.0576291 	 acc:  0.47
364 : loss:  1.0609139 	 acc:  0.4
366 : loss:  1.0466486 	 acc:  0.39
368 : loss:  1.0592661 	 acc:  0.48
370 : loss:  1.0255567 	 acc:  0.45
acc:  0.44
acc:  0.51
acc:  0.42
acc:  0.51
acc:  0.48
acc:  0.42
acc:  0.45
acc:  0.41
acc:  0.49
acc:  0.49
acc:  0.46
acc:  0.38
acc:  0.38
acc:  0.53
acc:  0.47
acc:  0.47
acc:  0.44
acc:  0.37
acc:  0.38
acc:  0.47
acc:  0.4
acc:  0.42
acc:  0.38
acc:  0.48
acc:  0.47
acc:  0.49
acc:  0.39
acc:  0.4
acc:  0.45
acc:  0.36
acc:  0.51
acc:  0.51
acc:  0.51
acc:  0.5
acc:  0.44
acc:  0.48
acc:  0.39
acc:  0.43
acc:  0.43
acc:  0.37
acc:  0.44
acc:  0.41
acc:  0.35
acc:  0.48
acc:  0.42
acc:  0.44
acc:  0.41
acc:  0.48
acc:  0.47
acc:  0.39
acc:  0.36
acc:  0.41
acc:  0.48
acc:  0.46
acc:  0.47
acc:  0.48
acc:  0.49
acc:  0.49
acc:  0.45
acc:  0.41
acc:  0.39
acc:  0.45
acc:  0.5
acc:  0.38
acc:  0.44
acc:  0.4
acc:  0.39
acc:  0.39
acc:  0.39
acc:  0.41
acc:  0.42
acc:  0.42
acc:  0.53
acc:  0.49
acc:  0.46
acc:  0.45
acc:  0.4
acc:  0.48
acc:  0.41
acc:  0.36
acc:  0.43
acc:  0.34
acc:  0.41
acc:  0.37
acc:  0.36
acc:  0.53
acc:  0.46
acc:  0.44
acc:  0.43
acc:  0.51
acc:  0.46
acc:  0.47
acc:  0.41
acc:  0.41
acc:  0.49
acc:  0.5
acc:  0.39
acc:  0.44
acc:  0.42
acc:  0.36

100 	 [51.49258221 52.51971041  0.34251178]
0 	val accuracy:  0.43760002 	 f_! score:  [0.51492582 0.5251971  0.00342512]

372 : loss:  1.0658854 	 acc:  0.37
374 : loss:  1.0157694 	 acc:  0.45
376 : loss:  1.0251526 	 acc:  0.5
378 : loss:  1.0393758 	 acc:  0.45
380 : loss:  1.0607527 	 acc:  0.42
382 : loss:  1.0499424 	 acc:  0.39
384 : loss:  0.9956995 	 acc:  0.5
386 : loss:  1.0494002 	 acc:  0.44
388 : loss:  1.0543174 	 acc:  0.42
390 : loss:  1.0625675 	 acc:  0.37
392 : loss:  0.98841786 	 acc:  0.53
394 : loss:  1.0603228 	 acc:  0.39
396 : loss:  1.0037391 	 acc:  0.4
398 : loss:  1.0598891 	 acc:  0.38
400 : loss:  1.0322218 	 acc:  0.39
402 : loss:  1.041054 	 acc:  0.45
404 : loss:  1.1326257 	 acc:  0.34
406 : loss:  1.0756644 	 acc:  0.32
408 : loss:  1.0904799 	 acc:  0.36
410 : loss:  1.0735462 	 acc:  0.4
412 : loss:  1.0869021 	 acc:  0.35
414 : loss:  1.056102 	 acc:  0.43
416 : loss:  1.0684807 	 acc:  0.4
418 : loss:  0.991447 	 acc:  0.54
420 : loss:  1.0584874 	 acc:  0.43
422 : loss:  1.0885252 	 acc:  0.4
424 : loss:  1.0502871 	 acc:  0.47
426 : loss:  1.0872644 	 acc:  0.39
428 : loss:  1.089317 	 acc:  0.37
430 : loss:  1.1231601 	 acc:  0.34
432 : loss:  1.0123417 	 acc:  0.46
434 : loss:  1.0013334 	 acc:  0.51
436 : loss:  1.0223528 	 acc:  0.47
438 : loss:  0.988768 	 acc:  0.48
440 : loss:  1.064049 	 acc:  0.44
442 : loss:  1.089559 	 acc:  0.42
444 : loss:  1.0894966 	 acc:  0.42
446 : loss:  1.0699162 	 acc:  0.39
448 : loss:  1.0418677 	 acc:  0.45
450 : loss:  0.96952516 	 acc:  0.51
452 : loss:  1.0570931 	 acc:  0.4
454 : loss:  1.1186376 	 acc:  0.31
456 : loss:  1.0058228 	 acc:  0.47
458 : loss:  1.0081896 	 acc:  0.45
460 : loss:  1.0342813 	 acc:  0.45
462 : loss:  1.0250992 	 acc:  0.45
464 : loss:  1.0588645 	 acc:  0.38
466 : loss:  0.985178 	 acc:  0.5
468 : loss:  1.0232004 	 acc:  0.43
470 : loss:  1.0252894 	 acc:  0.48
472 : loss:  1.0677154 	 acc:  0.46
474 : loss:  1.041458 	 acc:  0.43
476 : loss:  0.99101347 	 acc:  0.46
478 : loss:  1.0288877 	 acc:  0.49
480 : loss:  1.0783112 	 acc:  0.38
482 : loss:  1.0874112 	 acc:  0.41
484 : loss:  0.9779285 	 acc:  0.5
486 : loss:  1.0107691 	 acc:  0.48
488 : loss:  1.0990533 	 acc:  0.31
490 : loss:  0.97569406 	 acc:  0.5
492 : loss:  0.9613704 	 acc:  0.54
494 : loss:  1.0520419 	 acc:  0.41
496 : loss:  1.0559391 	 acc:  0.43
498 : loss:  1.0349599 	 acc:  0.47
500 : loss:  1.0598078 	 acc:  0.42
502 : loss:  1.0250682 	 acc:  0.49
504 : loss:  1.0627831 	 acc:  0.39
506 : loss:  1.1107321 	 acc:  0.39
508 : loss:  1.028015 	 acc:  0.45
510 : loss:  1.0621885 	 acc:  0.41
512 : loss:  1.0193139 	 acc:  0.48
514 : loss:  1.0283871 	 acc:  0.44
516 : loss:  1.0114784 	 acc:  0.47
518 : loss:  1.0521657 	 acc:  0.47
520 : loss:  1.0474446 	 acc:  0.48
522 : loss:  1.0616992 	 acc:  0.44
524 : loss:  1.0253161 	 acc:  0.46
526 : loss:  1.0354888 	 acc:  0.46
528 : loss:  0.9865894 	 acc:  0.54
530 : loss:  0.9991743 	 acc:  0.51
532 : loss:  1.0127891 	 acc:  0.51
534 : loss:  0.97943574 	 acc:  0.54
536 : loss:  1.0510882 	 acc:  0.33
538 : loss:  1.0525082 	 acc:  0.48
540 : loss:  1.0302622 	 acc:  0.47
542 : loss:  0.988644 	 acc:  0.52
544 : loss:  1.116657 	 acc:  0.42
546 : loss:  1.0465161 	 acc:  0.49
548 : loss:  1.0518311 	 acc:  0.38
550 : loss:  1.0865318 	 acc:  0.36
552 : loss:  1.0132761 	 acc:  0.46
554 : loss:  1.0308566 	 acc:  0.5
556 : loss:  1.0529912 	 acc:  0.4
558 : loss:  1.003825 	 acc:  0.47
560 : loss:  1.0418239 	 acc:  0.36666667
562 : loss:  1.0157837 	 acc:  0.44
564 : loss:  0.9952973 	 acc:  0.53
566 : loss:  1.0407796 	 acc:  0.37
568 : loss:  1.0274999 	 acc:  0.43
570 : loss:  0.9838606 	 acc:  0.48
572 : loss:  1.0622226 	 acc:  0.43
574 : loss:  1.0380924 	 acc:  0.44
576 : loss:  1.0681978 	 acc:  0.42
578 : loss:  1.0278088 	 acc:  0.43
580 : loss:  1.0205417 	 acc:  0.44
582 : loss:  1.0112108 	 acc:  0.45
584 : loss:  1.0304439 	 acc:  0.41
586 : loss:  1.0764254 	 acc:  0.42
588 : loss:  1.0434506 	 acc:  0.43
590 : loss:  1.0134186 	 acc:  0.44
592 : loss:  1.1057615 	 acc:  0.32
594 : loss:  1.0429792 	 acc:  0.5
596 : loss:  1.0099865 	 acc:  0.47
598 : loss:  1.0560739 	 acc:  0.42
600 : loss:  1.0664592 	 acc:  0.41
602 : loss:  0.9922191 	 acc:  0.48
604 : loss:  1.0331746 	 acc:  0.47
606 : loss:  1.0678911 	 acc:  0.41
608 : loss:  1.0385226 	 acc:  0.41
610 : loss:  1.0762511 	 acc:  0.37
612 : loss:  1.0362835 	 acc:  0.47
614 : loss:  1.0841715 	 acc:  0.44
616 : loss:  1.0476921 	 acc:  0.42
618 : loss:  1.0401193 	 acc:  0.46
620 : loss:  1.0607905 	 acc:  0.42
622 : loss:  1.0093703 	 acc:  0.45
624 : loss:  1.036177 	 acc:  0.51
626 : loss:  1.0199289 	 acc:  0.47
628 : loss:  1.0438881 	 acc:  0.38
630 : loss:  1.0305171 	 acc:  0.45
632 : loss:  1.0092185 	 acc:  0.44
634 : loss:  1.0869298 	 acc:  0.42
636 : loss:  1.0739489 	 acc:  0.4
638 : loss:  1.0063145 	 acc:  0.45
640 : loss:  0.9891817 	 acc:  0.53
642 : loss:  1.0678588 	 acc:  0.37
644 : loss:  1.0290325 	 acc:  0.5
646 : loss:  1.0540855 	 acc:  0.38
648 : loss:  0.9763009 	 acc:  0.51
650 : loss:  1.0171965 	 acc:  0.5
652 : loss:  1.0255655 	 acc:  0.43
654 : loss:  1.0394746 	 acc:  0.43
656 : loss:  1.0199167 	 acc:  0.46
658 : loss:  0.98980516 	 acc:  0.47
660 : loss:  1.0178328 	 acc:  0.46
662 : loss:  1.0567224 	 acc:  0.41
664 : loss:  0.9961129 	 acc:  0.47
666 : loss:  1.0573058 	 acc:  0.42
668 : loss:  1.036589 	 acc:  0.41
670 : loss:  1.075066 	 acc:  0.44
672 : loss:  1.0296599 	 acc:  0.48
674 : loss:  1.0021719 	 acc:  0.46
676 : loss:  1.0399765 	 acc:  0.45
678 : loss:  0.9746884 	 acc:  0.55
680 : loss:  1.045051 	 acc:  0.39
682 : loss:  1.0636468 	 acc:  0.4
684 : loss:  1.0535576 	 acc:  0.4
686 : loss:  1.0391024 	 acc:  0.43
688 : loss:  1.0112693 	 acc:  0.54
690 : loss:  1.056365 	 acc:  0.41
692 : loss:  1.0321075 	 acc:  0.47
694 : loss:  1.0058993 	 acc:  0.46
696 : loss:  0.9536879 	 acc:  0.55
698 : loss:  0.96650946 	 acc:  0.55
700 : loss:  1.0029007 	 acc:  0.52
702 : loss:  1.0150398 	 acc:  0.48
704 : loss:  0.9763194 	 acc:  0.57
706 : loss:  1.030878 	 acc:  0.47
708 : loss:  1.0388484 	 acc:  0.48
710 : loss:  1.0034733 	 acc:  0.45
712 : loss:  1.0163772 	 acc:  0.46
714 : loss:  1.0424185 	 acc:  0.42
716 : loss:  1.083359 	 acc:  0.4
718 : loss:  1.0158921 	 acc:  0.42
720 : loss:  1.0249562 	 acc:  0.5
722 : loss:  1.034054 	 acc:  0.38
724 : loss:  1.0853556 	 acc:  0.47
726 : loss:  1.0460147 	 acc:  0.44
728 : loss:  1.0000376 	 acc:  0.45
730 : loss:  1.0131613 	 acc:  0.51
732 : loss:  1.0009677 	 acc:  0.53
734 : loss:  1.0210638 	 acc:  0.5
736 : loss:  1.0533123 	 acc:  0.41
738 : loss:  1.014359 	 acc:  0.48
740 : loss:  1.0403785 	 acc:  0.42
742 : loss:  1.0725611 	 acc:  0.39
acc:  0.49
acc:  0.42
acc:  0.56
acc:  0.46
acc:  0.46
acc:  0.38
acc:  0.57
acc:  0.35
acc:  0.46
acc:  0.39
acc:  0.4
acc:  0.46
acc:  0.54
acc:  0.42
acc:  0.45
acc:  0.5
acc:  0.45
acc:  0.45
acc:  0.51
acc:  0.43
acc:  0.43
acc:  0.48
acc:  0.42
acc:  0.47
acc:  0.49
acc:  0.53
acc:  0.47
acc:  0.38
acc:  0.48
acc:  0.39
acc:  0.44
acc:  0.43
acc:  0.51
acc:  0.5
acc:  0.42
acc:  0.44
acc:  0.37
acc:  0.51
acc:  0.43
acc:  0.46
acc:  0.43
acc:  0.52
acc:  0.43
acc:  0.45
acc:  0.49
acc:  0.37
acc:  0.45
acc:  0.37
acc:  0.55
acc:  0.41
acc:  0.48
acc:  0.43
acc:  0.5
acc:  0.44
acc:  0.48
acc:  0.39
acc:  0.4
acc:  0.4
acc:  0.44
acc:  0.5
acc:  0.43
acc:  0.47
acc:  0.38
acc:  0.48
acc:  0.55
acc:  0.38
acc:  0.48
acc:  0.39
acc:  0.46
acc:  0.48
acc:  0.44
acc:  0.5
acc:  0.47
acc:  0.41
acc:  0.51
acc:  0.45
acc:  0.39
acc:  0.4
acc:  0.45
acc:  0.46
acc:  0.4
acc:  0.4
acc:  0.57
acc:  0.6
acc:  0.39
acc:  0.43
acc:  0.42
acc:  0.44
acc:  0.39
acc:  0.46
acc:  0.53
acc:  0.48
acc:  0.47
acc:  0.48
acc:  0.48
acc:  0.42
acc:  0.48
acc:  0.38
acc:  0.49
acc:  0.52

100 	 [52.72875439 32.76844879 45.67747116]
1 	val accuracy:  0.45340005 	 f_! score:  [0.52728754 0.32768449 0.45677471]

744 : loss:  1.0125756 	 acc:  0.39
746 : loss:  1.0234587 	 acc:  0.41
748 : loss:  1.0091834 	 acc:  0.49
750 : loss:  1.0457006 	 acc:  0.4
752 : loss:  0.9775737 	 acc:  0.46
754 : loss:  1.0348239 	 acc:  0.52
756 : loss:  1.0618583 	 acc:  0.5
758 : loss:  1.0594053 	 acc:  0.45
760 : loss:  0.97173387 	 acc:  0.6
762 : loss:  1.0249592 	 acc:  0.42
764 : loss:  1.0605301 	 acc:  0.45
766 : loss:  1.0625263 	 acc:  0.41
768 : loss:  1.073781 	 acc:  0.42
770 : loss:  1.0662811 	 acc:  0.44
772 : loss:  1.0388743 	 acc:  0.38
774 : loss:  0.94942784 	 acc:  0.54
776 : loss:  1.012117 	 acc:  0.5
778 : loss:  1.0968158 	 acc:  0.41
780 : loss:  1.0525782 	 acc:  0.44
782 : loss:  1.0696362 	 acc:  0.44
784 : loss:  1.009415 	 acc:  0.45
786 : loss:  1.1025201 	 acc:  0.39
788 : loss:  1.0146127 	 acc:  0.5
790 : loss:  1.0313294 	 acc:  0.47
792 : loss:  0.9676795 	 acc:  0.55
794 : loss:  0.9627565 	 acc:  0.6
796 : loss:  1.0483643 	 acc:  0.38
798 : loss:  1.0401803 	 acc:  0.42
800 : loss:  1.0054526 	 acc:  0.45
802 : loss:  1.0784224 	 acc:  0.4
804 : loss:  1.0650345 	 acc:  0.4
806 : loss:  0.9972435 	 acc:  0.52
808 : loss:  1.016167 	 acc:  0.45
810 : loss:  1.0187061 	 acc:  0.45
812 : loss:  1.0554068 	 acc:  0.43
814 : loss:  1.026274 	 acc:  0.43
816 : loss:  1.0195327 	 acc:  0.5
818 : loss:  1.0007067 	 acc:  0.54
820 : loss:  1.0575029 	 acc:  0.44
822 : loss:  1.0207027 	 acc:  0.43
824 : loss:  1.027578 	 acc:  0.47
826 : loss:  1.0120316 	 acc:  0.48
828 : loss:  1.0336422 	 acc:  0.46
830 : loss:  1.0280776 	 acc:  0.44
832 : loss:  1.0330054 	 acc:  0.46
834 : loss:  1.0121932 	 acc:  0.44
836 : loss:  0.9888837 	 acc:  0.47
838 : loss:  1.0326419 	 acc:  0.48
840 : loss:  1.0272508 	 acc:  0.38
842 : loss:  0.9888725 	 acc:  0.56
844 : loss:  1.0408697 	 acc:  0.5
846 : loss:  1.0024405 	 acc:  0.44
848 : loss:  1.0738143 	 acc:  0.39
850 : loss:  1.0637666 	 acc:  0.4
852 : loss:  1.0084226 	 acc:  0.4
854 : loss:  1.0633075 	 acc:  0.41
856 : loss:  1.0495176 	 acc:  0.42
858 : loss:  1.005304 	 acc:  0.51
860 : loss:  0.9711901 	 acc:  0.56
862 : loss:  0.98796254 	 acc:  0.51
864 : loss:  1.0236361 	 acc:  0.44
866 : loss:  1.0413738 	 acc:  0.46
868 : loss:  1.0070442 	 acc:  0.5
870 : loss:  0.9914919 	 acc:  0.44
872 : loss:  1.0560889 	 acc:  0.44
874 : loss:  1.0125641 	 acc:  0.48
876 : loss:  1.0502964 	 acc:  0.39
878 : loss:  1.0393935 	 acc:  0.42
880 : loss:  0.98394454 	 acc:  0.47
882 : loss:  1.0380087 	 acc:  0.5
884 : loss:  1.1014365 	 acc:  0.37
886 : loss:  1.0204918 	 acc:  0.42
888 : loss:  1.0547732 	 acc:  0.4
890 : loss:  1.003017 	 acc:  0.49
892 : loss:  1.0683913 	 acc:  0.37
894 : loss:  1.0363204 	 acc:  0.46
896 : loss:  0.97010505 	 acc:  0.48
898 : loss:  1.0613421 	 acc:  0.39
900 : loss:  1.0117203 	 acc:  0.48
902 : loss:  1.0440098 	 acc:  0.51
904 : loss:  1.0291839 	 acc:  0.45
906 : loss:  1.0545243 	 acc:  0.38
908 : loss:  1.0821748 	 acc:  0.33
910 : loss:  1.11304 	 acc:  0.32
912 : loss:  0.96695876 	 acc:  0.53
914 : loss:  0.99738616 	 acc:  0.5
916 : loss:  1.0534818 	 acc:  0.4
918 : loss:  1.0346422 	 acc:  0.44
920 : loss:  1.0594052 	 acc:  0.45
922 : loss:  0.99616426 	 acc:  0.49
924 : loss:  1.0298762 	 acc:  0.42
926 : loss:  1.0330846 	 acc:  0.53
928 : loss:  0.98555845 	 acc:  0.49
930 : loss:  1.0115982 	 acc:  0.47
932 : loss:  1.0278507 	 acc:  0.43
934 : loss:  1.0367376 	 acc:  0.41
936 : loss:  0.9407879 	 acc:  0.54
938 : loss:  1.0055542 	 acc:  0.45
940 : loss:  1.036233 	 acc:  0.48
942 : loss:  1.0517374 	 acc:  0.39
944 : loss:  1.0118768 	 acc:  0.46
946 : loss:  1.0367771 	 acc:  0.42
948 : loss:  1.0673009 	 acc:  0.36
950 : loss:  1.0434054 	 acc:  0.42
952 : loss:  1.0421095 	 acc:  0.42
954 : loss:  1.0694484 	 acc:  0.35
956 : loss:  1.0820472 	 acc:  0.35
958 : loss:  1.0194494 	 acc:  0.51
960 : loss:  1.0891739 	 acc:  0.39
962 : loss:  1.0476825 	 acc:  0.45
964 : loss:  1.0295017 	 acc:  0.48
966 : loss:  1.0439878 	 acc:  0.44
968 : loss:  1.019045 	 acc:  0.43
970 : loss:  1.0395063 	 acc:  0.45
972 : loss:  1.0138309 	 acc:  0.52
974 : loss:  1.0058203 	 acc:  0.39
976 : loss:  1.0006833 	 acc:  0.5
978 : loss:  0.96042997 	 acc:  0.49
980 : loss:  1.0322336 	 acc:  0.39
982 : loss:  1.0144055 	 acc:  0.48
984 : loss:  1.0037019 	 acc:  0.49
986 : loss:  1.0288547 	 acc:  0.43
988 : loss:  1.0260994 	 acc:  0.42
990 : loss:  1.0052824 	 acc:  0.5
992 : loss:  1.0223433 	 acc:  0.44
994 : loss:  0.9960224 	 acc:  0.49
996 : loss:  1.0356228 	 acc:  0.41
998 : loss:  0.9780329 	 acc:  0.54
1000 : loss:  1.0229383 	 acc:  0.46
1002 : loss:  1.0674171 	 acc:  0.45
1004 : loss:  1.0581558 	 acc:  0.54
1006 : loss:  1.0261631 	 acc:  0.45
1008 : loss:  0.9843772 	 acc:  0.49
1010 : loss:  1.0372556 	 acc:  0.41
1012 : loss:  1.0252231 	 acc:  0.42
1014 : loss:  1.0118136 	 acc:  0.44
1016 : loss:  1.0392041 	 acc:  0.45
1018 : loss:  0.9823026 	 acc:  0.48
1020 : loss:  0.9986344 	 acc:  0.44
1022 : loss:  1.0543686 	 acc:  0.39
1024 : loss:  1.0127463 	 acc:  0.39
1026 : loss:  1.0487727 	 acc:  0.42
1028 : loss:  1.0244911 	 acc:  0.44
1030 : loss:  0.96149874 	 acc:  0.53
1032 : loss:  1.0172675 	 acc:  0.44
1034 : loss:  1.0707507 	 acc:  0.34
1036 : loss:  1.0204704 	 acc:  0.48
1038 : loss:  1.0532323 	 acc:  0.41
1040 : loss:  1.0566419 	 acc:  0.38
1042 : loss:  0.9678055 	 acc:  0.5
1044 : loss:  1.0328747 	 acc:  0.43
1046 : loss:  1.0615891 	 acc:  0.4
1048 : loss:  1.0247134 	 acc:  0.45
1050 : loss:  1.039201 	 acc:  0.51
1052 : loss:  1.0184503 	 acc:  0.51
1054 : loss:  1.0568243 	 acc:  0.47
1056 : loss:  1.0695901 	 acc:  0.39
1058 : loss:  1.0027715 	 acc:  0.44
1060 : loss:  1.1015173 	 acc:  0.41
1062 : loss:  1.0236125 	 acc:  0.49
1064 : loss:  1.0591699 	 acc:  0.4
1066 : loss:  1.0338862 	 acc:  0.5
1068 : loss:  1.0111923 	 acc:  0.45
1070 : loss:  1.0487477 	 acc:  0.43
1072 : loss:  1.0081166 	 acc:  0.47
1074 : loss:  1.0165845 	 acc:  0.54
1076 : loss:  0.96455646 	 acc:  0.49
1078 : loss:  1.0362989 	 acc:  0.44
1080 : loss:  1.0056835 	 acc:  0.47
1082 : loss:  1.0234892 	 acc:  0.46
1084 : loss:  0.98280936 	 acc:  0.54
1086 : loss:  0.98412114 	 acc:  0.49
1088 : loss:  1.0296155 	 acc:  0.46
1090 : loss:  1.0700608 	 acc:  0.34
1092 : loss:  0.9914772 	 acc:  0.51
1094 : loss:  1.0349526 	 acc:  0.39
1096 : loss:  1.0061898 	 acc:  0.5
1098 : loss:  1.0199074 	 acc:  0.44
1100 : loss:  1.0342082 	 acc:  0.48
1102 : loss:  1.0254894 	 acc:  0.45
1104 : loss:  1.0456326 	 acc:  0.43
1106 : loss:  0.99270815 	 acc:  0.46
1108 : loss:  1.0452781 	 acc:  0.49
1110 : loss:  1.0648369 	 acc:  0.44
1112 : loss:  0.95186406 	 acc:  0.58
1114 : loss:  1.0692847 	 acc:  0.39
acc:  0.44
acc:  0.49
acc:  0.53
acc:  0.42
acc:  0.52
acc:  0.51
acc:  0.49
acc:  0.41
acc:  0.44
acc:  0.52
acc:  0.45
acc:  0.46
acc:  0.49
acc:  0.41
acc:  0.5
acc:  0.45
acc:  0.46
acc:  0.38
acc:  0.43
acc:  0.49
acc:  0.55
acc:  0.5
acc:  0.44
acc:  0.4
acc:  0.43
acc:  0.49
acc:  0.4
acc:  0.44
acc:  0.49
acc:  0.45
acc:  0.48
acc:  0.43
acc:  0.47
acc:  0.53
acc:  0.48
acc:  0.36
acc:  0.44
acc:  0.52
acc:  0.45
acc:  0.59
acc:  0.43
acc:  0.51
acc:  0.48
acc:  0.59
acc:  0.47
acc:  0.45
acc:  0.51
acc:  0.47
acc:  0.48
acc:  0.46
acc:  0.49
acc:  0.54
acc:  0.57
acc:  0.5
acc:  0.45
acc:  0.49
acc:  0.45
acc:  0.35
acc:  0.48
acc:  0.52
acc:  0.41
acc:  0.46
acc:  0.5
acc:  0.54
acc:  0.47
acc:  0.48
acc:  0.39
acc:  0.51
acc:  0.56
acc:  0.47
acc:  0.37
acc:  0.47
acc:  0.54
acc:  0.5
acc:  0.46
acc:  0.5
acc:  0.46
acc:  0.48
acc:  0.48
acc:  0.36
acc:  0.5
acc:  0.47
acc:  0.43
acc:  0.41
acc:  0.38
acc:  0.51
acc:  0.46
acc:  0.35
acc:  0.42
acc:  0.42
acc:  0.52
acc:  0.44
acc:  0.49
acc:  0.51
acc:  0.48
acc:  0.46
acc:  0.33
acc:  0.43
acc:  0.4
acc:  0.52

100 	 [52.89335203 49.98706098 34.45624776]
2 	val accuracy:  0.4666 	 f_! score:  [0.52893352 0.49987061 0.34456248]

1116 : loss:  0.98671615 	 acc:  0.44
1118 : loss:  1.0190371 	 acc:  0.49
1120 : loss:  0.9886453 	 acc:  0.46
1122 : loss:  1.0284569 	 acc:  0.49
1124 : loss:  1.0480201 	 acc:  0.4
1126 : loss:  1.009596 	 acc:  0.44
1128 : loss:  1.0356307 	 acc:  0.43
1130 : loss:  0.9952145 	 acc:  0.53
1132 : loss:  1.0217263 	 acc:  0.49
1134 : loss:  1.0241556 	 acc:  0.48
1136 : loss:  1.0278188 	 acc:  0.42
1138 : loss:  1.0464207 	 acc:  0.46
1140 : loss:  1.0203587 	 acc:  0.46
1142 : loss:  1.0154047 	 acc:  0.45
1144 : loss:  0.99992824 	 acc:  0.51
1146 : loss:  1.0309062 	 acc:  0.45
1148 : loss:  0.9850418 	 acc:  0.55
1150 : loss:  0.9896088 	 acc:  0.48
1152 : loss:  1.0380319 	 acc:  0.42
1154 : loss:  1.0135124 	 acc:  0.45
1156 : loss:  0.98072976 	 acc:  0.55
1158 : loss:  0.9609336 	 acc:  0.58
1160 : loss:  1.0334527 	 acc:  0.5
1162 : loss:  1.0374204 	 acc:  0.49
1164 : loss:  1.0304321 	 acc:  0.39
1166 : loss:  1.0521432 	 acc:  0.41
1168 : loss:  0.9720328 	 acc:  0.54
1170 : loss:  1.0356013 	 acc:  0.46
1172 : loss:  1.0340503 	 acc:  0.41
1174 : loss:  1.0755211 	 acc:  0.4
1176 : loss:  1.0350331 	 acc:  0.5
1178 : loss:  1.0099399 	 acc:  0.47
1180 : loss:  0.9726328 	 acc:  0.5
1182 : loss:  0.97859734 	 acc:  0.5
1184 : loss:  1.0457091 	 acc:  0.42
1186 : loss:  1.0234355 	 acc:  0.42
1188 : loss:  0.98538476 	 acc:  0.5
1190 : loss:  1.0709407 	 acc:  0.46
1192 : loss:  1.0260046 	 acc:  0.41
1194 : loss:  1.0618424 	 acc:  0.41
1196 : loss:  1.0064577 	 acc:  0.52
1198 : loss:  1.0106357 	 acc:  0.49
1200 : loss:  1.0773807 	 acc:  0.39
1202 : loss:  1.0105851 	 acc:  0.42
1204 : loss:  0.99489015 	 acc:  0.42
1206 : loss:  1.0646126 	 acc:  0.43
1208 : loss:  1.0605735 	 acc:  0.38
1210 : loss:  1.0849209 	 acc:  0.39
1212 : loss:  1.099157 	 acc:  0.38
1214 : loss:  0.9985014 	 acc:  0.48
1216 : loss:  1.0481429 	 acc:  0.45
1218 : loss:  0.95544386 	 acc:  0.5
1220 : loss:  1.0126947 	 acc:  0.52
1222 : loss:  1.0081681 	 acc:  0.47
1224 : loss:  1.0300051 	 acc:  0.41
1226 : loss:  0.9752535 	 acc:  0.57
1228 : loss:  1.0244108 	 acc:  0.44
1230 : loss:  1.0311466 	 acc:  0.4
1232 : loss:  1.0548755 	 acc:  0.35
1234 : loss:  1.0452058 	 acc:  0.36
1236 : loss:  0.9778462 	 acc:  0.53
1238 : loss:  1.0245085 	 acc:  0.42
1240 : loss:  1.0432254 	 acc:  0.42
1242 : loss:  0.9769217 	 acc:  0.53
1244 : loss:  1.0423477 	 acc:  0.5
1246 : loss:  1.0262508 	 acc:  0.46
1248 : loss:  1.067366 	 acc:  0.37
1250 : loss:  1.0233188 	 acc:  0.41
1252 : loss:  1.0219033 	 acc:  0.49
1254 : loss:  1.0149537 	 acc:  0.47
1256 : loss:  1.0240396 	 acc:  0.43
1258 : loss:  0.9723195 	 acc:  0.51
1260 : loss:  0.9769677 	 acc:  0.44
1262 : loss:  1.0054028 	 acc:  0.49
1264 : loss:  0.9712559 	 acc:  0.55
1266 : loss:  1.03577 	 acc:  0.4
1268 : loss:  1.0490342 	 acc:  0.42
1270 : loss:  1.0506207 	 acc:  0.37
1272 : loss:  1.0366863 	 acc:  0.41
1274 : loss:  1.0550395 	 acc:  0.39
1276 : loss:  1.073202 	 acc:  0.43
1278 : loss:  1.0339329 	 acc:  0.48
1280 : loss:  0.9766838 	 acc:  0.6
1282 : loss:  1.0518897 	 acc:  0.36
1284 : loss:  1.0606602 	 acc:  0.41
1286 : loss:  1.0269647 	 acc:  0.49
1288 : loss:  1.0171524 	 acc:  0.47
1290 : loss:  0.989182 	 acc:  0.48
1292 : loss:  1.0109941 	 acc:  0.46
1294 : loss:  1.0580086 	 acc:  0.44
1296 : loss:  1.0583489 	 acc:  0.37
1298 : loss:  1.0202363 	 acc:  0.47
1300 : loss:  1.0421956 	 acc:  0.42
1302 : loss:  0.96008337 	 acc:  0.53
1304 : loss:  1.0809418 	 acc:  0.43
1306 : loss:  1.0535396 	 acc:  0.43
1308 : loss:  0.96988004 	 acc:  0.54
1310 : loss:  1.0425209 	 acc:  0.37
1312 : loss:  0.9337052 	 acc:  0.56
1314 : loss:  0.98599863 	 acc:  0.52
1316 : loss:  1.0934002 	 acc:  0.34
1318 : loss:  1.0833352 	 acc:  0.38
1320 : loss:  1.1255372 	 acc:  0.47
1322 : loss:  0.977763 	 acc:  0.47
1324 : loss:  1.0650619 	 acc:  0.41
1326 : loss:  1.0998617 	 acc:  0.42
1328 : loss:  1.0210589 	 acc:  0.45
1330 : loss:  1.0047225 	 acc:  0.48
1332 : loss:  1.070091 	 acc:  0.48
1334 : loss:  1.0499094 	 acc:  0.38
1336 : loss:  1.0342276 	 acc:  0.46
1338 : loss:  1.0487214 	 acc:  0.44
1340 : loss:  1.0334946 	 acc:  0.43
1342 : loss:  1.0685438 	 acc:  0.38
1344 : loss:  0.997739 	 acc:  0.53
1346 : loss:  0.9899663 	 acc:  0.54
1348 : loss:  1.0480168 	 acc:  0.48
1350 : loss:  0.98970723 	 acc:  0.46
1352 : loss:  1.0204308 	 acc:  0.47
1354 : loss:  1.0013326 	 acc:  0.53
1356 : loss:  1.0034003 	 acc:  0.49
1358 : loss:  1.0895234 	 acc:  0.42
1360 : loss:  1.0421535 	 acc:  0.42
1362 : loss:  1.0417752 	 acc:  0.45
1364 : loss:  1.0094824 	 acc:  0.46
1366 : loss:  1.0195736 	 acc:  0.43
1368 : loss:  1.0287664 	 acc:  0.48
1370 : loss:  1.0656704 	 acc:  0.39
1372 : loss:  1.0284959 	 acc:  0.47
1374 : loss:  0.9791992 	 acc:  0.5
1376 : loss:  1.0435858 	 acc:  0.44
1378 : loss:  1.033871 	 acc:  0.44
1380 : loss:  1.0191532 	 acc:  0.46
1382 : loss:  1.0854145 	 acc:  0.37
1384 : loss:  1.0036521 	 acc:  0.44
1386 : loss:  1.0002757 	 acc:  0.5
1388 : loss:  1.0101386 	 acc:  0.43
1390 : loss:  1.0211215 	 acc:  0.43
1392 : loss:  1.069712 	 acc:  0.44
1394 : loss:  0.9998074 	 acc:  0.47
1396 : loss:  1.0020758 	 acc:  0.41
1398 : loss:  1.0280395 	 acc:  0.45
1400 : loss:  1.0410757 	 acc:  0.48
1402 : loss:  0.9867837 	 acc:  0.48
1404 : loss:  1.0340077 	 acc:  0.41
1406 : loss:  0.9477487 	 acc:  0.55
1408 : loss:  1.0287883 	 acc:  0.4
1410 : loss:  1.0466394 	 acc:  0.47
1412 : loss:  1.0142928 	 acc:  0.45
1414 : loss:  1.033487 	 acc:  0.4
1416 : loss:  1.0583285 	 acc:  0.45
1418 : loss:  0.9622253 	 acc:  0.52
1420 : loss:  1.0809143 	 acc:  0.38
1422 : loss:  0.9880643 	 acc:  0.49
1424 : loss:  1.020657 	 acc:  0.47
1426 : loss:  1.0233266 	 acc:  0.4
1428 : loss:  1.0040973 	 acc:  0.46
1430 : loss:  1.0326796 	 acc:  0.44
1432 : loss:  1.0767921 	 acc:  0.39
1434 : loss:  1.0614356 	 acc:  0.46
1436 : loss:  1.0300387 	 acc:  0.49
1438 : loss:  1.0355487 	 acc:  0.48
1440 : loss:  0.99313354 	 acc:  0.5
1442 : loss:  1.0437857 	 acc:  0.42
1444 : loss:  1.0714443 	 acc:  0.4
1446 : loss:  0.9341998 	 acc:  0.49
1448 : loss:  1.0512362 	 acc:  0.42
1450 : loss:  1.0476537 	 acc:  0.47
1452 : loss:  1.0371039 	 acc:  0.43
1454 : loss:  0.99146765 	 acc:  0.44
1456 : loss:  1.0104526 	 acc:  0.47
1458 : loss:  1.0728778 	 acc:  0.42
1460 : loss:  1.0635465 	 acc:  0.36
1462 : loss:  1.0116265 	 acc:  0.52
1464 : loss:  1.0330788 	 acc:  0.41
1466 : loss:  0.9865865 	 acc:  0.56
1468 : loss:  1.0007492 	 acc:  0.49
1470 : loss:  0.9722314 	 acc:  0.45
1472 : loss:  1.0494374 	 acc:  0.49
1474 : loss:  0.98724717 	 acc:  0.52
1476 : loss:  1.0706335 	 acc:  0.45
1478 : loss:  1.022346 	 acc:  0.51
1480 : loss:  1.0073478 	 acc:  0.56
1482 : loss:  1.0004543 	 acc:  0.54
1484 : loss:  1.0917437 	 acc:  0.4
1486 : loss:  0.95262927 	 acc:  0.52
acc:  0.54
acc:  0.4
acc:  0.43
acc:  0.47
acc:  0.46
acc:  0.46
acc:  0.38
acc:  0.44
acc:  0.44
acc:  0.45
acc:  0.42
acc:  0.47
acc:  0.52
acc:  0.41
acc:  0.55
acc:  0.35
acc:  0.45
acc:  0.49
acc:  0.51
acc:  0.49
acc:  0.43
acc:  0.39
acc:  0.52
acc:  0.47
acc:  0.46
acc:  0.51
acc:  0.52
acc:  0.48
acc:  0.38
acc:  0.5
acc:  0.4
acc:  0.43
acc:  0.56
acc:  0.48
acc:  0.45
acc:  0.51
acc:  0.49
acc:  0.42
acc:  0.48
acc:  0.45
acc:  0.42
acc:  0.47
acc:  0.53
acc:  0.51
acc:  0.54
acc:  0.36
acc:  0.47
acc:  0.41
acc:  0.39
acc:  0.5
acc:  0.42
acc:  0.52
acc:  0.49
acc:  0.48
acc:  0.48
acc:  0.53
acc:  0.44
acc:  0.43
acc:  0.5
acc:  0.47
acc:  0.38
acc:  0.41
acc:  0.46
acc:  0.48
acc:  0.4
acc:  0.52
acc:  0.55
acc:  0.48
acc:  0.52
acc:  0.43
acc:  0.5
acc:  0.43
acc:  0.5
acc:  0.4
acc:  0.47
acc:  0.41
acc:  0.45
acc:  0.42
acc:  0.44
acc:  0.47
acc:  0.48
acc:  0.43
acc:  0.49
acc:  0.52
acc:  0.51
acc:  0.45
acc:  0.48
acc:  0.47
acc:  0.43
acc:  0.37
acc:  0.35
acc:  0.47
acc:  0.53
acc:  0.41
acc:  0.5
acc:  0.4
acc:  0.46
acc:  0.39
acc:  0.42
acc:  0.44

100 	 [55.61564714 53.45205816  3.57222275]
3 	val accuracy:  0.45940003 	 f_! score:  [0.55615647 0.53452058 0.03572223]

100 	 [55.61564714 53.45205816  3.57222275]
100 	 [52.28050068 41.72346493 37.25      ]
100 	 [60.19558231 75.55785864  1.90102576]
100 	 [3377. 3307. 3316.]
---train_last_layer Test  Twitter ---
0.45940003
f1:  [0.55615647 0.53452058 0.03572223]
--- 99.11865758895874 seconds ---


  Twitter
65730
2018-10-14 18:51:44.262899: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-10-14 18:51:44.262949: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-14 18:51:44.262956: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-10-14 18:51:44.262962: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-10-14 18:51:44.263052: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2905 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
models/CNN/pretrained_model.ckpt-17931
---init ready   CNN ---
2018-10-14 18:51:47.404898: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-10-14 18:51:47.404935: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-14 18:51:47.404940: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-10-14 18:51:47.404944: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-10-14 18:51:47.405036: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2905 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/CNN/pretrained_model.ckpt-0
0 : loss:  1.7413532 	 acc:  0.35
2 : loss:  1.1226609 	 acc:  0.29
4 : loss:  1.0976754 	 acc:  0.34
6 : loss:  1.0914458 	 acc:  0.42
8 : loss:  1.103421 	 acc:  0.36
10 : loss:  1.1352774 	 acc:  0.29
12 : loss:  1.0813665 	 acc:  0.39
14 : loss:  1.0893995 	 acc:  0.36
16 : loss:  1.1124207 	 acc:  0.34
18 : loss:  1.074339 	 acc:  0.45
20 : loss:  1.0900265 	 acc:  0.41
22 : loss:  1.0528808 	 acc:  0.45
24 : loss:  1.0529431 	 acc:  0.49
26 : loss:  1.0441574 	 acc:  0.48
28 : loss:  1.0423329 	 acc:  0.47
30 : loss:  1.07736 	 acc:  0.38
32 : loss:  1.049407 	 acc:  0.46
34 : loss:  0.99649984 	 acc:  0.52
36 : loss:  0.97669977 	 acc:  0.56
38 : loss:  1.1014695 	 acc:  0.35
40 : loss:  1.0437775 	 acc:  0.48
42 : loss:  1.0682788 	 acc:  0.46
44 : loss:  1.068183 	 acc:  0.39
46 : loss:  1.0362613 	 acc:  0.47
48 : loss:  1.0633997 	 acc:  0.42
50 : loss:  1.0367585 	 acc:  0.44
52 : loss:  1.0221962 	 acc:  0.46
54 : loss:  1.007649 	 acc:  0.48
56 : loss:  0.98583925 	 acc:  0.52
58 : loss:  1.0013524 	 acc:  0.45
60 : loss:  0.9712731 	 acc:  0.51
62 : loss:  1.0096331 	 acc:  0.45
64 : loss:  1.0026503 	 acc:  0.47
66 : loss:  1.0703894 	 acc:  0.43
68 : loss:  0.99543643 	 acc:  0.48
70 : loss:  0.9905523 	 acc:  0.51
72 : loss:  0.9707931 	 acc:  0.48
74 : loss:  0.96115255 	 acc:  0.55
76 : loss:  0.9798668 	 acc:  0.48
78 : loss:  0.92860115 	 acc:  0.52
80 : loss:  0.96088046 	 acc:  0.48
82 : loss:  0.9487936 	 acc:  0.49
84 : loss:  0.93203187 	 acc:  0.48
86 : loss:  1.0852745 	 acc:  0.43333334
88 : loss:  0.9621149 	 acc:  0.46
90 : loss:  0.95918393 	 acc:  0.52
92 : loss:  1.0089545 	 acc:  0.46
94 : loss:  0.97184813 	 acc:  0.53
96 : loss:  0.99301547 	 acc:  0.44
98 : loss:  1.0091642 	 acc:  0.46
100 : loss:  0.9371739 	 acc:  0.55
102 : loss:  0.9890243 	 acc:  0.51
104 : loss:  1.0442405 	 acc:  0.45
106 : loss:  1.0307608 	 acc:  0.47
108 : loss:  0.9313371 	 acc:  0.57
110 : loss:  1.0230706 	 acc:  0.52
112 : loss:  0.93991005 	 acc:  0.54
114 : loss:  0.94022554 	 acc:  0.5
116 : loss:  0.9459496 	 acc:  0.47
118 : loss:  0.9837507 	 acc:  0.44
120 : loss:  0.9698796 	 acc:  0.46
122 : loss:  1.0167301 	 acc:  0.42
124 : loss:  0.9051255 	 acc:  0.48
126 : loss:  0.9052969 	 acc:  0.48
128 : loss:  0.9564939 	 acc:  0.56
130 : loss:  1.0035707 	 acc:  0.45
132 : loss:  0.97967666 	 acc:  0.52
134 : loss:  0.9668225 	 acc:  0.51
136 : loss:  0.93166983 	 acc:  0.58
138 : loss:  0.9175444 	 acc:  0.56
140 : loss:  0.990205 	 acc:  0.48
142 : loss:  0.9501876 	 acc:  0.47
144 : loss:  0.95532095 	 acc:  0.57
146 : loss:  0.99439514 	 acc:  0.45
148 : loss:  0.9460395 	 acc:  0.45
150 : loss:  1.0326465 	 acc:  0.43
152 : loss:  1.0378896 	 acc:  0.43
154 : loss:  0.9616901 	 acc:  0.46
156 : loss:  0.9353166 	 acc:  0.52
158 : loss:  0.9679934 	 acc:  0.45
160 : loss:  0.90087926 	 acc:  0.56
162 : loss:  0.8891297 	 acc:  0.53
164 : loss:  0.9562708 	 acc:  0.43
166 : loss:  0.8900803 	 acc:  0.48
168 : loss:  1.0503441 	 acc:  0.51
170 : loss:  1.0094036 	 acc:  0.5
172 : loss:  0.8613524 	 acc:  0.45
174 : loss:  0.86594045 	 acc:  0.53
176 : loss:  0.92588216 	 acc:  0.55
178 : loss:  0.923387 	 acc:  0.53
180 : loss:  0.9150885 	 acc:  0.61
182 : loss:  0.95208794 	 acc:  0.51
184 : loss:  0.96499145 	 acc:  0.46
186 : loss:  0.99159396 	 acc:  0.47
188 : loss:  0.92453074 	 acc:  0.53
190 : loss:  0.9877746 	 acc:  0.42
192 : loss:  0.97800094 	 acc:  0.46
194 : loss:  0.9718247 	 acc:  0.48
196 : loss:  0.95595634 	 acc:  0.61
198 : loss:  1.0200692 	 acc:  0.46
200 : loss:  1.0108451 	 acc:  0.45
202 : loss:  0.95221746 	 acc:  0.52
204 : loss:  0.87631285 	 acc:  0.57
206 : loss:  0.8354624 	 acc:  0.57
208 : loss:  0.9505379 	 acc:  0.54
210 : loss:  1.0046774 	 acc:  0.42
212 : loss:  0.8652716 	 acc:  0.57
214 : loss:  0.8837511 	 acc:  0.55
216 : loss:  0.97589755 	 acc:  0.44
218 : loss:  0.9335015 	 acc:  0.5
220 : loss:  0.9189179 	 acc:  0.53
222 : loss:  0.932693 	 acc:  0.58
224 : loss:  0.9115138 	 acc:  0.52
226 : loss:  0.9847734 	 acc:  0.49
228 : loss:  0.906884 	 acc:  0.48
230 : loss:  1.0640658 	 acc:  0.47
232 : loss:  0.8989715 	 acc:  0.55
234 : loss:  0.9640579 	 acc:  0.5
236 : loss:  0.83423513 	 acc:  0.64
238 : loss:  0.9192096 	 acc:  0.6
240 : loss:  1.031845 	 acc:  0.5
242 : loss:  0.96534944 	 acc:  0.54
244 : loss:  0.90952545 	 acc:  0.57
246 : loss:  0.9112737 	 acc:  0.58
248 : loss:  1.000859 	 acc:  0.49
250 : loss:  0.8365713 	 acc:  0.59
252 : loss:  0.95581156 	 acc:  0.45
254 : loss:  0.8994127 	 acc:  0.53
256 : loss:  0.855609 	 acc:  0.6
258 : loss:  1.0543538 	 acc:  0.36
260 : loss:  1.0329133 	 acc:  0.42
262 : loss:  0.932575 	 acc:  0.54
264 : loss:  0.9231619 	 acc:  0.52
266 : loss:  0.92938596 	 acc:  0.58
268 : loss:  0.93898827 	 acc:  0.5
270 : loss:  0.9850925 	 acc:  0.53
272 : loss:  0.8578265 	 acc:  0.64
274 : loss:  0.9258783 	 acc:  0.56
276 : loss:  0.9189339 	 acc:  0.5
278 : loss:  0.8397557 	 acc:  0.62
280 : loss:  0.8770139 	 acc:  0.57
282 : loss:  0.842119 	 acc:  0.59
284 : loss:  0.9388018 	 acc:  0.56
286 : loss:  0.98890585 	 acc:  0.4
288 : loss:  0.80183136 	 acc:  0.61
290 : loss:  0.9020423 	 acc:  0.61
292 : loss:  0.915932 	 acc:  0.55
294 : loss:  0.87539285 	 acc:  0.6
296 : loss:  0.9057728 	 acc:  0.56
298 : loss:  0.9845552 	 acc:  0.54
300 : loss:  0.9206364 	 acc:  0.48
302 : loss:  0.83265406 	 acc:  0.62
304 : loss:  0.8824983 	 acc:  0.59
306 : loss:  0.94542444 	 acc:  0.57
308 : loss:  0.9370276 	 acc:  0.57
310 : loss:  0.94154096 	 acc:  0.57
312 : loss:  0.9015868 	 acc:  0.58
314 : loss:  0.7801214 	 acc:  0.71
316 : loss:  0.8869733 	 acc:  0.55
318 : loss:  0.9110317 	 acc:  0.62
320 : loss:  0.9295223 	 acc:  0.48
322 : loss:  0.965952 	 acc:  0.47
324 : loss:  0.9324086 	 acc:  0.56
326 : loss:  0.8578075 	 acc:  0.62
328 : loss:  0.89892936 	 acc:  0.58
330 : loss:  0.9070465 	 acc:  0.53
332 : loss:  0.8590349 	 acc:  0.56
334 : loss:  0.8989985 	 acc:  0.59
336 : loss:  0.8965338 	 acc:  0.55
338 : loss:  0.8262324 	 acc:  0.64
340 : loss:  0.91899157 	 acc:  0.53
342 : loss:  0.85655427 	 acc:  0.5
344 : loss:  0.9150386 	 acc:  0.58
346 : loss:  0.89082986 	 acc:  0.61
348 : loss:  0.9076439 	 acc:  0.56
350 : loss:  0.8734081 	 acc:  0.56
352 : loss:  0.9488977 	 acc:  0.47
354 : loss:  0.92102814 	 acc:  0.59
356 : loss:  0.86280596 	 acc:  0.56
358 : loss:  0.7568728 	 acc:  0.65
360 : loss:  0.9232403 	 acc:  0.63
362 : loss:  0.88258433 	 acc:  0.54
364 : loss:  0.82787234 	 acc:  0.62
366 : loss:  0.8172559 	 acc:  0.59
368 : loss:  0.91796005 	 acc:  0.49
370 : loss:  0.93000823 	 acc:  0.52

Saving...
saved to models/CNN/pretrained_model.ckpt-372

100 	 [63.49299675 57.61802899 52.11274449]
0 	val accuracy:  0.581 	 f_! score:  [0.63492997 0.57618029 0.52112744]

372 : loss:  0.9083816 	 acc:  0.61
374 : loss:  0.8428674 	 acc:  0.65
376 : loss:  0.89129657 	 acc:  0.56
378 : loss:  0.813416 	 acc:  0.6
380 : loss:  0.7935489 	 acc:  0.65
382 : loss:  0.9437676 	 acc:  0.52
384 : loss:  0.7981216 	 acc:  0.65
386 : loss:  0.92028296 	 acc:  0.53
388 : loss:  0.8556532 	 acc:  0.63
390 : loss:  0.91920006 	 acc:  0.57
392 : loss:  0.9058848 	 acc:  0.51
394 : loss:  0.9445797 	 acc:  0.5
396 : loss:  0.8188865 	 acc:  0.59
398 : loss:  0.94110346 	 acc:  0.58
400 : loss:  0.83881545 	 acc:  0.59
402 : loss:  0.86694336 	 acc:  0.57
404 : loss:  0.9052491 	 acc:  0.52
406 : loss:  0.8617363 	 acc:  0.57
408 : loss:  0.955109 	 acc:  0.47
410 : loss:  0.915314 	 acc:  0.55
412 : loss:  0.90011376 	 acc:  0.57
414 : loss:  0.84715515 	 acc:  0.59
416 : loss:  0.95691544 	 acc:  0.5
418 : loss:  0.89671105 	 acc:  0.52
420 : loss:  0.851481 	 acc:  0.57
422 : loss:  0.76518553 	 acc:  0.65
424 : loss:  0.8371799 	 acc:  0.63
426 : loss:  0.8060098 	 acc:  0.59
428 : loss:  0.8654917 	 acc:  0.6
430 : loss:  0.86668587 	 acc:  0.62
432 : loss:  0.93582773 	 acc:  0.56
434 : loss:  0.80858946 	 acc:  0.63
436 : loss:  0.857568 	 acc:  0.64
438 : loss:  0.8644056 	 acc:  0.56
440 : loss:  0.88323456 	 acc:  0.58
442 : loss:  0.8930983 	 acc:  0.58
444 : loss:  0.86932737 	 acc:  0.53
446 : loss:  0.873448 	 acc:  0.55
448 : loss:  0.82224923 	 acc:  0.58
450 : loss:  0.82715917 	 acc:  0.6
452 : loss:  0.80444586 	 acc:  0.59
454 : loss:  0.7661517 	 acc:  0.68
456 : loss:  0.8001196 	 acc:  0.63
458 : loss:  0.8339325 	 acc:  0.65
460 : loss:  0.93026805 	 acc:  0.49
462 : loss:  0.81650186 	 acc:  0.59
464 : loss:  1.0103774 	 acc:  0.49
466 : loss:  0.8004793 	 acc:  0.61
468 : loss:  0.9637635 	 acc:  0.53
470 : loss:  0.8227124 	 acc:  0.62
472 : loss:  0.8978696 	 acc:  0.53
474 : loss:  0.79353034 	 acc:  0.67
476 : loss:  0.8595999 	 acc:  0.6
478 : loss:  0.8382929 	 acc:  0.58
480 : loss:  0.8696445 	 acc:  0.55
482 : loss:  0.8316007 	 acc:  0.62
484 : loss:  0.9376085 	 acc:  0.56
486 : loss:  0.95119303 	 acc:  0.53
488 : loss:  0.92718726 	 acc:  0.45
490 : loss:  0.82286835 	 acc:  0.62
492 : loss:  0.8645594 	 acc:  0.57
494 : loss:  0.90697867 	 acc:  0.54
496 : loss:  0.82831115 	 acc:  0.59
498 : loss:  0.9196849 	 acc:  0.53
500 : loss:  0.9586754 	 acc:  0.51
502 : loss:  0.8415764 	 acc:  0.66
504 : loss:  0.89556676 	 acc:  0.6
506 : loss:  0.98926306 	 acc:  0.53
508 : loss:  0.95952255 	 acc:  0.52
510 : loss:  0.94363314 	 acc:  0.55
512 : loss:  0.871195 	 acc:  0.6
514 : loss:  0.87714034 	 acc:  0.59
516 : loss:  0.8697548 	 acc:  0.54
518 : loss:  0.8375092 	 acc:  0.61
520 : loss:  0.8337477 	 acc:  0.6
522 : loss:  0.8108961 	 acc:  0.63
524 : loss:  0.88526726 	 acc:  0.53
526 : loss:  0.92275184 	 acc:  0.58
528 : loss:  0.9237034 	 acc:  0.56
530 : loss:  0.86806047 	 acc:  0.54
532 : loss:  0.83572763 	 acc:  0.59
534 : loss:  0.88693345 	 acc:  0.55
536 : loss:  0.9076876 	 acc:  0.51
538 : loss:  0.8969825 	 acc:  0.61
540 : loss:  0.88432884 	 acc:  0.56
542 : loss:  0.8498023 	 acc:  0.6
544 : loss:  0.8738337 	 acc:  0.58
546 : loss:  0.78441954 	 acc:  0.61
548 : loss:  0.849356 	 acc:  0.6
550 : loss:  0.84854597 	 acc:  0.55
552 : loss:  0.85641026 	 acc:  0.63
554 : loss:  0.90720797 	 acc:  0.54
556 : loss:  0.8504002 	 acc:  0.61
558 : loss:  0.7766053 	 acc:  0.66
560 : loss:  0.8198089 	 acc:  0.59
562 : loss:  0.77525854 	 acc:  0.65
564 : loss:  0.8129607 	 acc:  0.69
566 : loss:  0.805251 	 acc:  0.67
568 : loss:  0.8765699 	 acc:  0.56
570 : loss:  0.8533377 	 acc:  0.61
572 : loss:  0.8090352 	 acc:  0.58
574 : loss:  0.892611 	 acc:  0.56
576 : loss:  0.8655241 	 acc:  0.6
578 : loss:  0.9072219 	 acc:  0.61
580 : loss:  0.8656053 	 acc:  0.54
582 : loss:  0.83484155 	 acc:  0.58
584 : loss:  0.82543236 	 acc:  0.6
586 : loss:  0.8801025 	 acc:  0.62
588 : loss:  0.88934386 	 acc:  0.56
590 : loss:  0.85493124 	 acc:  0.62
592 : loss:  0.82071084 	 acc:  0.63
594 : loss:  0.87149996 	 acc:  0.64
596 : loss:  0.8124942 	 acc:  0.64
598 : loss:  0.9412927 	 acc:  0.6
600 : loss:  0.7233925 	 acc:  0.69
602 : loss:  0.8637487 	 acc:  0.63
604 : loss:  1.0070287 	 acc:  0.5
606 : loss:  0.852623 	 acc:  0.59
608 : loss:  0.7779764 	 acc:  0.64
610 : loss:  0.8404205 	 acc:  0.6
612 : loss:  0.91074485 	 acc:  0.59
614 : loss:  0.8631256 	 acc:  0.61
616 : loss:  0.9173453 	 acc:  0.56
618 : loss:  0.8766457 	 acc:  0.61
620 : loss:  0.86959475 	 acc:  0.6
622 : loss:  0.8681174 	 acc:  0.6
624 : loss:  0.8473304 	 acc:  0.6
626 : loss:  0.8130493 	 acc:  0.62
628 : loss:  0.9251336 	 acc:  0.53
630 : loss:  0.9102935 	 acc:  0.58
632 : loss:  0.82306886 	 acc:  0.59
634 : loss:  0.8655723 	 acc:  0.55
636 : loss:  0.76786405 	 acc:  0.65
638 : loss:  0.8952349 	 acc:  0.55
640 : loss:  0.9655015 	 acc:  0.5
642 : loss:  0.75683165 	 acc:  0.62
644 : loss:  0.91714096 	 acc:  0.52
646 : loss:  0.8282323 	 acc:  0.59
648 : loss:  0.84560984 	 acc:  0.62
650 : loss:  0.8690746 	 acc:  0.6
652 : loss:  0.85948527 	 acc:  0.56
654 : loss:  0.83008766 	 acc:  0.56
656 : loss:  0.78723556 	 acc:  0.65
658 : loss:  0.91057783 	 acc:  0.55
660 : loss:  0.939419 	 acc:  0.58
662 : loss:  0.87025905 	 acc:  0.58
664 : loss:  0.9042434 	 acc:  0.57
666 : loss:  0.854879 	 acc:  0.59
668 : loss:  0.8244322 	 acc:  0.63
670 : loss:  0.80881506 	 acc:  0.62
672 : loss:  0.8401814 	 acc:  0.66
674 : loss:  0.8803664 	 acc:  0.51
676 : loss:  0.9219867 	 acc:  0.55
678 : loss:  0.8506083 	 acc:  0.62
680 : loss:  0.92183805 	 acc:  0.57
682 : loss:  0.9049309 	 acc:  0.56
684 : loss:  0.8607199 	 acc:  0.55
686 : loss:  0.8333819 	 acc:  0.6
688 : loss:  0.78552276 	 acc:  0.66
690 : loss:  0.87939066 	 acc:  0.56
692 : loss:  0.8644899 	 acc:  0.62
694 : loss:  0.90612626 	 acc:  0.59
696 : loss:  0.77113366 	 acc:  0.63
698 : loss:  0.82744265 	 acc:  0.62
700 : loss:  0.9310388 	 acc:  0.54
702 : loss:  0.8428154 	 acc:  0.64
704 : loss:  0.9323179 	 acc:  0.59
706 : loss:  0.8066854 	 acc:  0.65
708 : loss:  0.8830728 	 acc:  0.53
710 : loss:  0.8610587 	 acc:  0.59
712 : loss:  0.8063978 	 acc:  0.67
714 : loss:  0.7318911 	 acc:  0.72
716 : loss:  0.8411426 	 acc:  0.63
718 : loss:  0.8213267 	 acc:  0.62
720 : loss:  0.75463504 	 acc:  0.65
722 : loss:  0.9250981 	 acc:  0.58
724 : loss:  0.85036093 	 acc:  0.57
726 : loss:  0.8309032 	 acc:  0.62
728 : loss:  0.8356516 	 acc:  0.6
730 : loss:  0.7567939 	 acc:  0.66
732 : loss:  1.0762141 	 acc:  0.46666667
734 : loss:  0.8678816 	 acc:  0.68
736 : loss:  0.8110711 	 acc:  0.58
738 : loss:  0.86700803 	 acc:  0.6
740 : loss:  0.93643814 	 acc:  0.52
742 : loss:  0.8430489 	 acc:  0.69

Saving...
saved to models/CNN/pretrained_model.ckpt-744

100 	 [64.83295554 64.02843813 45.29547891]
1 	val accuracy:  0.59940004 	 f_! score:  [0.64832956 0.64028438 0.45295479]

744 : loss:  0.86794114 	 acc:  0.56
746 : loss:  0.7632499 	 acc:  0.64
748 : loss:  0.8628824 	 acc:  0.57
750 : loss:  0.7589195 	 acc:  0.69
752 : loss:  0.8651992 	 acc:  0.64
754 : loss:  0.8588536 	 acc:  0.63
756 : loss:  0.7577015 	 acc:  0.67
758 : loss:  0.8442514 	 acc:  0.59
760 : loss:  0.7539394 	 acc:  0.65
762 : loss:  0.7938393 	 acc:  0.67
764 : loss:  0.89620215 	 acc:  0.6
766 : loss:  0.85429746 	 acc:  0.56
768 : loss:  0.91558397 	 acc:  0.58
770 : loss:  0.79402214 	 acc:  0.69
772 : loss:  0.8269278 	 acc:  0.65
774 : loss:  0.80867594 	 acc:  0.63
776 : loss:  0.71324646 	 acc:  0.67
778 : loss:  0.856218 	 acc:  0.62
780 : loss:  0.7585857 	 acc:  0.66
782 : loss:  0.8767583 	 acc:  0.63
784 : loss:  0.8088848 	 acc:  0.65
786 : loss:  0.76417005 	 acc:  0.63
788 : loss:  0.7705652 	 acc:  0.69
790 : loss:  0.90916234 	 acc:  0.58
792 : loss:  0.80179846 	 acc:  0.68
794 : loss:  0.80455863 	 acc:  0.61
796 : loss:  0.8855616 	 acc:  0.64
798 : loss:  0.70690507 	 acc:  0.67
800 : loss:  0.8290784 	 acc:  0.63
802 : loss:  0.84478843 	 acc:  0.59
804 : loss:  0.8442832 	 acc:  0.56
806 : loss:  0.72940576 	 acc:  0.67
808 : loss:  0.81049156 	 acc:  0.58
810 : loss:  0.7547564 	 acc:  0.66
812 : loss:  0.7486017 	 acc:  0.61
814 : loss:  0.66529936 	 acc:  0.72
816 : loss:  0.7801513 	 acc:  0.63
818 : loss:  0.7397782 	 acc:  0.66
820 : loss:  0.81045705 	 acc:  0.62
822 : loss:  0.8087498 	 acc:  0.61
824 : loss:  0.7330783 	 acc:  0.64
826 : loss:  0.8482527 	 acc:  0.6
828 : loss:  0.8362887 	 acc:  0.61
830 : loss:  0.83396393 	 acc:  0.64
832 : loss:  0.8302102 	 acc:  0.64
834 : loss:  0.78428847 	 acc:  0.6
836 : loss:  0.7936055 	 acc:  0.62
838 : loss:  0.9021489 	 acc:  0.6
840 : loss:  0.7810044 	 acc:  0.69
842 : loss:  0.84232223 	 acc:  0.56
844 : loss:  0.66897035 	 acc:  0.75
846 : loss:  0.73443246 	 acc:  0.64
848 : loss:  0.8912732 	 acc:  0.55
850 : loss:  0.8257382 	 acc:  0.59
852 : loss:  0.8276815 	 acc:  0.6
854 : loss:  0.7308157 	 acc:  0.71
856 : loss:  0.70253503 	 acc:  0.68
858 : loss:  0.819342 	 acc:  0.59
860 : loss:  0.76280546 	 acc:  0.68
862 : loss:  0.796585 	 acc:  0.58
864 : loss:  0.70347303 	 acc:  0.69
866 : loss:  0.8905985 	 acc:  0.51
868 : loss:  0.77557635 	 acc:  0.69
870 : loss:  1.0441477 	 acc:  0.48
872 : loss:  0.75042206 	 acc:  0.68
874 : loss:  0.83143926 	 acc:  0.66
876 : loss:  0.9264809 	 acc:  0.58
878 : loss:  0.9333554 	 acc:  0.59
880 : loss:  0.7492757 	 acc:  0.63
882 : loss:  0.71613675 	 acc:  0.72
884 : loss:  0.8780681 	 acc:  0.54
886 : loss:  0.7977464 	 acc:  0.66
888 : loss:  0.8445743 	 acc:  0.66
890 : loss:  0.8408449 	 acc:  0.65
892 : loss:  0.8575372 	 acc:  0.59
894 : loss:  0.819386 	 acc:  0.67
896 : loss:  0.7602957 	 acc:  0.61
898 : loss:  0.80530715 	 acc:  0.59
900 : loss:  0.7741392 	 acc:  0.65
902 : loss:  0.84046453 	 acc:  0.61
904 : loss:  0.8363576 	 acc:  0.57
906 : loss:  0.7817814 	 acc:  0.67
908 : loss:  0.8461976 	 acc:  0.59
910 : loss:  0.81991583 	 acc:  0.59
912 : loss:  0.8300679 	 acc:  0.57
914 : loss:  0.8343107 	 acc:  0.61
916 : loss:  0.8693289 	 acc:  0.58
918 : loss:  0.78345656 	 acc:  0.65
920 : loss:  0.72940445 	 acc:  0.72
922 : loss:  0.85755974 	 acc:  0.59
924 : loss:  0.74751806 	 acc:  0.59
926 : loss:  0.80210745 	 acc:  0.63
928 : loss:  0.8603718 	 acc:  0.56
930 : loss:  0.8873608 	 acc:  0.52
932 : loss:  0.7914142 	 acc:  0.66
934 : loss:  0.7958042 	 acc:  0.64
936 : loss:  0.81432295 	 acc:  0.65
938 : loss:  0.8124114 	 acc:  0.63
940 : loss:  0.77900785 	 acc:  0.65
942 : loss:  0.7856076 	 acc:  0.67
944 : loss:  0.82628703 	 acc:  0.63
946 : loss:  0.88836175 	 acc:  0.53
948 : loss:  0.80086493 	 acc:  0.63
950 : loss:  0.8661014 	 acc:  0.61
952 : loss:  0.9104156 	 acc:  0.59
954 : loss:  0.81532615 	 acc:  0.68
956 : loss:  0.7947341 	 acc:  0.67
958 : loss:  0.8701423 	 acc:  0.53
960 : loss:  0.7919573 	 acc:  0.66
962 : loss:  0.92604357 	 acc:  0.6
964 : loss:  0.8630291 	 acc:  0.59
966 : loss:  0.71486557 	 acc:  0.69
968 : loss:  0.74231666 	 acc:  0.67
970 : loss:  0.8032891 	 acc:  0.6
972 : loss:  0.7866642 	 acc:  0.65
974 : loss:  0.7271048 	 acc:  0.67
976 : loss:  0.8297665 	 acc:  0.59
978 : loss:  0.75960624 	 acc:  0.64
980 : loss:  0.8065493 	 acc:  0.64
982 : loss:  0.9026742 	 acc:  0.57
984 : loss:  0.76900804 	 acc:  0.64
986 : loss:  0.72861236 	 acc:  0.68
988 : loss:  0.8304026 	 acc:  0.61
990 : loss:  0.82975936 	 acc:  0.65
992 : loss:  0.81939393 	 acc:  0.64
994 : loss:  0.7555493 	 acc:  0.64
996 : loss:  0.85826576 	 acc:  0.56
998 : loss:  0.8151465 	 acc:  0.62
1000 : loss:  0.76168656 	 acc:  0.69
1002 : loss:  0.83703446 	 acc:  0.62
1004 : loss:  0.7717299 	 acc:  0.65
1006 : loss:  0.9047926 	 acc:  0.55
1008 : loss:  0.8209413 	 acc:  0.68
1010 : loss:  0.8980797 	 acc:  0.6
1012 : loss:  1.0272151 	 acc:  0.52
1014 : loss:  0.85359 	 acc:  0.57
1016 : loss:  0.7844055 	 acc:  0.61
1018 : loss:  0.87517977 	 acc:  0.63
1020 : loss:  0.89141166 	 acc:  0.64
1022 : loss:  0.85604095 	 acc:  0.5
1024 : loss:  0.9327866 	 acc:  0.59
1026 : loss:  0.8064422 	 acc:  0.6
1028 : loss:  0.86629355 	 acc:  0.64
1030 : loss:  0.9101075 	 acc:  0.62
1032 : loss:  0.7818833 	 acc:  0.61
1034 : loss:  0.753937 	 acc:  0.7
1036 : loss:  0.91287696 	 acc:  0.52
1038 : loss:  0.7742487 	 acc:  0.64
1040 : loss:  0.74591994 	 acc:  0.62
1042 : loss:  0.850311 	 acc:  0.64
1044 : loss:  1.0027279 	 acc:  0.53333336
1046 : loss:  0.8186005 	 acc:  0.63
1048 : loss:  0.7989224 	 acc:  0.65
1050 : loss:  0.7592784 	 acc:  0.57
1052 : loss:  0.8779071 	 acc:  0.6
1054 : loss:  0.85379547 	 acc:  0.62
1056 : loss:  0.8552608 	 acc:  0.56
1058 : loss:  0.8610691 	 acc:  0.61
1060 : loss:  0.78731096 	 acc:  0.68
1062 : loss:  0.81315345 	 acc:  0.64
1064 : loss:  0.88158095 	 acc:  0.58
1066 : loss:  0.8195771 	 acc:  0.58
1068 : loss:  0.8161502 	 acc:  0.55
1070 : loss:  0.8930482 	 acc:  0.52
1072 : loss:  0.78865236 	 acc:  0.63
1074 : loss:  0.7982038 	 acc:  0.66
1076 : loss:  0.74471766 	 acc:  0.65
1078 : loss:  0.8012132 	 acc:  0.61
1080 : loss:  0.87272143 	 acc:  0.56
1082 : loss:  0.82962286 	 acc:  0.64
1084 : loss:  0.77428675 	 acc:  0.63
1086 : loss:  0.8191651 	 acc:  0.66
1088 : loss:  0.8000592 	 acc:  0.67
1090 : loss:  0.7550699 	 acc:  0.67
1092 : loss:  0.8767844 	 acc:  0.56
1094 : loss:  0.84839404 	 acc:  0.56
1096 : loss:  0.80495626 	 acc:  0.62
1098 : loss:  0.86821115 	 acc:  0.58
1100 : loss:  0.82086456 	 acc:  0.63
1102 : loss:  0.79463285 	 acc:  0.63
1104 : loss:  0.8374443 	 acc:  0.58
1106 : loss:  0.82702273 	 acc:  0.62
1108 : loss:  0.7743819 	 acc:  0.64
1110 : loss:  0.87174714 	 acc:  0.55
1112 : loss:  0.8578208 	 acc:  0.59
1114 : loss:  0.87264913 	 acc:  0.6

Saving...
saved to models/CNN/pretrained_model.ckpt-1116

100 	 [68.00407527 59.20402001 55.66152353]
2 	val accuracy:  0.6152 	 f_! score:  [0.68004075 0.5920402  0.55661524]

1116 : loss:  0.8374479 	 acc:  0.64
1118 : loss:  0.798242 	 acc:  0.63
1120 : loss:  0.8245332 	 acc:  0.62
1122 : loss:  0.8061949 	 acc:  0.65
1124 : loss:  0.7576754 	 acc:  0.64
1126 : loss:  0.7066947 	 acc:  0.7
1128 : loss:  0.8215691 	 acc:  0.63
1130 : loss:  0.8307767 	 acc:  0.59
1132 : loss:  0.6980708 	 acc:  0.67
1134 : loss:  0.89064866 	 acc:  0.6
1136 : loss:  0.863314 	 acc:  0.66
1138 : loss:  0.7801137 	 acc:  0.66
1140 : loss:  0.84156716 	 acc:  0.62
1142 : loss:  0.89032876 	 acc:  0.53
1144 : loss:  0.8059279 	 acc:  0.63
1146 : loss:  0.7649214 	 acc:  0.66
1148 : loss:  0.831717 	 acc:  0.58
1150 : loss:  0.74179846 	 acc:  0.66
1152 : loss:  0.73629403 	 acc:  0.63
1154 : loss:  0.90545523 	 acc:  0.62
1156 : loss:  0.76448077 	 acc:  0.65
1158 : loss:  0.7602849 	 acc:  0.69
1160 : loss:  0.84089553 	 acc:  0.62
1162 : loss:  0.8804692 	 acc:  0.55
1164 : loss:  0.81214744 	 acc:  0.63
1166 : loss:  0.9069568 	 acc:  0.57
1168 : loss:  0.8355713 	 acc:  0.56
1170 : loss:  0.67779577 	 acc:  0.72
1172 : loss:  0.76959664 	 acc:  0.64
1174 : loss:  0.7908576 	 acc:  0.63
1176 : loss:  0.82413644 	 acc:  0.59
1178 : loss:  0.8235163 	 acc:  0.6
1180 : loss:  0.96377677 	 acc:  0.54
1182 : loss:  0.7496248 	 acc:  0.67
1184 : loss:  0.7051743 	 acc:  0.7
1186 : loss:  0.81840587 	 acc:  0.58
1188 : loss:  0.84844285 	 acc:  0.6
1190 : loss:  0.90097463 	 acc:  0.51
1192 : loss:  0.7752157 	 acc:  0.66
1194 : loss:  0.8168011 	 acc:  0.6
1196 : loss:  0.7743547 	 acc:  0.64
1198 : loss:  0.6603428 	 acc:  0.69
1200 : loss:  0.64035875 	 acc:  0.74
1202 : loss:  0.80854684 	 acc:  0.63
1204 : loss:  0.85524553 	 acc:  0.5
1206 : loss:  0.8448329 	 acc:  0.59
1208 : loss:  0.71470284 	 acc:  0.65
1210 : loss:  0.7011075 	 acc:  0.68
1212 : loss:  0.98390234 	 acc:  0.5
1214 : loss:  0.7035339 	 acc:  0.67
1216 : loss:  0.8231894 	 acc:  0.57
1218 : loss:  0.8300731 	 acc:  0.54
1220 : loss:  0.84530294 	 acc:  0.6
1222 : loss:  0.9032358 	 acc:  0.54
1224 : loss:  0.93504274 	 acc:  0.51
1226 : loss:  0.8581582 	 acc:  0.57
1228 : loss:  0.88011914 	 acc:  0.58
1230 : loss:  0.8160182 	 acc:  0.59
1232 : loss:  0.71549463 	 acc:  0.68
1234 : loss:  0.6815872 	 acc:  0.7
1236 : loss:  0.89000535 	 acc:  0.57
1238 : loss:  0.7922279 	 acc:  0.65
1240 : loss:  0.73871475 	 acc:  0.7
1242 : loss:  0.7555055 	 acc:  0.66
1244 : loss:  0.7842803 	 acc:  0.64
1246 : loss:  0.82451975 	 acc:  0.62
1248 : loss:  0.8544957 	 acc:  0.62
1250 : loss:  0.801242 	 acc:  0.64
1252 : loss:  0.78397095 	 acc:  0.65
1254 : loss:  0.897483 	 acc:  0.58
1256 : loss:  0.8549489 	 acc:  0.69
1258 : loss:  0.7635103 	 acc:  0.6
1260 : loss:  0.7755849 	 acc:  0.65
1262 : loss:  0.8817119 	 acc:  0.55
1264 : loss:  0.73386604 	 acc:  0.63
1266 : loss:  0.7255461 	 acc:  0.65
1268 : loss:  0.7602293 	 acc:  0.61
1270 : loss:  0.79265696 	 acc:  0.61
1272 : loss:  0.72025377 	 acc:  0.68
1274 : loss:  0.8262498 	 acc:  0.6
1276 : loss:  0.82926744 	 acc:  0.64
1278 : loss:  0.6333216 	 acc:  0.71
1280 : loss:  0.8621807 	 acc:  0.53
1282 : loss:  0.72211486 	 acc:  0.67
1284 : loss:  0.8454315 	 acc:  0.58
1286 : loss:  0.77574813 	 acc:  0.6
1288 : loss:  0.8261755 	 acc:  0.61
1290 : loss:  0.8701512 	 acc:  0.56
1292 : loss:  0.83042735 	 acc:  0.65
1294 : loss:  0.833744 	 acc:  0.59
1296 : loss:  0.85665977 	 acc:  0.61
1298 : loss:  0.83650434 	 acc:  0.61
1300 : loss:  0.813003 	 acc:  0.68
1302 : loss:  0.7273748 	 acc:  0.68
1304 : loss:  0.6552334 	 acc:  0.72
1306 : loss:  0.81532395 	 acc:  0.63
1308 : loss:  0.7862936 	 acc:  0.59
1310 : loss:  0.76932937 	 acc:  0.61
1312 : loss:  0.83281046 	 acc:  0.62
1314 : loss:  0.7327127 	 acc:  0.63
1316 : loss:  0.84414244 	 acc:  0.61
1318 : loss:  0.7796562 	 acc:  0.64
1320 : loss:  0.78260267 	 acc:  0.68
1322 : loss:  0.8110664 	 acc:  0.59
1324 : loss:  0.6608682 	 acc:  0.71
1326 : loss:  0.76750815 	 acc:  0.64
1328 : loss:  0.88758314 	 acc:  0.57
1330 : loss:  0.8569902 	 acc:  0.63
1332 : loss:  0.71559906 	 acc:  0.68
1334 : loss:  0.7979134 	 acc:  0.63
1336 : loss:  0.81974816 	 acc:  0.58
1338 : loss:  0.85028 	 acc:  0.58
1340 : loss:  0.85565877 	 acc:  0.58
1342 : loss:  0.8013041 	 acc:  0.6
1344 : loss:  0.78791314 	 acc:  0.67
1346 : loss:  0.7304065 	 acc:  0.68
1348 : loss:  0.81284493 	 acc:  0.64
1350 : loss:  0.80382645 	 acc:  0.59
1352 : loss:  0.82333994 	 acc:  0.63
1354 : loss:  0.950031 	 acc:  0.57
1356 : loss:  0.77607626 	 acc:  0.7
1358 : loss:  0.7731214 	 acc:  0.62
1360 : loss:  0.76567185 	 acc:  0.64
1362 : loss:  0.8176233 	 acc:  0.61
1364 : loss:  0.8327692 	 acc:  0.59
1366 : loss:  0.80905133 	 acc:  0.64
1368 : loss:  0.80167943 	 acc:  0.62
1370 : loss:  0.78339225 	 acc:  0.73
1372 : loss:  0.72866607 	 acc:  0.65
1374 : loss:  0.781993 	 acc:  0.63
1376 : loss:  0.8020469 	 acc:  0.64
1378 : loss:  0.77603334 	 acc:  0.59
1380 : loss:  0.7292123 	 acc:  0.63
1382 : loss:  0.756955 	 acc:  0.67
1384 : loss:  0.743364 	 acc:  0.61
1386 : loss:  0.8365604 	 acc:  0.64
1388 : loss:  0.88464254 	 acc:  0.58
1390 : loss:  0.83136445 	 acc:  0.64
1392 : loss:  0.7584007 	 acc:  0.67
1394 : loss:  0.76938033 	 acc:  0.66
1396 : loss:  0.95138407 	 acc:  0.52
1398 : loss:  0.8356646 	 acc:  0.65
1400 : loss:  0.85829896 	 acc:  0.61
1402 : loss:  0.8228665 	 acc:  0.56
1404 : loss:  0.8113003 	 acc:  0.61
1406 : loss:  0.6794597 	 acc:  0.7
1408 : loss:  0.8227909 	 acc:  0.65
1410 : loss:  0.79594636 	 acc:  0.67
1412 : loss:  0.61499316 	 acc:  0.71
1414 : loss:  0.7473247 	 acc:  0.69
1416 : loss:  0.8138994 	 acc:  0.64
1418 : loss:  0.8045189 	 acc:  0.63
1420 : loss:  0.73324275 	 acc:  0.68
1422 : loss:  0.7630031 	 acc:  0.63
1424 : loss:  0.75218153 	 acc:  0.66
1426 : loss:  0.80010486 	 acc:  0.67
1428 : loss:  0.8358917 	 acc:  0.63
1430 : loss:  0.83864534 	 acc:  0.55
1432 : loss:  0.73336565 	 acc:  0.69
1434 : loss:  0.8275536 	 acc:  0.63
1436 : loss:  0.70529604 	 acc:  0.71
1438 : loss:  0.78654695 	 acc:  0.66
1440 : loss:  0.80242383 	 acc:  0.66
1442 : loss:  0.79834265 	 acc:  0.67
1444 : loss:  0.81637853 	 acc:  0.63
1446 : loss:  0.6910537 	 acc:  0.71
1448 : loss:  0.83787644 	 acc:  0.58
1450 : loss:  0.82682234 	 acc:  0.55
1452 : loss:  0.7844995 	 acc:  0.62
1454 : loss:  0.78999573 	 acc:  0.59
1456 : loss:  0.83576095 	 acc:  0.61
1458 : loss:  0.75215775 	 acc:  0.64
1460 : loss:  0.7493878 	 acc:  0.7
1462 : loss:  0.71231186 	 acc:  0.72
1464 : loss:  0.76321626 	 acc:  0.62
1466 : loss:  0.7064681 	 acc:  0.71
1468 : loss:  0.82247406 	 acc:  0.61
1470 : loss:  0.743334 	 acc:  0.68
1472 : loss:  0.95997065 	 acc:  0.52
1474 : loss:  0.72119385 	 acc:  0.64
1476 : loss:  0.73396254 	 acc:  0.67
1478 : loss:  0.7269298 	 acc:  0.68
1480 : loss:  0.84901327 	 acc:  0.63
1482 : loss:  0.82872653 	 acc:  0.55
1484 : loss:  0.7409689 	 acc:  0.68
1486 : loss:  0.814691 	 acc:  0.6

Saving...
saved to models/CNN/pretrained_model.ckpt-1488

100 	 [68.13240435 61.45397971 56.64260352]
3 	val accuracy:  0.6253 	 f_! score:  [0.68132404 0.6145398  0.56642604]

100 	 [68.13240435 61.45397971 56.64260352]
100 	 [68.04246223 59.79499354 59.79544474]
100 	 [68.91834313 63.99749634 54.51002601]
100 	 [3348. 3361. 3291.]
--- Test   Twitter ---
0.6253
f1:  [0.68132404 0.6145398  0.56642604]
--- 126.79055953025818 seconds ---



Review
delete old models
/home/yannik/ba/TestHelper.py:64: DtypeWarning: Columns (1) have mixed types. Specify dtype option on import or set low_memory=False.
  input_data = self.decide_which_input(input_name)
509411
Tensor("ConvNet/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
Tensor("ConvNet_1/dense/BiasAdd:0", shape=(?, 3), dtype=float32)
---init ready   CNN ---
2018-10-14 19:13:08.161745: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-10-14 19:13:08.233804: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-10-14 19:13:08.234157: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 970 major: 5 minor: 2 memoryClockRate(GHz): 1.253
pciBusID: 0000:01:00.0
totalMemory: 3.94GiB freeMemory: 3.20GiB
2018-10-14 19:13:08.234174: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-10-14 19:13:08.948583: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-14 19:13:08.948613: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-10-14 19:13:08.948618: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-10-14 19:13:08.948762: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2909 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/CNN/pretrained_model.ckpt-0
2018-10-14 19:13:30.923734: W tensorflow/core/common_runtime/bfc_allocator.cc:219] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.60GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
0 : loss:  3.4158077 	 acc:  0.42
40 : loss:  1.0937879 	 acc:  0.34
80 : loss:  1.0901808 	 acc:  0.37
120 : loss:  1.0771583 	 acc:  0.42
160 : loss:  1.0962539 	 acc:  0.33
200 : loss:  1.0478042 	 acc:  0.48
240 : loss:  0.99608994 	 acc:  0.49
280 : loss:  0.94775033 	 acc:  0.6
320 : loss:  1.1484654 	 acc:  0.42
360 : loss:  0.8867984 	 acc:  0.55
400 : loss:  0.9469397 	 acc:  0.57
440 : loss:  0.9436441 	 acc:  0.56
480 : loss:  0.9965769 	 acc:  0.52
520 : loss:  0.89660144 	 acc:  0.58
560 : loss:  0.87307894 	 acc:  0.62
600 : loss:  0.9197851 	 acc:  0.59
640 : loss:  1.0657759 	 acc:  0.48
680 : loss:  0.86176735 	 acc:  0.58
720 : loss:  0.9956881 	 acc:  0.53
760 : loss:  0.8790946 	 acc:  0.56
800 : loss:  0.84768844 	 acc:  0.57
840 : loss:  0.99406195 	 acc:  0.45
880 : loss:  0.999329 	 acc:  0.51
920 : loss:  0.8981774 	 acc:  0.54
960 : loss:  0.89573395 	 acc:  0.61
1000 : loss:  0.91770875 	 acc:  0.6
1040 : loss:  0.8258844 	 acc:  0.58
1080 : loss:  0.8684559 	 acc:  0.55
1120 : loss:  0.83406585 	 acc:  0.63
1160 : loss:  0.9139702 	 acc:  0.56
1200 : loss:  0.76917267 	 acc:  0.68
1240 : loss:  0.9905469 	 acc:  0.52
1280 : loss:  0.8922771 	 acc:  0.57
1320 : loss:  0.79247355 	 acc:  0.64
1360 : loss:  0.7951636 	 acc:  0.62
1400 : loss:  0.74747527 	 acc:  0.67
1440 : loss:  0.80106825 	 acc:  0.58
1480 : loss:  0.82542366 	 acc:  0.64
1520 : loss:  0.7592897 	 acc:  0.66
1560 : loss:  0.797505 	 acc:  0.61
1600 : loss:  0.8498642 	 acc:  0.6
1640 : loss:  0.80124015 	 acc:  0.6
1680 : loss:  0.8216262 	 acc:  0.62
1720 : loss:  0.88814044 	 acc:  0.59
1760 : loss:  0.7548463 	 acc:  0.66
1800 : loss:  0.8669758 	 acc:  0.64
1840 : loss:  0.90774584 	 acc:  0.57
1880 : loss:  0.7263518 	 acc:  0.67
1920 : loss:  1.0388049 	 acc:  0.56
1960 : loss:  0.7432388 	 acc:  0.62
2000 : loss:  0.7735449 	 acc:  0.61
2040 : loss:  0.7815848 	 acc:  0.68
2080 : loss:  0.89515394 	 acc:  0.59
2120 : loss:  0.7912368 	 acc:  0.66
2160 : loss:  0.8181281 	 acc:  0.64
2200 : loss:  0.83294713 	 acc:  0.64
2240 : loss:  0.81361085 	 acc:  0.61
2280 : loss:  0.8137689 	 acc:  0.64
2320 : loss:  0.84810853 	 acc:  0.59
2360 : loss:  0.7210412 	 acc:  0.66
2400 : loss:  0.8028317 	 acc:  0.63
2440 : loss:  0.7400426 	 acc:  0.66
2480 : loss:  0.74137133 	 acc:  0.65
2520 : loss:  0.8028833 	 acc:  0.62
2560 : loss:  0.70283675 	 acc:  0.69
2600 : loss:  0.731048 	 acc:  0.63
2640 : loss:  0.9237411 	 acc:  0.53
2680 : loss:  0.73311144 	 acc:  0.65
2720 : loss:  0.7256976 	 acc:  0.71
2760 : loss:  0.7747143 	 acc:  0.67
2800 : loss:  0.72078454 	 acc:  0.68
2840 : loss:  0.6754612 	 acc:  0.73
2880 : loss:  0.7485888 	 acc:  0.62
2920 : loss:  0.81447005 	 acc:  0.61
2960 : loss:  0.7982395 	 acc:  0.65
3000 : loss:  0.85933006 	 acc:  0.6
3040 : loss:  0.7063598 	 acc:  0.69
3080 : loss:  0.81778723 	 acc:  0.71
3120 : loss:  0.86267793 	 acc:  0.6
3160 : loss:  0.78575134 	 acc:  0.63
3200 : loss:  0.674361 	 acc:  0.68
3240 : loss:  0.87773496 	 acc:  0.63
3280 : loss:  0.7372372 	 acc:  0.69
3320 : loss:  0.77326536 	 acc:  0.65
3360 : loss:  0.68893886 	 acc:  0.7
3400 : loss:  0.7741759 	 acc:  0.61
3440 : loss:  0.6408121 	 acc:  0.67
3480 : loss:  0.7791295 	 acc:  0.62
3520 : loss:  0.83541924 	 acc:  0.6
3560 : loss:  0.74464816 	 acc:  0.66
3600 : loss:  0.69122523 	 acc:  0.7
3640 : loss:  0.7140509 	 acc:  0.67
3680 : loss:  0.7113712 	 acc:  0.67
3720 : loss:  0.6593813 	 acc:  0.7
3760 : loss:  0.65170664 	 acc:  0.7
3800 : loss:  0.7567583 	 acc:  0.65
3840 : loss:  0.60148495 	 acc:  0.74
3880 : loss:  0.73407483 	 acc:  0.66
3920 : loss:  0.93252105 	 acc:  0.58
3960 : loss:  0.7367979 	 acc:  0.68
4000 : loss:  0.8041996 	 acc:  0.59
4040 : loss:  0.8140498 	 acc:  0.63
4080 : loss:  0.7208706 	 acc:  0.69
4120 : loss:  0.7550926 	 acc:  0.65
4160 : loss:  0.7104865 	 acc:  0.65
4200 : loss:  0.6546783 	 acc:  0.68
4240 : loss:  0.7477385 	 acc:  0.65
4280 : loss:  0.7777145 	 acc:  0.66
4320 : loss:  0.7958049 	 acc:  0.68
4360 : loss:  0.82585907 	 acc:  0.56
4400 : loss:  0.79266167 	 acc:  0.6
4440 : loss:  0.7611609 	 acc:  0.71
4480 : loss:  0.8355953 	 acc:  0.56

Saving...
saved to models/CNN/pretrained_model.ckpt-4482

500 	 [360.18355297 214.3715991  373.68958486]
0 	val accuracy:  0.66608 	 f_! score:  [0.72036711 0.4287432  0.74737917]

4520 : loss:  0.83126336 	 acc:  0.62
4560 : loss:  0.8056517 	 acc:  0.61
4600 : loss:  0.80433065 	 acc:  0.62
4640 : loss:  0.71976197 	 acc:  0.71
4680 : loss:  0.6869752 	 acc:  0.7
4720 : loss:  0.70127 	 acc:  0.67
4760 : loss:  0.8261869 	 acc:  0.65
4800 : loss:  0.64658093 	 acc:  0.71
4840 : loss:  0.7561858 	 acc:  0.64
4880 : loss:  0.70826995 	 acc:  0.73
4920 : loss:  0.700926 	 acc:  0.67
4960 : loss:  0.79232985 	 acc:  0.63
5000 : loss:  0.8045391 	 acc:  0.62
5040 : loss:  0.6332895 	 acc:  0.7
5080 : loss:  0.78649306 	 acc:  0.66
5120 : loss:  0.7108078 	 acc:  0.64
5160 : loss:  0.6504906 	 acc:  0.75
5200 : loss:  0.82321787 	 acc:  0.61
5240 : loss:  0.771487 	 acc:  0.69
5280 : loss:  0.70092034 	 acc:  0.69
5320 : loss:  0.82261145 	 acc:  0.71
5360 : loss:  0.7001312 	 acc:  0.7
5400 : loss:  0.7353316 	 acc:  0.71
5440 : loss:  0.66848874 	 acc:  0.69
5480 : loss:  0.80901545 	 acc:  0.63
5520 : loss:  0.78747314 	 acc:  0.61
5560 : loss:  0.7074746 	 acc:  0.7
5600 : loss:  0.8451119 	 acc:  0.59
5640 : loss:  0.83784413 	 acc:  0.59
5680 : loss:  0.73997253 	 acc:  0.69
5720 : loss:  0.79579896 	 acc:  0.65
5760 : loss:  0.79598117 	 acc:  0.63
5800 : loss:  0.75568503 	 acc:  0.58
5840 : loss:  0.74230814 	 acc:  0.71
5880 : loss:  0.68332046 	 acc:  0.71
5920 : loss:  0.76992995 	 acc:  0.7
5960 : loss:  0.843854 	 acc:  0.61
6000 : loss:  0.9000223 	 acc:  0.6
6040 : loss:  0.69262105 	 acc:  0.7
6080 : loss:  0.85022 	 acc:  0.59
6120 : loss:  0.7511992 	 acc:  0.66
6160 : loss:  0.7110898 	 acc:  0.73
6200 : loss:  0.85605913 	 acc:  0.63
6240 : loss:  0.6418558 	 acc:  0.74
6280 : loss:  0.76213783 	 acc:  0.57
6320 : loss:  0.7820173 	 acc:  0.63
6360 : loss:  0.7011676 	 acc:  0.72
6400 : loss:  0.66701454 	 acc:  0.75
6440 : loss:  0.6691481 	 acc:  0.72
6480 : loss:  0.6794632 	 acc:  0.75
6520 : loss:  0.68784356 	 acc:  0.67
6560 : loss:  0.7752898 	 acc:  0.66
6600 : loss:  0.7162954 	 acc:  0.69
6640 : loss:  0.8524653 	 acc:  0.59
6680 : loss:  0.7787597 	 acc:  0.66
6720 : loss:  0.6753902 	 acc:  0.7
6760 : loss:  0.6368454 	 acc:  0.67
6800 : loss:  0.69120765 	 acc:  0.71
6840 : loss:  0.80164224 	 acc:  0.64
6880 : loss:  0.6649372 	 acc:  0.71
6920 : loss:  0.7075997 	 acc:  0.65
6960 : loss:  0.71679175 	 acc:  0.68
7000 : loss:  0.92080426 	 acc:  0.54
7040 : loss:  0.82294744 	 acc:  0.64
7080 : loss:  0.818437 	 acc:  0.62
7120 : loss:  0.6776078 	 acc:  0.66
7160 : loss:  0.7122065 	 acc:  0.69
7200 : loss:  0.73752755 	 acc:  0.68
7240 : loss:  0.7483578 	 acc:  0.63
7280 : loss:  0.8123509 	 acc:  0.58
7320 : loss:  0.69952285 	 acc:  0.67
7360 : loss:  0.70032233 	 acc:  0.66
7400 : loss:  0.7541088 	 acc:  0.55
7440 : loss:  0.72128755 	 acc:  0.69
7480 : loss:  0.72269255 	 acc:  0.67
7520 : loss:  0.7847202 	 acc:  0.64
7560 : loss:  0.8432655 	 acc:  0.66
7600 : loss:  0.69187844 	 acc:  0.71
7640 : loss:  0.61838007 	 acc:  0.74
7680 : loss:  0.70352817 	 acc:  0.67
7720 : loss:  0.6689263 	 acc:  0.7
7760 : loss:  0.8198223 	 acc:  0.6
7800 : loss:  0.8391735 	 acc:  0.63
7840 : loss:  0.6640459 	 acc:  0.73
7880 : loss:  0.7148416 	 acc:  0.67
7920 : loss:  0.7827593 	 acc:  0.61
7960 : loss:  0.6501141 	 acc:  0.68
8000 : loss:  0.73337096 	 acc:  0.68
8040 : loss:  0.6344439 	 acc:  0.74
8080 : loss:  0.6839651 	 acc:  0.7
8120 : loss:  0.6393751 	 acc:  0.71
8160 : loss:  0.6355728 	 acc:  0.77
8200 : loss:  0.7754833 	 acc:  0.64
8240 : loss:  0.64250314 	 acc:  0.72
8280 : loss:  0.70943624 	 acc:  0.66
8320 : loss:  0.728397 	 acc:  0.66
8360 : loss:  0.7473037 	 acc:  0.68
8400 : loss:  0.62275374 	 acc:  0.71
8440 : loss:  0.81030625 	 acc:  0.67
8480 : loss:  0.69708174 	 acc:  0.66
8520 : loss:  0.67703056 	 acc:  0.71
8560 : loss:  0.6891906 	 acc:  0.69
8600 : loss:  0.6967709 	 acc:  0.67
8640 : loss:  0.73316646 	 acc:  0.7
8680 : loss:  0.77060026 	 acc:  0.68
8720 : loss:  0.64606553 	 acc:  0.69
8760 : loss:  0.6810677 	 acc:  0.72
8800 : loss:  0.6690921 	 acc:  0.71
8840 : loss:  0.66555667 	 acc:  0.67
8880 : loss:  0.6885836 	 acc:  0.76
8920 : loss:  0.8334726 	 acc:  0.6
8960 : loss:  0.7892181 	 acc:  0.63

Saving...
saved to models/CNN/pretrained_model.ckpt-8965

500 	 [362.59091681 240.12434927 381.92956901]
1 	val accuracy:  0.68143994 	 f_! score:  [0.72518183 0.4802487  0.76385914]

9000 : loss:  0.5637629 	 acc:  0.75
9040 : loss:  0.7256748 	 acc:  0.73
9080 : loss:  0.6411339 	 acc:  0.72
9120 : loss:  0.7370502 	 acc:  0.66
9160 : loss:  0.67027235 	 acc:  0.65
9200 : loss:  0.61116815 	 acc:  0.76
9240 : loss:  0.7743714 	 acc:  0.66
9280 : loss:  0.56911945 	 acc:  0.79
9320 : loss:  0.61761594 	 acc:  0.72
9360 : loss:  0.72141004 	 acc:  0.66
9400 : loss:  0.6848654 	 acc:  0.71
9440 : loss:  0.70900786 	 acc:  0.67
9480 : loss:  0.68709964 	 acc:  0.74
9520 : loss:  0.64800924 	 acc:  0.7
9560 : loss:  0.5419647 	 acc:  0.75
9600 : loss:  0.66374046 	 acc:  0.73
9640 : loss:  0.616805 	 acc:  0.73
9680 : loss:  0.65409577 	 acc:  0.67
9720 : loss:  0.7279402 	 acc:  0.67
9760 : loss:  0.7235052 	 acc:  0.63
9800 : loss:  0.69459623 	 acc:  0.69
9840 : loss:  0.7550485 	 acc:  0.67
9880 : loss:  0.73735434 	 acc:  0.69
9920 : loss:  0.6384771 	 acc:  0.71
9960 : loss:  0.7259587 	 acc:  0.68
10000 : loss:  0.7468712 	 acc:  0.66
10040 : loss:  0.61593634 	 acc:  0.74
10080 : loss:  0.6857829 	 acc:  0.65
10120 : loss:  0.7880007 	 acc:  0.69
10160 : loss:  0.62800723 	 acc:  0.76
10200 : loss:  0.6389976 	 acc:  0.77
10240 : loss:  0.7653331 	 acc:  0.63
10280 : loss:  0.7102903 	 acc:  0.67
10320 : loss:  0.6457977 	 acc:  0.7
10360 : loss:  0.63964415 	 acc:  0.7
10400 : loss:  0.6624317 	 acc:  0.69
10440 : loss:  0.6721993 	 acc:  0.68
10480 : loss:  0.5525448 	 acc:  0.77
10520 : loss:  0.7294609 	 acc:  0.65
10560 : loss:  0.6755748 	 acc:  0.68
10600 : loss:  0.7620212 	 acc:  0.62
10640 : loss:  0.7429325 	 acc:  0.62
10680 : loss:  0.67171425 	 acc:  0.7
10720 : loss:  0.68173623 	 acc:  0.67
10760 : loss:  0.6822349 	 acc:  0.65
10800 : loss:  0.6399608 	 acc:  0.69
10840 : loss:  0.7292378 	 acc:  0.66
10880 : loss:  0.6135972 	 acc:  0.69
10920 : loss:  0.66918784 	 acc:  0.71
10960 : loss:  0.61263674 	 acc:  0.74
11000 : loss:  0.78839254 	 acc:  0.64
11040 : loss:  0.6991182 	 acc:  0.68
11080 : loss:  0.8381313 	 acc:  0.62
11120 : loss:  0.6892076 	 acc:  0.74
11160 : loss:  0.7335628 	 acc:  0.67
11200 : loss:  0.7236808 	 acc:  0.67
11240 : loss:  0.70110166 	 acc:  0.65
11280 : loss:  0.6315344 	 acc:  0.69
11320 : loss:  0.66363585 	 acc:  0.71
11360 : loss:  0.75198233 	 acc:  0.61
11400 : loss:  0.6519081 	 acc:  0.74
11440 : loss:  0.62226874 	 acc:  0.71
11480 : loss:  0.6925161 	 acc:  0.72
11520 : loss:  0.8529503 	 acc:  0.62
11560 : loss:  0.7147427 	 acc:  0.67
11600 : loss:  0.7094514 	 acc:  0.66
11640 : loss:  0.6537741 	 acc:  0.74
11680 : loss:  0.62729573 	 acc:  0.71
11720 : loss:  0.76665246 	 acc:  0.65
11760 : loss:  0.53531134 	 acc:  0.77
11800 : loss:  0.6286985 	 acc:  0.73
11840 : loss:  0.66865945 	 acc:  0.67
11880 : loss:  0.6647736 	 acc:  0.69
11920 : loss:  0.66150594 	 acc:  0.74
11960 : loss:  0.74694365 	 acc:  0.67
12000 : loss:  0.6751289 	 acc:  0.69
12040 : loss:  0.67897856 	 acc:  0.71
12080 : loss:  0.7155613 	 acc:  0.64
12120 : loss:  0.7522671 	 acc:  0.63
12160 : loss:  0.6927979 	 acc:  0.72
12200 : loss:  0.6459751 	 acc:  0.76
12240 : loss:  0.6715238 	 acc:  0.67
12280 : loss:  0.67969084 	 acc:  0.74
12320 : loss:  0.6388649 	 acc:  0.71
12360 : loss:  0.71890074 	 acc:  0.63
12400 : loss:  0.7286377 	 acc:  0.66
12440 : loss:  0.6899586 	 acc:  0.73
12480 : loss:  0.64957845 	 acc:  0.73
12520 : loss:  0.7108284 	 acc:  0.72
12560 : loss:  0.7053816 	 acc:  0.68
12600 : loss:  0.66543394 	 acc:  0.76
12640 : loss:  0.69975007 	 acc:  0.67
12680 : loss:  0.69879556 	 acc:  0.69
12720 : loss:  0.6210363 	 acc:  0.75
12760 : loss:  0.65802145 	 acc:  0.72
12800 : loss:  0.75191617 	 acc:  0.61
12840 : loss:  0.77475893 	 acc:  0.66
12880 : loss:  0.71796256 	 acc:  0.64
12920 : loss:  0.7739988 	 acc:  0.69
12960 : loss:  0.7778166 	 acc:  0.63
13000 : loss:  0.7183276 	 acc:  0.66
13040 : loss:  0.6451334 	 acc:  0.72
13080 : loss:  0.85910994 	 acc:  0.55
13120 : loss:  0.61650693 	 acc:  0.72
13160 : loss:  0.70446944 	 acc:  0.67
13200 : loss:  0.8294563 	 acc:  0.68
13240 : loss:  0.65854096 	 acc:  0.74
13280 : loss:  0.74015975 	 acc:  0.65
13320 : loss:  0.6214726 	 acc:  0.74
13360 : loss:  0.66366243 	 acc:  0.7
13400 : loss:  0.6559787 	 acc:  0.69
13440 : loss:  0.5929073 	 acc:  0.69

Saving...
saved to models/CNN/pretrained_model.ckpt-13448

500 	 [370.26167705 249.29312797 382.33225812]
2 	val accuracy:  0.68868 	 f_! score:  [0.74052335 0.49858626 0.76466452]

13480 : loss:  0.5704578 	 acc:  0.75
13520 : loss:  0.5807052 	 acc:  0.76
13560 : loss:  0.7161737 	 acc:  0.65
13600 : loss:  0.55881333 	 acc:  0.76
13640 : loss:  0.6143949 	 acc:  0.75
13680 : loss:  0.5493357 	 acc:  0.77
13720 : loss:  0.72566766 	 acc:  0.65
13760 : loss:  0.8423865 	 acc:  0.59
13800 : loss:  0.6451138 	 acc:  0.73
13840 : loss:  0.8781177 	 acc:  0.57
13880 : loss:  0.55028296 	 acc:  0.79
13920 : loss:  0.5973271 	 acc:  0.76
13960 : loss:  0.583444 	 acc:  0.72
14000 : loss:  0.74551773 	 acc:  0.69
14040 : loss:  0.7305385 	 acc:  0.66
14080 : loss:  0.62830657 	 acc:  0.71
14120 : loss:  0.8460731 	 acc:  0.63
14160 : loss:  0.69674003 	 acc:  0.66
14200 : loss:  0.5939324 	 acc:  0.73
14240 : loss:  0.65133595 	 acc:  0.7
14280 : loss:  0.7015299 	 acc:  0.68
14320 : loss:  0.7549672 	 acc:  0.67
14360 : loss:  0.68605024 	 acc:  0.68
14400 : loss:  0.5478569 	 acc:  0.81
14440 : loss:  0.73903656 	 acc:  0.63
14480 : loss:  0.727541 	 acc:  0.68
14520 : loss:  0.6422802 	 acc:  0.68
14560 : loss:  0.6049904 	 acc:  0.75
14600 : loss:  0.7058106 	 acc:  0.66
14640 : loss:  0.744018 	 acc:  0.7
14680 : loss:  0.70221627 	 acc:  0.69
14720 : loss:  0.6593522 	 acc:  0.69
14760 : loss:  0.69276917 	 acc:  0.69
14800 : loss:  0.6664209 	 acc:  0.75
14840 : loss:  0.7893123 	 acc:  0.64
14880 : loss:  0.9326432 	 acc:  0.57
14920 : loss:  0.52148736 	 acc:  0.8
14960 : loss:  0.626034 	 acc:  0.7
15000 : loss:  0.6900204 	 acc:  0.69
15040 : loss:  0.8109758 	 acc:  0.67
15080 : loss:  0.72294945 	 acc:  0.72
15120 : loss:  0.71974236 	 acc:  0.69
15160 : loss:  0.7496388 	 acc:  0.66
15200 : loss:  0.76210284 	 acc:  0.65
15240 : loss:  0.7160614 	 acc:  0.68
15280 : loss:  0.7021406 	 acc:  0.65
15320 : loss:  0.7372399 	 acc:  0.63
15360 : loss:  0.75519925 	 acc:  0.66
15400 : loss:  0.6102139 	 acc:  0.72
15440 : loss:  0.5868723 	 acc:  0.73
15480 : loss:  0.7201911 	 acc:  0.66
15520 : loss:  0.5888756 	 acc:  0.72
15560 : loss:  0.76557493 	 acc:  0.64
15600 : loss:  0.5969146 	 acc:  0.72
15640 : loss:  0.66841155 	 acc:  0.67
15680 : loss:  0.69028497 	 acc:  0.7
15720 : loss:  0.61758417 	 acc:  0.78
15760 : loss:  0.71380186 	 acc:  0.73
15800 : loss:  0.69334334 	 acc:  0.69
15840 : loss:  0.7528825 	 acc:  0.65
15880 : loss:  0.74027956 	 acc:  0.62
15920 : loss:  0.6800382 	 acc:  0.71
15960 : loss:  0.65591913 	 acc:  0.69
16000 : loss:  0.6752115 	 acc:  0.69
16040 : loss:  0.62476563 	 acc:  0.74
16080 : loss:  0.7153858 	 acc:  0.67
16120 : loss:  0.73355436 	 acc:  0.66
16160 : loss:  0.62648696 	 acc:  0.73
16200 : loss:  0.7187222 	 acc:  0.7
16240 : loss:  0.61821985 	 acc:  0.74
16280 : loss:  0.7069206 	 acc:  0.7
16320 : loss:  0.5850761 	 acc:  0.74
16360 : loss:  0.6885357 	 acc:  0.7
16400 : loss:  0.74847865 	 acc:  0.67
16440 : loss:  0.7554164 	 acc:  0.61
16480 : loss:  0.5837406 	 acc:  0.75
16520 : loss:  0.6323654 	 acc:  0.74
16560 : loss:  0.6698398 	 acc:  0.67
16600 : loss:  0.66550505 	 acc:  0.72
16640 : loss:  0.6710305 	 acc:  0.73
16680 : loss:  0.6629225 	 acc:  0.71
16720 : loss:  0.657267 	 acc:  0.72
16760 : loss:  0.7522337 	 acc:  0.64
16800 : loss:  0.6409854 	 acc:  0.75
16840 : loss:  0.6128034 	 acc:  0.73
16880 : loss:  0.7038016 	 acc:  0.69
16920 : loss:  0.79401064 	 acc:  0.67
16960 : loss:  0.7544218 	 acc:  0.65
17000 : loss:  0.706683 	 acc:  0.68
17040 : loss:  0.7963673 	 acc:  0.61
17080 : loss:  0.6123153 	 acc:  0.73
17120 : loss:  0.7102155 	 acc:  0.66
17160 : loss:  0.7112474 	 acc:  0.73
17200 : loss:  0.62965083 	 acc:  0.76
17240 : loss:  0.61756694 	 acc:  0.74
17280 : loss:  0.6775583 	 acc:  0.71
17320 : loss:  0.63006216 	 acc:  0.71
17360 : loss:  0.70642143 	 acc:  0.67
17400 : loss:  0.63981706 	 acc:  0.75
17440 : loss:  0.7151108 	 acc:  0.72
17480 : loss:  0.75460815 	 acc:  0.67
17520 : loss:  0.94524926 	 acc:  0.58
17560 : loss:  0.7070485 	 acc:  0.7
17600 : loss:  0.6861396 	 acc:  0.69
17640 : loss:  0.7046812 	 acc:  0.67
17680 : loss:  0.7215356 	 acc:  0.73
17720 : loss:  0.72910136 	 acc:  0.7
17760 : loss:  0.57679266 	 acc:  0.75
17800 : loss:  0.6251717 	 acc:  0.73
17840 : loss:  0.6305515 	 acc:  0.75
17880 : loss:  0.6391563 	 acc:  0.72
17920 : loss:  0.68215567 	 acc:  0.68

Saving...
saved to models/CNN/pretrained_model.ckpt-17931

500 	 [372.53234617 224.8939116  388.57392663]
3 	val accuracy:  0.69114 	 f_! score:  [0.74506469 0.44978782 0.77714785]

500 	 [372.53234617 224.8939116  388.57392663]
500 	 [336.49643496 286.43076842 389.53689877]
500 	 [420.1430538  188.85309994 390.19203919]
500 	 [18794. 13823. 17383.]
--- Test   Review ---
0.69114
f1:  [0.74506469 0.44978782 0.77714785]
65730
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-10-14 19:26:34.043729: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-10-14 19:26:34.045202: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-14 19:26:34.045213: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-10-14 19:26:34.045218: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-10-14 19:26:34.046662: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2909 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
models/CNN/pretrained_model.ckpt-17931
2018-10-14 19:26:37.867808: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-10-14 19:26:37.867850: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-14 19:26:37.867856: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-10-14 19:26:37.867860: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-10-14 19:26:37.867944: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2909 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
acc:  0.33
/home/yannik/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
acc:  0.29
acc:  0.39
acc:  0.34
acc:  0.43
acc:  0.35
acc:  0.46
acc:  0.35
acc:  0.42
acc:  0.31
acc:  0.28
acc:  0.33
acc:  0.37
acc:  0.34
acc:  0.29
acc:  0.31
acc:  0.41
acc:  0.34
acc:  0.37
acc:  0.35
acc:  0.34
acc:  0.39
acc:  0.31
acc:  0.3
acc:  0.31
acc:  0.22
acc:  0.33
acc:  0.39
acc:  0.3
acc:  0.35
acc:  0.38
acc:  0.43
acc:  0.25
acc:  0.27
acc:  0.36
acc:  0.34
acc:  0.29
acc:  0.29
acc:  0.3
acc:  0.31
acc:  0.31
acc:  0.33
acc:  0.28
acc:  0.34
acc:  0.4
acc:  0.29
acc:  0.36
acc:  0.29
acc:  0.34
acc:  0.33
acc:  0.29
acc:  0.31
acc:  0.39
acc:  0.37
acc:  0.4
acc:  0.4
acc:  0.3
acc:  0.38
acc:  0.4
acc:  0.34
acc:  0.33
acc:  0.33
acc:  0.33
acc:  0.24
acc:  0.33
acc:  0.45
acc:  0.25
acc:  0.31
acc:  0.41
acc:  0.36
acc:  0.38
acc:  0.33
acc:  0.4
acc:  0.3
acc:  0.36
acc:  0.31
acc:  0.34
acc:  0.3
acc:  0.23
acc:  0.2
acc:  0.36
acc:  0.35
acc:  0.36
acc:  0.32
acc:  0.32
acc:  0.37
acc:  0.27
acc:  0.22
acc:  0.24
acc:  0.37
acc:  0.23
acc:  0.32
acc:  0.19
acc:  0.36
acc:  0.36
acc:  0.32
acc:  0.35
acc:  0.37
acc:  0.31
acc:  0.36

100 	 [ 0.         49.62620165  0.        ]
val accuracy:  0.3321 	 f_! score:  [0.         0.49626202 0.        ]

100 	 [ 0.         49.62620165  0.        ]
100 	 [ 0.         33.21888889  0.        ]
100 	 [ 0.         99.95833333  0.        ]
100 	 [3343. 3322. 3335.]
---just Test  Twitter ---
0.3321
f1:  [0.         0.49626202 0.        ]
--- 8.400211572647095 seconds ---
65730
Vorsicht! es muss vorher ein  CNN  Netz abgespeichert worden sein
2018-10-14 19:26:51.070672: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-10-14 19:26:51.070704: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-14 19:26:51.070711: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-10-14 19:26:51.070715: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-10-14 19:26:51.070797: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2909 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
models/CNN/pretrained_model.ckpt-17931
2018-10-14 19:26:53.905012: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-10-14 19:26:53.905054: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-14 19:26:53.905062: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-10-14 19:26:53.905068: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-10-14 19:26:53.905158: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2909 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
0 : loss:  1.089859 	 acc:  0.42
2 : loss:  1.1060786 	 acc:  0.23
4 : loss:  1.0924395 	 acc:  0.43
6 : loss:  1.092196 	 acc:  0.35
8 : loss:  1.1172488 	 acc:  0.3
10 : loss:  1.0840017 	 acc:  0.4
12 : loss:  1.0933877 	 acc:  0.39
14 : loss:  1.0945647 	 acc:  0.35
16 : loss:  1.0900487 	 acc:  0.28
18 : loss:  1.1086737 	 acc:  0.3
20 : loss:  1.1001229 	 acc:  0.35
22 : loss:  1.0861481 	 acc:  0.33
24 : loss:  1.0733788 	 acc:  0.38
26 : loss:  1.0912659 	 acc:  0.39
28 : loss:  1.066709 	 acc:  0.51
30 : loss:  1.0870322 	 acc:  0.39
32 : loss:  1.0809444 	 acc:  0.38
34 : loss:  1.0738841 	 acc:  0.48
36 : loss:  1.0945053 	 acc:  0.37
38 : loss:  1.0787501 	 acc:  0.39
40 : loss:  1.071639 	 acc:  0.35
42 : loss:  1.1013682 	 acc:  0.33
44 : loss:  1.0433478 	 acc:  0.53
46 : loss:  1.062536 	 acc:  0.44
48 : loss:  1.0853505 	 acc:  0.34
50 : loss:  1.0774864 	 acc:  0.44
52 : loss:  1.061495 	 acc:  0.45
54 : loss:  1.1080946 	 acc:  0.36
56 : loss:  1.0839818 	 acc:  0.45
58 : loss:  1.0719376 	 acc:  0.44
60 : loss:  1.0781959 	 acc:  0.37
62 : loss:  1.0562053 	 acc:  0.43
64 : loss:  1.0631043 	 acc:  0.4
66 : loss:  1.0814729 	 acc:  0.37
68 : loss:  1.0544862 	 acc:  0.44
70 : loss:  1.0895156 	 acc:  0.34
72 : loss:  1.0735245 	 acc:  0.42
74 : loss:  1.1095378 	 acc:  0.32
76 : loss:  1.0879575 	 acc:  0.38
78 : loss:  1.0718966 	 acc:  0.41
80 : loss:  1.0531392 	 acc:  0.41
82 : loss:  1.0228221 	 acc:  0.5
84 : loss:  1.0910304 	 acc:  0.34
86 : loss:  1.0706767 	 acc:  0.42
88 : loss:  1.0653921 	 acc:  0.42
90 : loss:  1.0344425 	 acc:  0.41
92 : loss:  1.09034 	 acc:  0.33
94 : loss:  1.0687186 	 acc:  0.42
96 : loss:  1.0478987 	 acc:  0.46
98 : loss:  1.081538 	 acc:  0.43
100 : loss:  1.0683441 	 acc:  0.46
102 : loss:  1.0520395 	 acc:  0.45
104 : loss:  1.0695606 	 acc:  0.43
106 : loss:  1.0413487 	 acc:  0.45
108 : loss:  1.0577852 	 acc:  0.44
110 : loss:  1.0750926 	 acc:  0.44
112 : loss:  1.0400639 	 acc:  0.46
114 : loss:  1.1005188 	 acc:  0.34
116 : loss:  1.0763689 	 acc:  0.47
118 : loss:  1.0526233 	 acc:  0.4
120 : loss:  1.0632023 	 acc:  0.39
122 : loss:  1.056345 	 acc:  0.43
124 : loss:  1.0862982 	 acc:  0.38
126 : loss:  1.1457248 	 acc:  0.27
128 : loss:  1.0964212 	 acc:  0.39
130 : loss:  1.057711 	 acc:  0.41
132 : loss:  1.0221231 	 acc:  0.46
134 : loss:  1.059683 	 acc:  0.43
136 : loss:  0.99703264 	 acc:  0.49
138 : loss:  1.0925562 	 acc:  0.35
140 : loss:  1.0522046 	 acc:  0.43
2018-10-14 19:27:16.268347: W tensorflow/core/common_runtime/bfc_allocator.cc:219] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.31GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
142 : loss:  0.9924883 	 acc:  0.6666667
144 : loss:  1.0513785 	 acc:  0.44
146 : loss:  1.061008 	 acc:  0.42
148 : loss:  1.070945 	 acc:  0.4
150 : loss:  1.0816894 	 acc:  0.4
152 : loss:  1.0573564 	 acc:  0.45
154 : loss:  1.0449755 	 acc:  0.48
156 : loss:  1.0432019 	 acc:  0.54
158 : loss:  1.0782592 	 acc:  0.39
160 : loss:  1.0475168 	 acc:  0.44
162 : loss:  1.0494034 	 acc:  0.46
164 : loss:  1.0374691 	 acc:  0.48
166 : loss:  1.0388832 	 acc:  0.45
168 : loss:  1.071726 	 acc:  0.38
170 : loss:  1.0460668 	 acc:  0.42
172 : loss:  1.0533262 	 acc:  0.38
174 : loss:  1.0566928 	 acc:  0.44
176 : loss:  1.0541725 	 acc:  0.41
178 : loss:  1.0665528 	 acc:  0.47
180 : loss:  1.0639331 	 acc:  0.44
182 : loss:  1.0530903 	 acc:  0.47
184 : loss:  1.0951662 	 acc:  0.38
186 : loss:  1.0572318 	 acc:  0.41
188 : loss:  1.067039 	 acc:  0.43
190 : loss:  1.0375379 	 acc:  0.47
192 : loss:  1.0674893 	 acc:  0.38
194 : loss:  1.0943787 	 acc:  0.36
196 : loss:  1.0782378 	 acc:  0.37
198 : loss:  1.0266078 	 acc:  0.5
200 : loss:  1.0384041 	 acc:  0.41
202 : loss:  1.09366 	 acc:  0.34
204 : loss:  1.0593562 	 acc:  0.5
206 : loss:  1.099394 	 acc:  0.31
208 : loss:  1.0842109 	 acc:  0.45
210 : loss:  1.0552977 	 acc:  0.49
212 : loss:  1.068651 	 acc:  0.36
214 : loss:  1.0338515 	 acc:  0.47
216 : loss:  1.0552704 	 acc:  0.48
218 : loss:  1.0741824 	 acc:  0.4
220 : loss:  1.0605415 	 acc:  0.39
222 : loss:  1.0065285 	 acc:  0.48
224 : loss:  1.0926747 	 acc:  0.36
226 : loss:  1.0028422 	 acc:  0.55
228 : loss:  1.1048841 	 acc:  0.37
230 : loss:  1.061688 	 acc:  0.42
232 : loss:  1.0299747 	 acc:  0.44
234 : loss:  1.0595056 	 acc:  0.38
236 : loss:  0.98974746 	 acc:  0.51
238 : loss:  1.0140448 	 acc:  0.51
240 : loss:  1.0756906 	 acc:  0.42
242 : loss:  1.0506743 	 acc:  0.47
244 : loss:  1.0527155 	 acc:  0.4
246 : loss:  1.055978 	 acc:  0.44
248 : loss:  1.087066 	 acc:  0.41
250 : loss:  1.0364891 	 acc:  0.46
252 : loss:  1.0254886 	 acc:  0.46
254 : loss:  1.024097 	 acc:  0.48
256 : loss:  1.1126593 	 acc:  0.31
258 : loss:  1.0484852 	 acc:  0.42
260 : loss:  1.0221213 	 acc:  0.44
262 : loss:  1.0364205 	 acc:  0.38
264 : loss:  1.0274049 	 acc:  0.5
266 : loss:  1.0242994 	 acc:  0.49
268 : loss:  1.0662262 	 acc:  0.38
270 : loss:  1.0821761 	 acc:  0.35
272 : loss:  1.0382782 	 acc:  0.44
274 : loss:  1.0043895 	 acc:  0.47
276 : loss:  1.0776465 	 acc:  0.45
278 : loss:  1.043077 	 acc:  0.41
280 : loss:  1.0265495 	 acc:  0.48
282 : loss:  1.0211723 	 acc:  0.49
284 : loss:  1.0459024 	 acc:  0.44
286 : loss:  1.0636201 	 acc:  0.43
288 : loss:  1.0764204 	 acc:  0.41
290 : loss:  1.0335853 	 acc:  0.47
292 : loss:  1.0119923 	 acc:  0.51
294 : loss:  1.0050277 	 acc:  0.5
296 : loss:  1.0703255 	 acc:  0.4
298 : loss:  1.0520663 	 acc:  0.41
300 : loss:  1.0276775 	 acc:  0.5
302 : loss:  1.0847973 	 acc:  0.37
304 : loss:  1.0204164 	 acc:  0.46
306 : loss:  1.0533749 	 acc:  0.41
308 : loss:  1.0317404 	 acc:  0.54
310 : loss:  1.0464647 	 acc:  0.44
312 : loss:  1.0547132 	 acc:  0.4
314 : loss:  0.9944858 	 acc:  0.5
316 : loss:  1.054289 	 acc:  0.36
318 : loss:  1.0423466 	 acc:  0.49
320 : loss:  1.0178356 	 acc:  0.48
322 : loss:  1.0486661 	 acc:  0.46
324 : loss:  1.0429599 	 acc:  0.44
326 : loss:  1.0731896 	 acc:  0.34
328 : loss:  0.99076957 	 acc:  0.53
330 : loss:  1.0774249 	 acc:  0.38
332 : loss:  1.0721316 	 acc:  0.41
334 : loss:  1.0323164 	 acc:  0.46
336 : loss:  1.0338659 	 acc:  0.45
338 : loss:  1.0326461 	 acc:  0.45
340 : loss:  1.0309106 	 acc:  0.49
342 : loss:  1.0389472 	 acc:  0.46
344 : loss:  1.0262069 	 acc:  0.39
346 : loss:  1.0505332 	 acc:  0.45
348 : loss:  1.050459 	 acc:  0.41
350 : loss:  1.1169543 	 acc:  0.38
352 : loss:  1.0192658 	 acc:  0.51
354 : loss:  1.0276622 	 acc:  0.46
356 : loss:  1.032551 	 acc:  0.48
358 : loss:  1.0256726 	 acc:  0.5
360 : loss:  1.0305552 	 acc:  0.46
362 : loss:  1.0258794 	 acc:  0.43
364 : loss:  1.0671296 	 acc:  0.38
366 : loss:  1.0587113 	 acc:  0.41
368 : loss:  1.0705239 	 acc:  0.37
370 : loss:  1.0272006 	 acc:  0.43
acc:  0.5
acc:  0.44
acc:  0.46
acc:  0.4
acc:  0.51
acc:  0.43
acc:  0.51
acc:  0.47
acc:  0.4
acc:  0.45
acc:  0.43
acc:  0.4
acc:  0.42
acc:  0.54
acc:  0.5
acc:  0.44
acc:  0.39
acc:  0.46
acc:  0.43
acc:  0.52
acc:  0.43
acc:  0.47
acc:  0.48
acc:  0.46
acc:  0.46
acc:  0.49
acc:  0.48
acc:  0.44
acc:  0.43
acc:  0.42
acc:  0.43
acc:  0.47
acc:  0.49
acc:  0.46
acc:  0.43
acc:  0.42
acc:  0.49
acc:  0.49
acc:  0.42
acc:  0.54
acc:  0.43
acc:  0.41
acc:  0.37
acc:  0.46
acc:  0.41
acc:  0.42
acc:  0.54
acc:  0.48
acc:  0.45
acc:  0.43
acc:  0.38
acc:  0.6
acc:  0.45
acc:  0.45
acc:  0.45
acc:  0.43
acc:  0.38
acc:  0.39
acc:  0.42
acc:  0.46
acc:  0.43
acc:  0.5
acc:  0.43
acc:  0.47
acc:  0.47
acc:  0.37
acc:  0.41
acc:  0.49
acc:  0.5
acc:  0.46
acc:  0.49
acc:  0.49
acc:  0.48
acc:  0.53
acc:  0.42
acc:  0.44
acc:  0.51
acc:  0.52
acc:  0.43
acc:  0.49
acc:  0.43
acc:  0.41
acc:  0.47
acc:  0.46
acc:  0.52
acc:  0.44
acc:  0.42
acc:  0.44
acc:  0.47
acc:  0.4
acc:  0.44
acc:  0.51
acc:  0.37
acc:  0.47
acc:  0.5
acc:  0.49
acc:  0.5
acc:  0.45
acc:  0.44
acc:  0.41

100 	 [51.6525615  49.97753226 28.67024778]
0 	val accuracy:  0.45479995 	 f_! score:  [0.51652561 0.49977532 0.28670248]

372 : loss:  1.0702561 	 acc:  0.4
374 : loss:  1.040064 	 acc:  0.45
376 : loss:  1.0711963 	 acc:  0.38
378 : loss:  1.071498 	 acc:  0.4
380 : loss:  1.066772 	 acc:  0.4
382 : loss:  1.0302888 	 acc:  0.42
384 : loss:  1.0480593 	 acc:  0.39
386 : loss:  1.0274235 	 acc:  0.44
388 : loss:  1.0570209 	 acc:  0.47
390 : loss:  1.0562593 	 acc:  0.37
392 : loss:  1.1009166 	 acc:  0.36
394 : loss:  1.1035346 	 acc:  0.41
396 : loss:  1.0773249 	 acc:  0.39
398 : loss:  1.0482248 	 acc:  0.37
400 : loss:  1.0676805 	 acc:  0.4
402 : loss:  1.0320945 	 acc:  0.45
404 : loss:  1.0665519 	 acc:  0.39
406 : loss:  1.0658166 	 acc:  0.35
408 : loss:  1.0420257 	 acc:  0.45
410 : loss:  1.0011591 	 acc:  0.47
412 : loss:  1.0251036 	 acc:  0.44
414 : loss:  1.0497848 	 acc:  0.37
416 : loss:  1.0583752 	 acc:  0.44
418 : loss:  1.0140852 	 acc:  0.49
420 : loss:  1.0433669 	 acc:  0.46
422 : loss:  1.0771792 	 acc:  0.38
424 : loss:  1.0192809 	 acc:  0.48
426 : loss:  1.1140721 	 acc:  0.37
428 : loss:  1.0464778 	 acc:  0.44
430 : loss:  1.0459286 	 acc:  0.42
432 : loss:  1.0863711 	 acc:  0.44
434 : loss:  1.1206642 	 acc:  0.34
436 : loss:  1.0066143 	 acc:  0.48
438 : loss:  1.0489628 	 acc:  0.44
440 : loss:  0.9778139 	 acc:  0.52
442 : loss:  1.0260692 	 acc:  0.46
444 : loss:  1.0208262 	 acc:  0.41
446 : loss:  1.0684494 	 acc:  0.44
448 : loss:  1.0432405 	 acc:  0.51
450 : loss:  1.0623697 	 acc:  0.41
452 : loss:  1.0294785 	 acc:  0.45
454 : loss:  1.0650203 	 acc:  0.44
456 : loss:  1.0533051 	 acc:  0.44
458 : loss:  1.041422 	 acc:  0.42
460 : loss:  1.0692242 	 acc:  0.36
462 : loss:  1.035281 	 acc:  0.46
464 : loss:  1.0653304 	 acc:  0.41
466 : loss:  1.0612352 	 acc:  0.44
468 : loss:  1.0121962 	 acc:  0.52
470 : loss:  1.0480378 	 acc:  0.45
472 : loss:  1.0751299 	 acc:  0.42
474 : loss:  1.038987 	 acc:  0.48
476 : loss:  1.0919944 	 acc:  0.4
478 : loss:  1.0956055 	 acc:  0.41
480 : loss:  1.0830526 	 acc:  0.31
482 : loss:  0.99907357 	 acc:  0.36666667
484 : loss:  1.0410999 	 acc:  0.49
486 : loss:  1.0218366 	 acc:  0.5
488 : loss:  1.0194215 	 acc:  0.47
490 : loss:  1.0069484 	 acc:  0.49
492 : loss:  1.0483719 	 acc:  0.44
494 : loss:  1.0635508 	 acc:  0.41
496 : loss:  0.9679685 	 acc:  0.56
498 : loss:  1.0603795 	 acc:  0.4
500 : loss:  1.0393754 	 acc:  0.43
502 : loss:  1.059642 	 acc:  0.44
504 : loss:  1.0325509 	 acc:  0.49
506 : loss:  1.0451839 	 acc:  0.42
508 : loss:  1.0600426 	 acc:  0.36
510 : loss:  1.0445534 	 acc:  0.41
512 : loss:  1.0186546 	 acc:  0.47
514 : loss:  1.0458028 	 acc:  0.43
516 : loss:  1.0519874 	 acc:  0.48
518 : loss:  1.0671581 	 acc:  0.4
520 : loss:  1.0282402 	 acc:  0.46
522 : loss:  1.0441217 	 acc:  0.43
524 : loss:  1.0366939 	 acc:  0.47
526 : loss:  1.0221395 	 acc:  0.51
528 : loss:  1.0406438 	 acc:  0.49
530 : loss:  1.041426 	 acc:  0.44
532 : loss:  1.1098319 	 acc:  0.31
534 : loss:  1.0123798 	 acc:  0.57
536 : loss:  1.0249193 	 acc:  0.43
538 : loss:  1.0513183 	 acc:  0.42
540 : loss:  1.0263072 	 acc:  0.46
542 : loss:  1.062977 	 acc:  0.43
544 : loss:  1.049079 	 acc:  0.45
546 : loss:  1.0206432 	 acc:  0.47
548 : loss:  1.0742289 	 acc:  0.4
550 : loss:  1.0026193 	 acc:  0.46
552 : loss:  1.0514675 	 acc:  0.41
554 : loss:  1.0357261 	 acc:  0.49
556 : loss:  1.0024283 	 acc:  0.44
558 : loss:  1.014193 	 acc:  0.52
560 : loss:  1.0280894 	 acc:  0.43
562 : loss:  1.0059619 	 acc:  0.48
564 : loss:  1.0545874 	 acc:  0.46
566 : loss:  1.0270935 	 acc:  0.47
568 : loss:  1.039213 	 acc:  0.38
570 : loss:  1.0499386 	 acc:  0.39
572 : loss:  1.013516 	 acc:  0.44
574 : loss:  1.0123456 	 acc:  0.51
576 : loss:  1.0232952 	 acc:  0.47
578 : loss:  1.0167978 	 acc:  0.51
580 : loss:  1.0505502 	 acc:  0.43
582 : loss:  1.057046 	 acc:  0.46
584 : loss:  1.0256449 	 acc:  0.53
586 : loss:  1.0182596 	 acc:  0.44
588 : loss:  1.0396024 	 acc:  0.45
590 : loss:  1.0092641 	 acc:  0.54
592 : loss:  1.0698153 	 acc:  0.39
594 : loss:  1.0415719 	 acc:  0.43
596 : loss:  0.9865316 	 acc:  0.54
598 : loss:  1.0672523 	 acc:  0.49
600 : loss:  1.0378299 	 acc:  0.43
602 : loss:  1.0259072 	 acc:  0.47
604 : loss:  1.0469661 	 acc:  0.43
606 : loss:  1.0500547 	 acc:  0.41
608 : loss:  1.0265974 	 acc:  0.48
610 : loss:  1.0254085 	 acc:  0.48
612 : loss:  1.0698595 	 acc:  0.41
614 : loss:  0.99495727 	 acc:  0.5
616 : loss:  1.0471516 	 acc:  0.42
618 : loss:  1.0184116 	 acc:  0.47
620 : loss:  1.0540845 	 acc:  0.39
622 : loss:  1.0456933 	 acc:  0.45
624 : loss:  1.1245749 	 acc:  0.34
626 : loss:  1.0908953 	 acc:  0.38
628 : loss:  1.0300603 	 acc:  0.45
630 : loss:  1.0342959 	 acc:  0.47
632 : loss:  1.0445094 	 acc:  0.45
634 : loss:  1.0515648 	 acc:  0.39
636 : loss:  1.0581299 	 acc:  0.41
638 : loss:  1.052989 	 acc:  0.47
640 : loss:  1.0766987 	 acc:  0.41
642 : loss:  0.97237307 	 acc:  0.48
644 : loss:  1.020784 	 acc:  0.47
646 : loss:  0.97978896 	 acc:  0.5
648 : loss:  1.0527819 	 acc:  0.49
650 : loss:  0.949235 	 acc:  0.51
652 : loss:  0.9810132 	 acc:  0.46
654 : loss:  1.0909873 	 acc:  0.42
656 : loss:  1.0045197 	 acc:  0.46
658 : loss:  1.078733 	 acc:  0.42
660 : loss:  1.0274321 	 acc:  0.44
662 : loss:  1.0764371 	 acc:  0.38
664 : loss:  1.0628636 	 acc:  0.43
666 : loss:  0.9812014 	 acc:  0.48
668 : loss:  1.076681 	 acc:  0.37
670 : loss:  1.029321 	 acc:  0.49
672 : loss:  0.9897591 	 acc:  0.49
674 : loss:  1.015063 	 acc:  0.43
676 : loss:  1.073334 	 acc:  0.39
678 : loss:  1.0800576 	 acc:  0.32
680 : loss:  1.0392472 	 acc:  0.47
682 : loss:  1.0464019 	 acc:  0.44
684 : loss:  1.0095168 	 acc:  0.53
686 : loss:  1.0079821 	 acc:  0.48
688 : loss:  1.0271506 	 acc:  0.44
690 : loss:  1.0230714 	 acc:  0.47
692 : loss:  1.073367 	 acc:  0.41
694 : loss:  1.0450417 	 acc:  0.45
696 : loss:  1.0849135 	 acc:  0.41
698 : loss:  1.0011226 	 acc:  0.54
700 : loss:  0.9796385 	 acc:  0.52
702 : loss:  1.0259347 	 acc:  0.49
704 : loss:  1.0512934 	 acc:  0.46
706 : loss:  0.97377425 	 acc:  0.53
708 : loss:  0.9661 	 acc:  0.52
710 : loss:  1.0423329 	 acc:  0.43
712 : loss:  1.032736 	 acc:  0.47
714 : loss:  1.0652072 	 acc:  0.45
716 : loss:  1.034334 	 acc:  0.44
718 : loss:  1.0857884 	 acc:  0.35
720 : loss:  1.0362588 	 acc:  0.46
722 : loss:  0.98672193 	 acc:  0.53
724 : loss:  1.0186837 	 acc:  0.43
726 : loss:  1.0395503 	 acc:  0.48
728 : loss:  1.0249857 	 acc:  0.5
730 : loss:  0.9989752 	 acc:  0.51
732 : loss:  1.0440509 	 acc:  0.47
734 : loss:  1.0260731 	 acc:  0.39
736 : loss:  1.0497713 	 acc:  0.41
738 : loss:  0.99060816 	 acc:  0.47
740 : loss:  1.0775511 	 acc:  0.46
742 : loss:  1.0340967 	 acc:  0.45
acc:  0.43
acc:  0.45
acc:  0.46
acc:  0.39
acc:  0.45
acc:  0.46
acc:  0.45
acc:  0.51
acc:  0.47
acc:  0.46
acc:  0.53
acc:  0.43
acc:  0.5
acc:  0.41
acc:  0.47
acc:  0.46
acc:  0.48
acc:  0.37
acc:  0.43
acc:  0.44
acc:  0.38
acc:  0.47
acc:  0.48
acc:  0.43
acc:  0.41
acc:  0.53
acc:  0.49
acc:  0.55
acc:  0.44
acc:  0.48
acc:  0.48
acc:  0.52
acc:  0.45
acc:  0.54
acc:  0.45
acc:  0.53
acc:  0.43
acc:  0.41
acc:  0.41
acc:  0.44
acc:  0.5
acc:  0.45
acc:  0.47
acc:  0.47
acc:  0.45
acc:  0.43
acc:  0.42
acc:  0.58
acc:  0.46
acc:  0.48
acc:  0.46
acc:  0.45
acc:  0.48
acc:  0.43
acc:  0.39
acc:  0.55
acc:  0.41
acc:  0.52
acc:  0.43
acc:  0.46
acc:  0.46
acc:  0.49
acc:  0.44
acc:  0.47
acc:  0.48
acc:  0.5
acc:  0.47
acc:  0.41
acc:  0.42
acc:  0.45
acc:  0.43
acc:  0.48
acc:  0.49
acc:  0.45
acc:  0.42
acc:  0.51
acc:  0.48
acc:  0.47
acc:  0.47
acc:  0.47
acc:  0.47
acc:  0.48
acc:  0.44
acc:  0.51
acc:  0.43
acc:  0.53
acc:  0.51
acc:  0.52
acc:  0.47
acc:  0.44
acc:  0.4
acc:  0.49
acc:  0.42
acc:  0.43
acc:  0.46
acc:  0.52
acc:  0.46
acc:  0.48
acc:  0.49
acc:  0.45

100 	 [51.50567025 53.96128407 21.78964445]
1 	val accuracy:  0.4632 	 f_! score:  [0.5150567  0.53961284 0.21789644]

744 : loss:  1.0292327 	 acc:  0.45
746 : loss:  1.0535022 	 acc:  0.45
748 : loss:  0.98960197 	 acc:  0.48
750 : loss:  1.0370245 	 acc:  0.42
752 : loss:  1.0111371 	 acc:  0.49
754 : loss:  1.0408146 	 acc:  0.47
756 : loss:  1.0429317 	 acc:  0.47
758 : loss:  1.0409747 	 acc:  0.53
760 : loss:  1.0618318 	 acc:  0.39
762 : loss:  0.97407067 	 acc:  0.52
764 : loss:  1.1027818 	 acc:  0.41
766 : loss:  1.0123458 	 acc:  0.45
768 : loss:  1.0300149 	 acc:  0.41
770 : loss:  1.1320382 	 acc:  0.33
772 : loss:  0.9794719 	 acc:  0.55
774 : loss:  1.0573543 	 acc:  0.43
776 : loss:  1.0054182 	 acc:  0.52
778 : loss:  1.0265843 	 acc:  0.42
780 : loss:  1.0655731 	 acc:  0.41
782 : loss:  0.98692054 	 acc:  0.51
784 : loss:  0.9891202 	 acc:  0.44
786 : loss:  1.0935674 	 acc:  0.39
788 : loss:  0.98498267 	 acc:  0.5
790 : loss:  1.0189701 	 acc:  0.46
792 : loss:  1.0200573 	 acc:  0.49
794 : loss:  0.96131825 	 acc:  0.53
796 : loss:  1.0438128 	 acc:  0.47
798 : loss:  1.1381291 	 acc:  0.33
800 : loss:  0.9935308 	 acc:  0.48
802 : loss:  1.1150861 	 acc:  0.37
804 : loss:  1.0277461 	 acc:  0.46
806 : loss:  1.0338776 	 acc:  0.46
808 : loss:  1.0329812 	 acc:  0.38
810 : loss:  1.0167744 	 acc:  0.47
812 : loss:  1.0446321 	 acc:  0.44
814 : loss:  1.0901947 	 acc:  0.38
816 : loss:  1.0156941 	 acc:  0.49
818 : loss:  0.9949452 	 acc:  0.52
820 : loss:  1.0362514 	 acc:  0.46
822 : loss:  0.985435 	 acc:  0.53
824 : loss:  1.0322026 	 acc:  0.45
826 : loss:  1.0470672 	 acc:  0.36
828 : loss:  1.02154 	 acc:  0.46
830 : loss:  1.0333196 	 acc:  0.48
832 : loss:  1.0440364 	 acc:  0.43
834 : loss:  1.0518491 	 acc:  0.43
836 : loss:  1.0251806 	 acc:  0.51
838 : loss:  0.9998125 	 acc:  0.48
840 : loss:  1.037148 	 acc:  0.51
842 : loss:  1.0117654 	 acc:  0.48
844 : loss:  1.0752742 	 acc:  0.44
846 : loss:  1.0594658 	 acc:  0.4
848 : loss:  1.0254124 	 acc:  0.47
850 : loss:  1.0112767 	 acc:  0.48
852 : loss:  0.98979455 	 acc:  0.46
854 : loss:  1.0123562 	 acc:  0.51
856 : loss:  1.0564961 	 acc:  0.43
858 : loss:  1.0039536 	 acc:  0.56
860 : loss:  1.0283185 	 acc:  0.47
862 : loss:  1.0292728 	 acc:  0.47
864 : loss:  1.0336144 	 acc:  0.42
866 : loss:  1.0452858 	 acc:  0.5
868 : loss:  1.0221484 	 acc:  0.43
870 : loss:  1.0536474 	 acc:  0.45
872 : loss:  1.0492615 	 acc:  0.5
874 : loss:  0.9886959 	 acc:  0.49
876 : loss:  1.0137063 	 acc:  0.47
878 : loss:  1.0175861 	 acc:  0.5
880 : loss:  1.0404862 	 acc:  0.45
882 : loss:  1.0275817 	 acc:  0.45
884 : loss:  1.0327567 	 acc:  0.47
886 : loss:  0.9701292 	 acc:  0.48
888 : loss:  1.1060765 	 acc:  0.43
890 : loss:  1.0942235 	 acc:  0.45
892 : loss:  1.0047607 	 acc:  0.47
894 : loss:  1.0353768 	 acc:  0.47
896 : loss:  0.97203183 	 acc:  0.55
898 : loss:  1.0222352 	 acc:  0.53
900 : loss:  1.0479659 	 acc:  0.45
902 : loss:  1.0661477 	 acc:  0.42
904 : loss:  1.018891 	 acc:  0.5
906 : loss:  1.0489166 	 acc:  0.45
908 : loss:  1.0350512 	 acc:  0.47
910 : loss:  1.0384401 	 acc:  0.45
912 : loss:  1.0342649 	 acc:  0.47
914 : loss:  1.0229957 	 acc:  0.47
916 : loss:  0.9960001 	 acc:  0.51
918 : loss:  1.0319079 	 acc:  0.46
920 : loss:  1.0405948 	 acc:  0.46
922 : loss:  1.002497 	 acc:  0.53
924 : loss:  1.074379 	 acc:  0.42
926 : loss:  0.98907816 	 acc:  0.49
928 : loss:  1.0543892 	 acc:  0.44
930 : loss:  1.0164177 	 acc:  0.47
932 : loss:  1.0125355 	 acc:  0.46
934 : loss:  1.010574 	 acc:  0.51
936 : loss:  1.0941842 	 acc:  0.37
938 : loss:  0.9904164 	 acc:  0.51
940 : loss:  1.0465245 	 acc:  0.47
942 : loss:  1.0398681 	 acc:  0.45
944 : loss:  0.9860342 	 acc:  0.49
946 : loss:  1.0301822 	 acc:  0.35
948 : loss:  1.023143 	 acc:  0.44
950 : loss:  1.0936049 	 acc:  0.43
952 : loss:  1.0489382 	 acc:  0.47
954 : loss:  0.9959678 	 acc:  0.48
956 : loss:  0.9832878 	 acc:  0.53
958 : loss:  1.0377511 	 acc:  0.49
960 : loss:  1.0253572 	 acc:  0.45
962 : loss:  1.0490109 	 acc:  0.52
964 : loss:  1.0442901 	 acc:  0.52
966 : loss:  1.0018834 	 acc:  0.5
968 : loss:  1.0836208 	 acc:  0.41
970 : loss:  1.0563222 	 acc:  0.44
972 : loss:  1.0052646 	 acc:  0.52
974 : loss:  0.9949802 	 acc:  0.49
976 : loss:  0.96844393 	 acc:  0.51
978 : loss:  0.99488837 	 acc:  0.47
980 : loss:  1.0466242 	 acc:  0.47
982 : loss:  1.0421748 	 acc:  0.45
984 : loss:  0.9956926 	 acc:  0.54
986 : loss:  1.0058246 	 acc:  0.47
988 : loss:  1.0492898 	 acc:  0.44
990 : loss:  1.0168091 	 acc:  0.48
992 : loss:  1.0671482 	 acc:  0.42
994 : loss:  1.0146894 	 acc:  0.47
996 : loss:  1.0530596 	 acc:  0.42
998 : loss:  1.0316164 	 acc:  0.45
1000 : loss:  1.0232694 	 acc:  0.45
1002 : loss:  0.98997855 	 acc:  0.53
1004 : loss:  0.9738036 	 acc:  0.53
1006 : loss:  1.063052 	 acc:  0.42
1008 : loss:  1.0669013 	 acc:  0.47
1010 : loss:  1.0188689 	 acc:  0.48
1012 : loss:  0.9868551 	 acc:  0.48
1014 : loss:  1.0087013 	 acc:  0.47
1016 : loss:  0.98971575 	 acc:  0.48
1018 : loss:  0.9962768 	 acc:  0.47
1020 : loss:  1.0343379 	 acc:  0.5
1022 : loss:  1.0248162 	 acc:  0.48
1024 : loss:  1.0824336 	 acc:  0.49
1026 : loss:  1.0431796 	 acc:  0.43
1028 : loss:  1.0026824 	 acc:  0.47
1030 : loss:  1.0297058 	 acc:  0.42
1032 : loss:  0.9358734 	 acc:  0.59
1034 : loss:  1.0671893 	 acc:  0.48
1036 : loss:  1.0408384 	 acc:  0.39
1038 : loss:  1.0119615 	 acc:  0.46
1040 : loss:  1.0043089 	 acc:  0.47
1042 : loss:  0.96841127 	 acc:  0.54
1044 : loss:  1.0990697 	 acc:  0.45
1046 : loss:  1.0421542 	 acc:  0.46
1048 : loss:  1.0332011 	 acc:  0.5
1050 : loss:  1.0254561 	 acc:  0.47
1052 : loss:  1.0435417 	 acc:  0.45
1054 : loss:  1.0233643 	 acc:  0.41
1056 : loss:  1.0330061 	 acc:  0.48
1058 : loss:  1.0202851 	 acc:  0.46
1060 : loss:  1.0126181 	 acc:  0.45
1062 : loss:  1.0286602 	 acc:  0.39
1064 : loss:  1.0470223 	 acc:  0.43
1066 : loss:  1.0277481 	 acc:  0.46
1068 : loss:  0.99438214 	 acc:  0.47
1070 : loss:  1.0656052 	 acc:  0.41
1072 : loss:  1.0049648 	 acc:  0.52
1074 : loss:  1.0951291 	 acc:  0.4
1076 : loss:  1.0459712 	 acc:  0.39
1078 : loss:  0.9666985 	 acc:  0.51
1080 : loss:  1.0465227 	 acc:  0.45
1082 : loss:  1.0147011 	 acc:  0.49
1084 : loss:  1.0272112 	 acc:  0.41
1086 : loss:  0.9461945 	 acc:  0.6
1088 : loss:  1.0143807 	 acc:  0.43
1090 : loss:  1.0671407 	 acc:  0.43
1092 : loss:  1.1124341 	 acc:  0.41
1094 : loss:  1.0533452 	 acc:  0.4
1096 : loss:  1.0300213 	 acc:  0.43
1098 : loss:  0.98696023 	 acc:  0.48
1100 : loss:  1.0455471 	 acc:  0.46
1102 : loss:  1.0149567 	 acc:  0.5
1104 : loss:  1.0540202 	 acc:  0.49
1106 : loss:  0.9873153 	 acc:  0.52
1108 : loss:  1.013295 	 acc:  0.44
1110 : loss:  0.9363825 	 acc:  0.63
1112 : loss:  1.0475509 	 acc:  0.48
1114 : loss:  1.0082473 	 acc:  0.48
acc:  0.49
acc:  0.47
acc:  0.45
acc:  0.49
acc:  0.44
acc:  0.51
acc:  0.41
acc:  0.49
acc:  0.48
acc:  0.51
acc:  0.43
acc:  0.51
acc:  0.54
acc:  0.47
acc:  0.43
acc:  0.51
acc:  0.44
acc:  0.49
acc:  0.4
acc:  0.45
acc:  0.43
acc:  0.41
acc:  0.43
acc:  0.37
acc:  0.51
acc:  0.43
acc:  0.43
acc:  0.44
acc:  0.47
acc:  0.45
acc:  0.46
acc:  0.43
acc:  0.5
acc:  0.43
acc:  0.42
acc:  0.45
acc:  0.48
acc:  0.48
acc:  0.61
acc:  0.46
acc:  0.47
acc:  0.49
acc:  0.47
acc:  0.59
acc:  0.49
acc:  0.49
acc:  0.48
acc:  0.47
acc:  0.46
acc:  0.49
acc:  0.5
acc:  0.45
acc:  0.44
acc:  0.47
acc:  0.48
acc:  0.43
acc:  0.46
acc:  0.4
acc:  0.45
acc:  0.45
acc:  0.47
acc:  0.53
acc:  0.43
acc:  0.41
acc:  0.45
acc:  0.48
acc:  0.43
acc:  0.49
acc:  0.45
acc:  0.51
acc:  0.49
acc:  0.48
acc:  0.47
acc:  0.48
acc:  0.55
acc:  0.48
acc:  0.47
acc:  0.53
acc:  0.44
acc:  0.53
acc:  0.45
acc:  0.41
acc:  0.5
acc:  0.51
acc:  0.49
acc:  0.45
acc:  0.52
acc:  0.49
acc:  0.43
acc:  0.46
acc:  0.48
acc:  0.5
acc:  0.5
acc:  0.47
acc:  0.47
acc:  0.51
acc:  0.47
acc:  0.5
acc:  0.42
acc:  0.44

100 	 [52.14410911 54.35897414 23.69908063]
2 	val accuracy:  0.4697 	 f_! score:  [0.52144109 0.54358974 0.23699081]

1116 : loss:  0.954119 	 acc:  0.51
1118 : loss:  1.0646018 	 acc:  0.44
1120 : loss:  0.9791683 	 acc:  0.51
1122 : loss:  0.9786094 	 acc:  0.48
1124 : loss:  1.0107341 	 acc:  0.44
1126 : loss:  0.9680129 	 acc:  0.5
1128 : loss:  1.0897421 	 acc:  0.4
1130 : loss:  1.0425212 	 acc:  0.4
1132 : loss:  1.0678065 	 acc:  0.37
1134 : loss:  1.0581566 	 acc:  0.43
1136 : loss:  1.0291001 	 acc:  0.44
1138 : loss:  1.0062139 	 acc:  0.47
1140 : loss:  1.0220314 	 acc:  0.45
1142 : loss:  1.0581996 	 acc:  0.39
1144 : loss:  1.0461087 	 acc:  0.46
1146 : loss:  0.98397917 	 acc:  0.47
1148 : loss:  1.049712 	 acc:  0.35
1150 : loss:  1.0103008 	 acc:  0.55
1152 : loss:  0.99862 	 acc:  0.45
1154 : loss:  0.99302465 	 acc:  0.53
1156 : loss:  1.0213944 	 acc:  0.5
1158 : loss:  1.0182822 	 acc:  0.41
1160 : loss:  1.0053195 	 acc:  0.44
1162 : loss:  1.0471379 	 acc:  0.44
1164 : loss:  1.0105723 	 acc:  0.44
1166 : loss:  0.9929556 	 acc:  0.46
1168 : loss:  1.0542917 	 acc:  0.38
1170 : loss:  1.0126892 	 acc:  0.47
1172 : loss:  1.0161282 	 acc:  0.47
1174 : loss:  0.95555085 	 acc:  0.6
1176 : loss:  1.0441022 	 acc:  0.43
1178 : loss:  0.9870916 	 acc:  0.58
1180 : loss:  1.036227 	 acc:  0.42
1182 : loss:  1.0575862 	 acc:  0.41
1184 : loss:  0.999764 	 acc:  0.51
1186 : loss:  0.9891886 	 acc:  0.46
1188 : loss:  1.0667965 	 acc:  0.51
1190 : loss:  1.0451893 	 acc:  0.46
1192 : loss:  1.0530047 	 acc:  0.41
1194 : loss:  1.044616 	 acc:  0.52
1196 : loss:  1.0311605 	 acc:  0.45
1198 : loss:  1.0252458 	 acc:  0.45
1200 : loss:  1.0206271 	 acc:  0.43
1202 : loss:  1.0456473 	 acc:  0.39
1204 : loss:  1.0229199 	 acc:  0.49
1206 : loss:  1.0362519 	 acc:  0.43
1208 : loss:  0.97136056 	 acc:  0.55
1210 : loss:  1.0064201 	 acc:  0.5
1212 : loss:  0.9809074 	 acc:  0.5
1214 : loss:  1.0094899 	 acc:  0.49
1216 : loss:  1.0383286 	 acc:  0.45
1218 : loss:  0.97865415 	 acc:  0.5
1220 : loss:  0.9614956 	 acc:  0.48
1222 : loss:  1.0125535 	 acc:  0.44
1224 : loss:  1.0617399 	 acc:  0.41
1226 : loss:  1.0202996 	 acc:  0.45
1228 : loss:  0.97105426 	 acc:  0.56
1230 : loss:  1.0183579 	 acc:  0.47
1232 : loss:  1.1082766 	 acc:  0.39
1234 : loss:  1.0650122 	 acc:  0.4
1236 : loss:  0.9872636 	 acc:  0.51
1238 : loss:  1.0243546 	 acc:  0.45
1240 : loss:  0.9890232 	 acc:  0.49
1242 : loss:  1.0504357 	 acc:  0.44
1244 : loss:  1.0363673 	 acc:  0.44
1246 : loss:  1.0765351 	 acc:  0.38
1248 : loss:  1.0044575 	 acc:  0.47
1250 : loss:  0.9994627 	 acc:  0.49
1252 : loss:  1.0032699 	 acc:  0.47
1254 : loss:  1.0122075 	 acc:  0.5
1256 : loss:  1.0243144 	 acc:  0.49
1258 : loss:  0.93072367 	 acc:  0.57
1260 : loss:  0.9817273 	 acc:  0.51
1262 : loss:  1.0262456 	 acc:  0.44
1264 : loss:  1.0201759 	 acc:  0.44
1266 : loss:  1.031516 	 acc:  0.48
1268 : loss:  1.0321053 	 acc:  0.52
1270 : loss:  1.0039725 	 acc:  0.48
1272 : loss:  0.9672399 	 acc:  0.56
1274 : loss:  1.0020832 	 acc:  0.47
1276 : loss:  1.0139679 	 acc:  0.5
1278 : loss:  1.0799996 	 acc:  0.4
1280 : loss:  1.0311264 	 acc:  0.48
1282 : loss:  1.0496984 	 acc:  0.46
1284 : loss:  1.0084145 	 acc:  0.55
1286 : loss:  1.0333073 	 acc:  0.45
1288 : loss:  0.98483413 	 acc:  0.45
1290 : loss:  1.0743222 	 acc:  0.38
1292 : loss:  1.0028092 	 acc:  0.42
1294 : loss:  1.0518702 	 acc:  0.44
1296 : loss:  1.030932 	 acc:  0.46
1298 : loss:  1.0598427 	 acc:  0.5
1300 : loss:  1.0316156 	 acc:  0.38
1302 : loss:  1.0301138 	 acc:  0.43
1304 : loss:  1.0202708 	 acc:  0.47
1306 : loss:  1.0278716 	 acc:  0.45
1308 : loss:  0.9817643 	 acc:  0.54
1310 : loss:  0.99298024 	 acc:  0.45
1312 : loss:  1.0596075 	 acc:  0.42
1314 : loss:  0.97944856 	 acc:  0.49
1316 : loss:  0.98672235 	 acc:  0.49
1318 : loss:  1.0153426 	 acc:  0.5
1320 : loss:  1.0340451 	 acc:  0.43
1322 : loss:  1.0515212 	 acc:  0.41
1324 : loss:  1.0297183 	 acc:  0.47
1326 : loss:  1.031012 	 acc:  0.43
1328 : loss:  1.0403119 	 acc:  0.49
1330 : loss:  1.037909 	 acc:  0.43
1332 : loss:  1.0550044 	 acc:  0.41
1334 : loss:  1.0536408 	 acc:  0.43
1336 : loss:  1.0492085 	 acc:  0.45
1338 : loss:  1.0671246 	 acc:  0.48
1340 : loss:  1.0211644 	 acc:  0.53
1342 : loss:  1.0830499 	 acc:  0.41
1344 : loss:  1.0762062 	 acc:  0.35
1346 : loss:  1.093521 	 acc:  0.44
1348 : loss:  1.0383677 	 acc:  0.39
1350 : loss:  1.0703145 	 acc:  0.47
1352 : loss:  1.0281658 	 acc:  0.42
1354 : loss:  0.97321624 	 acc:  0.57
1356 : loss:  0.98548704 	 acc:  0.48
1358 : loss:  1.069645 	 acc:  0.44
1360 : loss:  1.0339259 	 acc:  0.41
1362 : loss:  1.031735 	 acc:  0.48
1364 : loss:  1.0261976 	 acc:  0.44
1366 : loss:  0.97654456 	 acc:  0.5
1368 : loss:  1.0291662 	 acc:  0.44
1370 : loss:  1.0534006 	 acc:  0.44
1372 : loss:  1.018 	 acc:  0.46
1374 : loss:  1.0440302 	 acc:  0.42
1376 : loss:  1.0294191 	 acc:  0.39
1378 : loss:  1.0292684 	 acc:  0.53
1380 : loss:  0.93146354 	 acc:  0.6
1382 : loss:  1.0264578 	 acc:  0.54
1384 : loss:  1.0279771 	 acc:  0.43
1386 : loss:  0.98653305 	 acc:  0.47
1388 : loss:  1.0111494 	 acc:  0.48
1390 : loss:  1.0283942 	 acc:  0.51
1392 : loss:  0.9816858 	 acc:  0.47
1394 : loss:  1.0166072 	 acc:  0.51
1396 : loss:  1.0059806 	 acc:  0.49
1398 : loss:  1.1060017 	 acc:  0.4
1400 : loss:  0.97361207 	 acc:  0.55
1402 : loss:  0.9813898 	 acc:  0.55
1404 : loss:  0.96583825 	 acc:  0.56
1406 : loss:  1.0088077 	 acc:  0.53
1408 : loss:  1.033791 	 acc:  0.46
1410 : loss:  1.0506252 	 acc:  0.47
1412 : loss:  1.0468205 	 acc:  0.38
1414 : loss:  0.9081078 	 acc:  0.61
1416 : loss:  1.0252968 	 acc:  0.45
1418 : loss:  1.0375179 	 acc:  0.48
1420 : loss:  1.032921 	 acc:  0.4
1422 : loss:  1.0479487 	 acc:  0.44
1424 : loss:  1.0322278 	 acc:  0.44
1426 : loss:  1.0673522 	 acc:  0.43
1428 : loss:  1.0249048 	 acc:  0.47
1430 : loss:  0.994027 	 acc:  0.48
1432 : loss:  1.0173528 	 acc:  0.48
1434 : loss:  1.0096611 	 acc:  0.45
1436 : loss:  1.0194666 	 acc:  0.49
1438 : loss:  1.0270592 	 acc:  0.47
1440 : loss:  1.0660473 	 acc:  0.41
1442 : loss:  1.039815 	 acc:  0.47
1444 : loss:  0.9778954 	 acc:  0.44
1446 : loss:  1.0570184 	 acc:  0.38
1448 : loss:  1.0253816 	 acc:  0.47
1450 : loss:  0.9845681 	 acc:  0.57
1452 : loss:  1.0118124 	 acc:  0.53
1454 : loss:  1.029045 	 acc:  0.49
1456 : loss:  0.9670168 	 acc:  0.5
1458 : loss:  1.1104556 	 acc:  0.34
1460 : loss:  1.1081439 	 acc:  0.34
1462 : loss:  1.0143173 	 acc:  0.55
1464 : loss:  1.0227512 	 acc:  0.48
1466 : loss:  1.0936466 	 acc:  0.4
1468 : loss:  1.0365471 	 acc:  0.45
1470 : loss:  1.0381976 	 acc:  0.41
1472 : loss:  1.0620226 	 acc:  0.37
1474 : loss:  0.98686713 	 acc:  0.49
1476 : loss:  1.07797 	 acc:  0.45
1478 : loss:  1.0074682 	 acc:  0.45
1480 : loss:  0.99417716 	 acc:  0.44
1482 : loss:  0.9856265 	 acc:  0.46
1484 : loss:  0.98137444 	 acc:  0.47
1486 : loss:  1.0191951 	 acc:  0.49
acc:  0.5
acc:  0.49
acc:  0.52
acc:  0.49
acc:  0.51
acc:  0.54
acc:  0.48
acc:  0.53
acc:  0.4
acc:  0.55
acc:  0.44
acc:  0.46
acc:  0.49
acc:  0.54
acc:  0.48
acc:  0.53
acc:  0.39
acc:  0.48
acc:  0.43
acc:  0.47
acc:  0.46
acc:  0.46
acc:  0.47
acc:  0.45
acc:  0.46
acc:  0.56
acc:  0.54
acc:  0.42
acc:  0.51
acc:  0.49
acc:  0.53
acc:  0.49
acc:  0.5
acc:  0.45
acc:  0.56
acc:  0.51
acc:  0.51
acc:  0.46
acc:  0.53
acc:  0.42
acc:  0.64
acc:  0.44
acc:  0.47
acc:  0.46
acc:  0.39
acc:  0.47
acc:  0.47
acc:  0.53
acc:  0.55
acc:  0.47
acc:  0.44
acc:  0.48
acc:  0.5
acc:  0.51
acc:  0.38
acc:  0.55
acc:  0.43
acc:  0.6
acc:  0.44
acc:  0.53
acc:  0.44
acc:  0.44
acc:  0.5
acc:  0.53
acc:  0.56
acc:  0.45
acc:  0.5
acc:  0.47
acc:  0.53
acc:  0.52
acc:  0.47
acc:  0.46
acc:  0.42
acc:  0.5
acc:  0.57
acc:  0.51
acc:  0.46
acc:  0.41
acc:  0.53
acc:  0.47
acc:  0.54
acc:  0.47
acc:  0.41
acc:  0.5
acc:  0.53
acc:  0.38
acc:  0.48
acc:  0.57
acc:  0.42
acc:  0.45
acc:  0.51
acc:  0.47
acc:  0.44
acc:  0.44
acc:  0.44
acc:  0.52
acc:  0.43
acc:  0.48
acc:  0.43
acc:  0.45

100 	 [53.3970735  50.20851011 39.68114541]
3 	val accuracy:  0.48349997 	 f_! score:  [0.53397073 0.5020851  0.39681145]

1488 : loss:  1.0352658 	 acc:  0.44
1490 : loss:  1.0627402 	 acc:  0.44
1492 : loss:  1.0117428 	 acc:  0.5
1494 : loss:  1.0395705 	 acc:  0.4
1496 : loss:  1.0374521 	 acc:  0.46
1498 : loss:  0.978918 	 acc:  0.48
1500 : loss:  1.0301886 	 acc:  0.51
1502 : loss:  1.0343347 	 acc:  0.47
1504 : loss:  0.96189225 	 acc:  0.54
1506 : loss:  0.9616944 	 acc:  0.56
1508 : loss:  1.0264524 	 acc:  0.45
1510 : loss:  0.98669153 	 acc:  0.52
1512 : loss:  1.0542789 	 acc:  0.47
1514 : loss:  1.0219939 	 acc:  0.4
1516 : loss:  1.0390557 	 acc:  0.43
1518 : loss:  0.9822294 	 acc:  0.48
1520 : loss:  1.045601 	 acc:  0.46
1522 : loss:  1.0774775 	 acc:  0.41
1524 : loss:  0.9794878 	 acc:  0.44
1526 : loss:  0.97343343 	 acc:  0.53
1528 : loss:  0.96746874 	 acc:  0.52
1530 : loss:  1.0356833 	 acc:  0.46
1532 : loss:  0.9987491 	 acc:  0.47
1534 : loss:  1.0155587 	 acc:  0.44
1536 : loss:  1.007189 	 acc:  0.45
1538 : loss:  1.0154498 	 acc:  0.45
1540 : loss:  0.9543812 	 acc:  0.58
1542 : loss:  0.99763733 	 acc:  0.45
1544 : loss:  1.0055053 	 acc:  0.48
1546 : loss:  0.97155166 	 acc:  0.56
1548 : loss:  0.9868719 	 acc:  0.52
1550 : loss:  1.0012759 	 acc:  0.51
1552 : loss:  1.0352405 	 acc:  0.43
1554 : loss:  1.0200638 	 acc:  0.52
1556 : loss:  1.0638608 	 acc:  0.42
1558 : loss:  0.9928445 	 acc:  0.53
1560 : loss:  1.0619001 	 acc:  0.42
1562 : loss:  0.9927543 	 acc:  0.45
1564 : loss:  1.0392141 	 acc:  0.48
1566 : loss:  1.0383912 	 acc:  0.43
1568 : loss:  1.0043077 	 acc:  0.46
1570 : loss:  0.957891 	 acc:  0.53
1572 : loss:  1.0269986 	 acc:  0.48
1574 : loss:  0.93996894 	 acc:  0.59
1576 : loss:  1.0157173 	 acc:  0.48
1578 : loss:  0.9658818 	 acc:  0.53
1580 : loss:  1.0785859 	 acc:  0.4
1582 : loss:  1.0612675 	 acc:  0.36
1584 : loss:  1.0428169 	 acc:  0.4
1586 : loss:  1.0100515 	 acc:  0.45
1588 : loss:  0.9781949 	 acc:  0.5
1590 : loss:  1.0592135 	 acc:  0.5
1592 : loss:  0.9855445 	 acc:  0.48
1594 : loss:  1.0571188 	 acc:  0.43
1596 : loss:  0.99848974 	 acc:  0.48
1598 : loss:  1.0551454 	 acc:  0.45
1600 : loss:  1.0327225 	 acc:  0.39
1602 : loss:  1.0728681 	 acc:  0.38
1604 : loss:  1.0012882 	 acc:  0.48
1606 : loss:  0.9684544 	 acc:  0.5
1608 : loss:  0.99879646 	 acc:  0.46
1610 : loss:  0.997999 	 acc:  0.52
1612 : loss:  1.0514053 	 acc:  0.44
1614 : loss:  1.0269501 	 acc:  0.5
1616 : loss:  1.0003737 	 acc:  0.52
1618 : loss:  1.0663686 	 acc:  0.41
1620 : loss:  0.9870139 	 acc:  0.52
1622 : loss:  0.98004204 	 acc:  0.51
1624 : loss:  1.0755174 	 acc:  0.47
1626 : loss:  0.94755083 	 acc:  0.48
1628 : loss:  1.0473354 	 acc:  0.44
1630 : loss:  0.9536413 	 acc:  0.58
1632 : loss:  1.025292 	 acc:  0.45
1634 : loss:  1.0401032 	 acc:  0.51
1636 : loss:  0.9449087 	 acc:  0.54
1638 : loss:  1.0064474 	 acc:  0.46
1640 : loss:  0.9390888 	 acc:  0.5
1642 : loss:  1.0696976 	 acc:  0.42
1644 : loss:  1.0284258 	 acc:  0.47
1646 : loss:  1.0213147 	 acc:  0.46
1648 : loss:  1.0145814 	 acc:  0.47
1650 : loss:  1.0287443 	 acc:  0.44
1652 : loss:  1.0665803 	 acc:  0.41
1654 : loss:  1.0414699 	 acc:  0.5
1656 : loss:  1.0455028 	 acc:  0.48
1658 : loss:  1.0350392 	 acc:  0.41
1660 : loss:  0.97117406 	 acc:  0.57
1662 : loss:  1.0179296 	 acc:  0.44
1664 : loss:  1.0520569 	 acc:  0.4
1666 : loss:  1.0808941 	 acc:  0.4
1668 : loss:  1.0001383 	 acc:  0.39
1670 : loss:  1.0396178 	 acc:  0.45
1672 : loss:  1.0701954 	 acc:  0.46
1674 : loss:  1.089002 	 acc:  0.32
1676 : loss:  1.078354 	 acc:  0.38
1678 : loss:  0.96634746 	 acc:  0.53
1680 : loss:  1.0732445 	 acc:  0.41
1682 : loss:  0.9680933 	 acc:  0.54
1684 : loss:  0.95971143 	 acc:  0.53
1686 : loss:  0.98369324 	 acc:  0.59
1688 : loss:  1.031434 	 acc:  0.41
1690 : loss:  1.0156429 	 acc:  0.5
1692 : loss:  1.087202 	 acc:  0.4
1694 : loss:  1.0161139 	 acc:  0.48
1696 : loss:  0.99997747 	 acc:  0.51
1698 : loss:  0.99695283 	 acc:  0.43
1700 : loss:  0.9935415 	 acc:  0.51
1702 : loss:  0.98779005 	 acc:  0.51
1704 : loss:  1.0171907 	 acc:  0.48
1706 : loss:  1.0448806 	 acc:  0.46
1708 : loss:  0.9782445 	 acc:  0.53
1710 : loss:  1.0689908 	 acc:  0.44
1712 : loss:  0.9931502 	 acc:  0.48
1714 : loss:  0.97333884 	 acc:  0.5
1716 : loss:  1.0554185 	 acc:  0.4
1718 : loss:  0.98495704 	 acc:  0.49
1720 : loss:  1.0477487 	 acc:  0.37
1722 : loss:  0.9938659 	 acc:  0.51
1724 : loss:  0.9209532 	 acc:  0.53333336
1726 : loss:  1.0750676 	 acc:  0.37
1728 : loss:  1.0336913 	 acc:  0.42
1730 : loss:  1.0210333 	 acc:  0.44
1732 : loss:  1.0596124 	 acc:  0.4
1734 : loss:  0.9820195 	 acc:  0.52
1736 : loss:  1.0119529 	 acc:  0.45
1738 : loss:  1.0230538 	 acc:  0.49
1740 : loss:  1.0023558 	 acc:  0.43
1742 : loss:  0.9718462 	 acc:  0.51
1744 : loss:  0.97422904 	 acc:  0.5
1746 : loss:  1.0171074 	 acc:  0.48
1748 : loss:  1.0098448 	 acc:  0.51
1750 : loss:  0.98562956 	 acc:  0.53
1752 : loss:  0.9864031 	 acc:  0.49
1754 : loss:  1.0207 	 acc:  0.44
1756 : loss:  1.0278524 	 acc:  0.46
1758 : loss:  1.0283626 	 acc:  0.44
1760 : loss:  1.0054231 	 acc:  0.45
1762 : loss:  1.0034916 	 acc:  0.45
1764 : loss:  1.0058854 	 acc:  0.48
1766 : loss:  1.0192605 	 acc:  0.48
1768 : loss:  1.039182 	 acc:  0.48
1770 : loss:  0.99330574 	 acc:  0.51
1772 : loss:  1.0271773 	 acc:  0.46
1774 : loss:  1.0346627 	 acc:  0.42
1776 : loss:  1.0438615 	 acc:  0.43
1778 : loss:  1.0395167 	 acc:  0.47
1780 : loss:  0.9938195 	 acc:  0.45
1782 : loss:  0.97374946 	 acc:  0.5
1784 : loss:  0.99285597 	 acc:  0.52
1786 : loss:  1.0003058 	 acc:  0.46
1788 : loss:  1.0143111 	 acc:  0.51
1790 : loss:  0.9752153 	 acc:  0.49
1792 : loss:  1.0000004 	 acc:  0.55
1794 : loss:  1.0347084 	 acc:  0.44
1796 : loss:  1.0314151 	 acc:  0.44
1798 : loss:  0.94340956 	 acc:  0.59
1800 : loss:  1.1001418 	 acc:  0.38
1802 : loss:  1.0061642 	 acc:  0.53
1804 : loss:  1.078353 	 acc:  0.35
1806 : loss:  1.0086001 	 acc:  0.51
1808 : loss:  0.95390695 	 acc:  0.54
1810 : loss:  1.0786625 	 acc:  0.44
1812 : loss:  0.96582955 	 acc:  0.54
1814 : loss:  1.0279528 	 acc:  0.4
1816 : loss:  1.0103858 	 acc:  0.44
1818 : loss:  1.0520804 	 acc:  0.48
1820 : loss:  1.0293002 	 acc:  0.48
1822 : loss:  1.0943888 	 acc:  0.36
1824 : loss:  0.9770388 	 acc:  0.43
1826 : loss:  1.0167978 	 acc:  0.51
1828 : loss:  1.0410402 	 acc:  0.41
1830 : loss:  1.0495882 	 acc:  0.37
1832 : loss:  1.0806309 	 acc:  0.4
1834 : loss:  1.0564996 	 acc:  0.43
1836 : loss:  1.0393679 	 acc:  0.44
1838 : loss:  1.0053391 	 acc:  0.45
1840 : loss:  0.97105056 	 acc:  0.54
1842 : loss:  0.9953668 	 acc:  0.47
1844 : loss:  0.9823074 	 acc:  0.49
1846 : loss:  0.99633235 	 acc:  0.56
1848 : loss:  0.9897208 	 acc:  0.47
1850 : loss:  1.0182471 	 acc:  0.53
1852 : loss:  1.0217971 	 acc:  0.53
1854 : loss:  1.033722 	 acc:  0.46
1856 : loss:  1.0105857 	 acc:  0.42
1858 : loss:  0.9910945 	 acc:  0.5
acc:  0.51
acc:  0.42
acc:  0.49
acc:  0.49
acc:  0.47
acc:  0.5
acc:  0.47
acc:  0.49
acc:  0.44
acc:  0.53
acc:  0.53
acc:  0.42
acc:  0.48
acc:  0.42
acc:  0.47
acc:  0.42
acc:  0.5
acc:  0.6
acc:  0.44
acc:  0.51
acc:  0.51
acc:  0.48
acc:  0.5
acc:  0.43
acc:  0.44
acc:  0.51
acc:  0.53
acc:  0.44
acc:  0.56
acc:  0.45
acc:  0.51
acc:  0.47
acc:  0.46
acc:  0.43
acc:  0.53
acc:  0.46
acc:  0.44
acc:  0.51
acc:  0.45
acc:  0.53
acc:  0.43
acc:  0.47
acc:  0.5
acc:  0.46
acc:  0.42
acc:  0.5
acc:  0.53
acc:  0.44
acc:  0.43
acc:  0.44
acc:  0.48
acc:  0.53
acc:  0.54
acc:  0.44
acc:  0.54
acc:  0.52
acc:  0.42
acc:  0.45
acc:  0.44
acc:  0.46
acc:  0.44
acc:  0.5
acc:  0.4
acc:  0.43
acc:  0.47
acc:  0.41
acc:  0.5
acc:  0.57
acc:  0.47
acc:  0.5
acc:  0.4
acc:  0.47
acc:  0.43
acc:  0.47
acc:  0.42
acc:  0.44
acc:  0.46
acc:  0.48
acc:  0.49
acc:  0.49
acc:  0.54
acc:  0.46
acc:  0.47
acc:  0.44
acc:  0.58
acc:  0.47
acc:  0.54
acc:  0.47
acc:  0.48
acc:  0.46
acc:  0.52
acc:  0.45
acc:  0.46
acc:  0.51
acc:  0.45
acc:  0.44
acc:  0.45
acc:  0.5
acc:  0.45
acc:  0.43

100 	 [54.02553547 54.44811771 21.95237048]
4 	val accuracy:  0.47489998 	 f_! score:  [0.54025535 0.54448118 0.2195237 ]

1860 : loss:  1.0754958 	 acc:  0.44
1862 : loss:  1.0491921 	 acc:  0.46
1864 : loss:  0.9850052 	 acc:  0.44
1866 : loss:  1.0283339 	 acc:  0.47
1868 : loss:  1.0604283 	 acc:  0.42
1870 : loss:  1.0353504 	 acc:  0.4
1872 : loss:  0.8860166 	 acc:  0.6333333
1874 : loss:  1.0905182 	 acc:  0.4
1876 : loss:  1.0411936 	 acc:  0.4
1878 : loss:  1.0229771 	 acc:  0.46
1880 : loss:  1.0238184 	 acc:  0.43
1882 : loss:  1.0753726 	 acc:  0.42
1884 : loss:  1.0094974 	 acc:  0.43
1886 : loss:  0.9732413 	 acc:  0.5
1888 : loss:  1.0010221 	 acc:  0.55
1890 : loss:  1.0310028 	 acc:  0.48
1892 : loss:  1.0719186 	 acc:  0.44
1894 : loss:  1.0554152 	 acc:  0.44
1896 : loss:  0.92136985 	 acc:  0.61
1898 : loss:  1.0215055 	 acc:  0.47
1900 : loss:  1.0769813 	 acc:  0.32
1902 : loss:  1.0378007 	 acc:  0.41
1904 : loss:  1.0124846 	 acc:  0.43
1906 : loss:  0.9977929 	 acc:  0.52
1908 : loss:  1.0132092 	 acc:  0.45
1910 : loss:  1.0512893 	 acc:  0.45
1912 : loss:  0.9872899 	 acc:  0.58
1914 : loss:  1.0390043 	 acc:  0.46
1916 : loss:  1.0148249 	 acc:  0.5
1918 : loss:  1.0546012 	 acc:  0.52
1920 : loss:  1.0234697 	 acc:  0.48
1922 : loss:  1.0319687 	 acc:  0.5
1924 : loss:  1.014997 	 acc:  0.47
1926 : loss:  0.9551135 	 acc:  0.57
1928 : loss:  0.95751834 	 acc:  0.52
1930 : loss:  1.0476332 	 acc:  0.43
1932 : loss:  1.0504433 	 acc:  0.47
1934 : loss:  1.0614225 	 acc:  0.46
1936 : loss:  1.0720164 	 acc:  0.49
1938 : loss:  0.94250053 	 acc:  0.54
1940 : loss:  1.0227257 	 acc:  0.45
1942 : loss:  1.0921335 	 acc:  0.36
1944 : loss:  1.0980401 	 acc:  0.4
1946 : loss:  0.98054636 	 acc:  0.5
1948 : loss:  0.96954346 	 acc:  0.54
1950 : loss:  1.0125585 	 acc:  0.48
1952 : loss:  1.0209593 	 acc:  0.44
1954 : loss:  1.0406272 	 acc:  0.44
1956 : loss:  0.9841655 	 acc:  0.46
1958 : loss:  1.0452446 	 acc:  0.54
1960 : loss:  1.0530467 	 acc:  0.47
1962 : loss:  1.0153179 	 acc:  0.48
1964 : loss:  1.0188224 	 acc:  0.51
1966 : loss:  1.0663459 	 acc:  0.41
1968 : loss:  1.0188985 	 acc:  0.42
1970 : loss:  1.0633559 	 acc:  0.37
1972 : loss:  1.0570942 	 acc:  0.41
1974 : loss:  0.99355227 	 acc:  0.49
1976 : loss:  0.9913927 	 acc:  0.48
1978 : loss:  0.98162895 	 acc:  0.43
1980 : loss:  0.9813868 	 acc:  0.49
1982 : loss:  0.9987351 	 acc:  0.48
1984 : loss:  1.0028595 	 acc:  0.46
1986 : loss:  1.0145783 	 acc:  0.47
1988 : loss:  0.9943057 	 acc:  0.48
1990 : loss:  0.9391847 	 acc:  0.55
1992 : loss:  0.97988075 	 acc:  0.51
1994 : loss:  1.0298667 	 acc:  0.5
1996 : loss:  0.99785936 	 acc:  0.41
1998 : loss:  1.0234165 	 acc:  0.45
2000 : loss:  0.93095225 	 acc:  0.64
2002 : loss:  1.0280771 	 acc:  0.48
2004 : loss:  1.0830005 	 acc:  0.42
2006 : loss:  0.9863548 	 acc:  0.5
2008 : loss:  0.95854616 	 acc:  0.52
2010 : loss:  1.0391611 	 acc:  0.43
2012 : loss:  1.0235372 	 acc:  0.49
2014 : loss:  1.0210614 	 acc:  0.42
2016 : loss:  1.0149797 	 acc:  0.44
2018 : loss:  1.0642498 	 acc:  0.41
2020 : loss:  1.0205518 	 acc:  0.47
2022 : loss:  1.011438 	 acc:  0.51
2024 : loss:  0.93707085 	 acc:  0.53
2026 : loss:  0.9979156 	 acc:  0.46
2028 : loss:  1.051641 	 acc:  0.42
2030 : loss:  1.1141373 	 acc:  0.4
2032 : loss:  1.0338444 	 acc:  0.43
2034 : loss:  0.9409692 	 acc:  0.53
2036 : loss:  0.9654494 	 acc:  0.52
2038 : loss:  1.0262749 	 acc:  0.49
2040 : loss:  1.0409149 	 acc:  0.43
2042 : loss:  0.9919008 	 acc:  0.43
2044 : loss:  0.95290536 	 acc:  0.55
2046 : loss:  1.0203665 	 acc:  0.45
2048 : loss:  1.0443801 	 acc:  0.43
2050 : loss:  0.98167586 	 acc:  0.51
2052 : loss:  0.9699753 	 acc:  0.5
2054 : loss:  1.011298 	 acc:  0.47
2056 : loss:  1.0228502 	 acc:  0.53
2058 : loss:  1.0725985 	 acc:  0.39
2060 : loss:  1.0591313 	 acc:  0.42
2062 : loss:  0.9957968 	 acc:  0.53
2064 : loss:  1.0200254 	 acc:  0.45
2066 : loss:  0.9444225 	 acc:  0.53
2068 : loss:  1.0176784 	 acc:  0.41
2070 : loss:  0.9266016 	 acc:  0.6
2072 : loss:  1.0133995 	 acc:  0.48
2074 : loss:  1.0197939 	 acc:  0.43
2076 : loss:  1.0028067 	 acc:  0.47
2078 : loss:  1.0112208 	 acc:  0.52
2080 : loss:  1.0424273 	 acc:  0.46
2082 : loss:  0.98470515 	 acc:  0.54
2084 : loss:  0.9544213 	 acc:  0.52
2086 : loss:  1.0040138 	 acc:  0.47
2088 : loss:  0.9662694 	 acc:  0.46
2090 : loss:  1.0189227 	 acc:  0.52
2092 : loss:  0.96144074 	 acc:  0.57
2094 : loss:  1.0568643 	 acc:  0.43
2096 : loss:  0.9791079 	 acc:  0.48
2098 : loss:  1.0334774 	 acc:  0.44
2100 : loss:  0.98127586 	 acc:  0.55
2102 : loss:  1.004103 	 acc:  0.43
2104 : loss:  1.0436991 	 acc:  0.42
2106 : loss:  1.0390884 	 acc:  0.42
2108 : loss:  0.99501497 	 acc:  0.55
2110 : loss:  1.0288428 	 acc:  0.48
2112 : loss:  0.9868562 	 acc:  0.51
2114 : loss:  1.0075827 	 acc:  0.49
2116 : loss:  1.0179868 	 acc:  0.47
2118 : loss:  1.0265889 	 acc:  0.38
2120 : loss:  1.0219948 	 acc:  0.5
2122 : loss:  0.9796619 	 acc:  0.49
2124 : loss:  0.990075 	 acc:  0.49
2126 : loss:  0.992193 	 acc:  0.5
2128 : loss:  0.9466387 	 acc:  0.55
2130 : loss:  1.0142704 	 acc:  0.41
2132 : loss:  0.97701764 	 acc:  0.56
2134 : loss:  0.96245265 	 acc:  0.56
2136 : loss:  1.0018501 	 acc:  0.48
2138 : loss:  0.9306061 	 acc:  0.54
2140 : loss:  1.0191414 	 acc:  0.47
2142 : loss:  1.0333579 	 acc:  0.45
2144 : loss:  0.9139325 	 acc:  0.55
2146 : loss:  0.9708897 	 acc:  0.58
2148 : loss:  0.9780251 	 acc:  0.57
2150 : loss:  1.0220953 	 acc:  0.48
2152 : loss:  0.9958422 	 acc:  0.5
2154 : loss:  1.0151736 	 acc:  0.47
2156 : loss:  0.97589296 	 acc:  0.53
2158 : loss:  1.0039055 	 acc:  0.5
2160 : loss:  1.0198298 	 acc:  0.46
2162 : loss:  1.0347668 	 acc:  0.45
2164 : loss:  0.99628854 	 acc:  0.46
2166 : loss:  0.9902813 	 acc:  0.52
2168 : loss:  0.9675173 	 acc:  0.58
2170 : loss:  0.9834005 	 acc:  0.55
2172 : loss:  1.0771513 	 acc:  0.41
2174 : loss:  1.0076305 	 acc:  0.53
2176 : loss:  1.0238314 	 acc:  0.49
2178 : loss:  1.0473713 	 acc:  0.46
2180 : loss:  1.0700723 	 acc:  0.43
2182 : loss:  0.9549511 	 acc:  0.54
2184 : loss:  1.0647688 	 acc:  0.43
2186 : loss:  1.0152308 	 acc:  0.48
2188 : loss:  0.99956465 	 acc:  0.53
2190 : loss:  1.0210638 	 acc:  0.44
2192 : loss:  0.9106237 	 acc:  0.62
2194 : loss:  1.0327514 	 acc:  0.44
2196 : loss:  0.962955 	 acc:  0.53
2198 : loss:  1.0064343 	 acc:  0.5
2200 : loss:  0.9948093 	 acc:  0.48
2202 : loss:  1.0250872 	 acc:  0.45
2204 : loss:  1.0253989 	 acc:  0.46
2206 : loss:  1.0413806 	 acc:  0.47
2208 : loss:  0.9901202 	 acc:  0.46
2210 : loss:  1.0535004 	 acc:  0.47
2212 : loss:  1.0169905 	 acc:  0.46
2214 : loss:  0.9977745 	 acc:  0.49
2216 : loss:  1.0215243 	 acc:  0.5
2218 : loss:  1.0662667 	 acc:  0.42
2220 : loss:  1.001811 	 acc:  0.46
2222 : loss:  1.0078497 	 acc:  0.49
2224 : loss:  1.0238547 	 acc:  0.39
2226 : loss:  0.9532311 	 acc:  0.55
2228 : loss:  1.0020405 	 acc:  0.48
2230 : loss:  1.0190862 	 acc:  0.47
acc:  0.52
acc:  0.43
acc:  0.52
acc:  0.46
acc:  0.55
acc:  0.53
acc:  0.47
acc:  0.55
acc:  0.48
acc:  0.47
acc:  0.62
acc:  0.47
acc:  0.51
acc:  0.54
acc:  0.48
acc:  0.47
acc:  0.51
acc:  0.48
acc:  0.48
acc:  0.45
acc:  0.41
acc:  0.51
acc:  0.4
acc:  0.45
acc:  0.45
acc:  0.36
acc:  0.48
acc:  0.48
acc:  0.5
acc:  0.54
acc:  0.41
acc:  0.42
acc:  0.44
acc:  0.48
acc:  0.49
acc:  0.44
acc:  0.49
acc:  0.54
acc:  0.43
acc:  0.61
acc:  0.56
acc:  0.5
acc:  0.5
acc:  0.46
acc:  0.5
acc:  0.53
acc:  0.42
acc:  0.51
acc:  0.5
acc:  0.54
acc:  0.51
acc:  0.44
acc:  0.55
acc:  0.46
acc:  0.51
acc:  0.46
acc:  0.49
acc:  0.52
acc:  0.54
acc:  0.43
acc:  0.46
acc:  0.52
acc:  0.56
acc:  0.49
acc:  0.5
acc:  0.48
acc:  0.55
acc:  0.36
acc:  0.51
acc:  0.53
acc:  0.52
acc:  0.57
acc:  0.56
acc:  0.51
acc:  0.41
acc:  0.49
acc:  0.54
acc:  0.43
acc:  0.54
acc:  0.47
acc:  0.51
acc:  0.45
acc:  0.57
acc:  0.48
acc:  0.5
acc:  0.51
acc:  0.41
acc:  0.51
acc:  0.55
acc:  0.42
acc:  0.44
acc:  0.46
acc:  0.5
acc:  0.57
acc:  0.48
acc:  0.5
acc:  0.45
acc:  0.49
acc:  0.48
acc:  0.5

100 	 [50.37651332 52.95925065 42.48865593]
5 	val accuracy:  0.4903 	 f_! score:  [0.50376513 0.52959251 0.42488656]

2232 : loss:  1.0681775 	 acc:  0.47
2234 : loss:  1.0449663 	 acc:  0.47
2236 : loss:  0.97772026 	 acc:  0.57
2238 : loss:  1.0437282 	 acc:  0.39
2240 : loss:  1.0216916 	 acc:  0.44
2242 : loss:  1.0241193 	 acc:  0.43
2244 : loss:  0.9834869 	 acc:  0.49
2246 : loss:  1.0173879 	 acc:  0.53
2248 : loss:  1.0499536 	 acc:  0.45
2250 : loss:  1.0011181 	 acc:  0.44
2252 : loss:  0.9969376 	 acc:  0.48
2254 : loss:  0.9684443 	 acc:  0.54
2256 : loss:  0.8946563 	 acc:  0.63
2258 : loss:  0.9821017 	 acc:  0.46
2260 : loss:  1.0202932 	 acc:  0.45
2262 : loss:  1.0504476 	 acc:  0.45
2264 : loss:  1.0079675 	 acc:  0.47
2266 : loss:  1.0326991 	 acc:  0.43
2268 : loss:  1.0090013 	 acc:  0.52
2270 : loss:  1.0104973 	 acc:  0.46
2272 : loss:  0.9844698 	 acc:  0.5
2274 : loss:  1.1029166 	 acc:  0.35
2276 : loss:  1.0166644 	 acc:  0.5
2278 : loss:  1.0128154 	 acc:  0.44
2280 : loss:  1.0771227 	 acc:  0.4
2282 : loss:  1.0464147 	 acc:  0.41
2284 : loss:  1.0603629 	 acc:  0.46
2286 : loss:  1.0918155 	 acc:  0.42
2288 : loss:  1.0695072 	 acc:  0.45
2290 : loss:  1.0071154 	 acc:  0.51
2292 : loss:  1.0642877 	 acc:  0.36
2294 : loss:  1.0338452 	 acc:  0.47
2296 : loss:  0.96366477 	 acc:  0.52
2298 : loss:  1.0067452 	 acc:  0.43
2300 : loss:  0.9751971 	 acc:  0.59
2302 : loss:  1.02666 	 acc:  0.42
2304 : loss:  1.0309613 	 acc:  0.42
2306 : loss:  1.0048704 	 acc:  0.5
2308 : loss:  0.9556201 	 acc:  0.55
2310 : loss:  1.0356019 	 acc:  0.5
2312 : loss:  1.0111922 	 acc:  0.53
2314 : loss:  0.9260334 	 acc:  0.57
2316 : loss:  1.0437003 	 acc:  0.48
2318 : loss:  1.0139757 	 acc:  0.47
2320 : loss:  1.1135018 	 acc:  0.3
2322 : loss:  0.9614787 	 acc:  0.53
2324 : loss:  1.0115863 	 acc:  0.47
2326 : loss:  0.9379457 	 acc:  0.53
2328 : loss:  1.0089511 	 acc:  0.43
2330 : loss:  1.0365996 	 acc:  0.42
2332 : loss:  0.99384207 	 acc:  0.48
2334 : loss:  0.98845917 	 acc:  0.48
2336 : loss:  1.0090958 	 acc:  0.46
2338 : loss:  0.99926114 	 acc:  0.5
2340 : loss:  1.059957 	 acc:  0.45
2342 : loss:  0.9746198 	 acc:  0.57
2344 : loss:  1.082551 	 acc:  0.38
2346 : loss:  1.0404257 	 acc:  0.46
2348 : loss:  1.0084001 	 acc:  0.49
2350 : loss:  1.0730908 	 acc:  0.45
2352 : loss:  1.0436939 	 acc:  0.4
2354 : loss:  1.0452129 	 acc:  0.46
2356 : loss:  0.9940692 	 acc:  0.46
2358 : loss:  0.9653748 	 acc:  0.56
2360 : loss:  0.9855413 	 acc:  0.47
2362 : loss:  1.08909 	 acc:  0.41
2364 : loss:  0.96478266 	 acc:  0.54
2366 : loss:  1.0143287 	 acc:  0.48
2368 : loss:  0.98989177 	 acc:  0.5
2370 : loss:  0.98213506 	 acc:  0.51
2372 : loss:  0.9646038 	 acc:  0.57
2374 : loss:  1.0152987 	 acc:  0.5
2376 : loss:  0.9740984 	 acc:  0.52
2378 : loss:  0.99161804 	 acc:  0.45
2380 : loss:  1.1110054 	 acc:  0.37
2382 : loss:  0.967399 	 acc:  0.55
2384 : loss:  0.9964937 	 acc:  0.48
2386 : loss:  1.0182686 	 acc:  0.48
2388 : loss:  0.9911594 	 acc:  0.47
2390 : loss:  0.9779767 	 acc:  0.54
2392 : loss:  1.0030078 	 acc:  0.44
2394 : loss:  1.0536602 	 acc:  0.53
2396 : loss:  0.99288297 	 acc:  0.52
2398 : loss:  0.9856441 	 acc:  0.54
2400 : loss:  1.0232049 	 acc:  0.45
2402 : loss:  0.9988588 	 acc:  0.48
2404 : loss:  1.0266665 	 acc:  0.43
2406 : loss:  0.99942803 	 acc:  0.46
2408 : loss:  0.9845025 	 acc:  0.53
2410 : loss:  1.0098265 	 acc:  0.51
2412 : loss:  1.0659983 	 acc:  0.43
2414 : loss:  0.94812584 	 acc:  0.56
2416 : loss:  1.0360833 	 acc:  0.47
2418 : loss:  1.0368891 	 acc:  0.46
2420 : loss:  0.9384409 	 acc:  0.55
2422 : loss:  1.0516942 	 acc:  0.46
2424 : loss:  0.9638054 	 acc:  0.52
2426 : loss:  1.0353715 	 acc:  0.42
2428 : loss:  1.0966374 	 acc:  0.34
2430 : loss:  0.98934335 	 acc:  0.49
2432 : loss:  1.0287365 	 acc:  0.47
2434 : loss:  0.9697753 	 acc:  0.53
2436 : loss:  0.96184 	 acc:  0.49
2438 : loss:  0.9546185 	 acc:  0.5
2440 : loss:  1.055103 	 acc:  0.49
2442 : loss:  1.1031517 	 acc:  0.41
2444 : loss:  1.0457966 	 acc:  0.44
2446 : loss:  0.9452419 	 acc:  0.58
2448 : loss:  0.99903333 	 acc:  0.49
2450 : loss:  0.9806715 	 acc:  0.48
2452 : loss:  1.0070182 	 acc:  0.42
2454 : loss:  1.0471202 	 acc:  0.37
2456 : loss:  1.0580075 	 acc:  0.5
2458 : loss:  1.0307811 	 acc:  0.52
2460 : loss:  1.031079 	 acc:  0.49
2462 : loss:  1.0409112 	 acc:  0.42
2464 : loss:  0.9995728 	 acc:  0.51
2466 : loss:  0.99796057 	 acc:  0.49
2468 : loss:  0.9904273 	 acc:  0.48
2470 : loss:  0.9777386 	 acc:  0.52
2472 : loss:  1.041314 	 acc:  0.43
2474 : loss:  0.96715134 	 acc:  0.57
2476 : loss:  0.99962175 	 acc:  0.45
2478 : loss:  0.9688009 	 acc:  0.51
2480 : loss:  1.0422661 	 acc:  0.48
2482 : loss:  0.9750673 	 acc:  0.5
2484 : loss:  1.0361297 	 acc:  0.51
2486 : loss:  1.039055 	 acc:  0.42
2488 : loss:  0.97515106 	 acc:  0.51
2490 : loss:  1.0109879 	 acc:  0.45
2492 : loss:  1.0693306 	 acc:  0.4
2494 : loss:  1.0469383 	 acc:  0.41
2496 : loss:  0.9995564 	 acc:  0.47
2498 : loss:  1.0579718 	 acc:  0.46
2500 : loss:  1.0000783 	 acc:  0.46
2502 : loss:  0.988303 	 acc:  0.51
2504 : loss:  1.0353771 	 acc:  0.46
2506 : loss:  1.04717 	 acc:  0.47
2508 : loss:  1.0370135 	 acc:  0.47
2510 : loss:  0.94062376 	 acc:  0.5
2512 : loss:  1.0137862 	 acc:  0.44
2514 : loss:  1.0359844 	 acc:  0.39
2516 : loss:  1.022781 	 acc:  0.47
2518 : loss:  0.9610353 	 acc:  0.53
2520 : loss:  0.9190615 	 acc:  0.52
2522 : loss:  1.0318542 	 acc:  0.47
2524 : loss:  1.0107783 	 acc:  0.46
2526 : loss:  1.0821996 	 acc:  0.39
2528 : loss:  0.9738636 	 acc:  0.5
2530 : loss:  0.99708194 	 acc:  0.57
2532 : loss:  0.9803999 	 acc:  0.51
2534 : loss:  0.98680913 	 acc:  0.47
2536 : loss:  1.0391254 	 acc:  0.47
2538 : loss:  1.0388094 	 acc:  0.44
2540 : loss:  0.9863278 	 acc:  0.47
2542 : loss:  1.0203178 	 acc:  0.43
2544 : loss:  1.0085162 	 acc:  0.48
2546 : loss:  1.076791 	 acc:  0.42
2548 : loss:  1.0131488 	 acc:  0.47
2550 : loss:  1.0110022 	 acc:  0.48
2552 : loss:  0.94882447 	 acc:  0.58
2554 : loss:  1.0537901 	 acc:  0.43
2556 : loss:  1.0293288 	 acc:  0.43
2558 : loss:  1.0074785 	 acc:  0.45
2560 : loss:  1.0607607 	 acc:  0.42
2562 : loss:  1.0286474 	 acc:  0.45
2564 : loss:  1.0295568 	 acc:  0.45
2566 : loss:  1.0206211 	 acc:  0.46
2568 : loss:  1.0254772 	 acc:  0.44
2570 : loss:  0.9845941 	 acc:  0.48
2572 : loss:  1.0254867 	 acc:  0.44
2574 : loss:  1.0230019 	 acc:  0.47
2576 : loss:  1.0182724 	 acc:  0.46
2578 : loss:  1.0021056 	 acc:  0.52
2580 : loss:  0.96417826 	 acc:  0.54
2582 : loss:  1.0462673 	 acc:  0.41
2584 : loss:  0.97128874 	 acc:  0.53
2586 : loss:  0.9371842 	 acc:  0.52
2588 : loss:  1.0102782 	 acc:  0.46
2590 : loss:  0.97655267 	 acc:  0.54
2592 : loss:  1.0234312 	 acc:  0.5
2594 : loss:  1.0600641 	 acc:  0.47
2596 : loss:  1.0587991 	 acc:  0.42
2598 : loss:  1.0353284 	 acc:  0.41
2600 : loss:  0.99743515 	 acc:  0.5
2602 : loss:  1.0057957 	 acc:  0.51
acc:  0.47
acc:  0.47
acc:  0.5
acc:  0.53
acc:  0.47
acc:  0.46
acc:  0.48
acc:  0.42
acc:  0.47
acc:  0.5
acc:  0.54
acc:  0.43
acc:  0.56
acc:  0.47
acc:  0.44
acc:  0.54
acc:  0.48
acc:  0.48
acc:  0.45
acc:  0.39
acc:  0.5
acc:  0.48
acc:  0.58
acc:  0.48
acc:  0.53
acc:  0.52
acc:  0.48
acc:  0.56
acc:  0.54
acc:  0.41
acc:  0.52
acc:  0.52
acc:  0.52
acc:  0.49
acc:  0.45
acc:  0.54
acc:  0.47
acc:  0.49
acc:  0.53
acc:  0.54
acc:  0.52
acc:  0.52
acc:  0.5
acc:  0.44
acc:  0.56
acc:  0.45
acc:  0.51
acc:  0.46
acc:  0.48
acc:  0.48
acc:  0.49
acc:  0.61
acc:  0.44
acc:  0.49
acc:  0.45
acc:  0.57
acc:  0.49
acc:  0.5
acc:  0.58
acc:  0.48
acc:  0.46
acc:  0.51
acc:  0.48
acc:  0.44
acc:  0.37
acc:  0.54
acc:  0.52
acc:  0.52
acc:  0.5
acc:  0.45
acc:  0.47
acc:  0.56
acc:  0.48
acc:  0.54
acc:  0.44
acc:  0.41
acc:  0.57
acc:  0.41
acc:  0.37
acc:  0.43
acc:  0.48
acc:  0.44
acc:  0.38
acc:  0.49
acc:  0.42
acc:  0.48
acc:  0.52
acc:  0.47
acc:  0.54
acc:  0.51
acc:  0.48
acc:  0.43
acc:  0.39
acc:  0.47
acc:  0.44
acc:  0.64
acc:  0.52
acc:  0.52
acc:  0.52
acc:  0.53

100 	 [55.02581202 47.72509942 41.89673298]
6 	val accuracy:  0.48860002 	 f_! score:  [0.55025812 0.47725099 0.41896733]

100 	 [55.02581202 47.72509942 41.89673298]
100 	 [50.84819801 51.49511843 43.83210585]
100 	 [60.76658427 45.18284357 40.73093485]
100 	 [3261. 3347. 3392.]
---train_last_layer Test  Twitter ---
0.48860002
f1:  [0.55025812 0.47725099 0.41896733]
--- 157.20955872535706 seconds ---


  Twitter
65730
2018-10-14 19:29:35.907375: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-10-14 19:29:35.907422: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-14 19:29:35.907429: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-10-14 19:29:35.907433: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-10-14 19:29:35.907533: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2909 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Restoring...
models/CNN/pretrained_model.ckpt-17931
---init ready   CNN ---
2018-10-14 19:29:38.868698: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2018-10-14 19:29:38.868736: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-10-14 19:29:38.868742: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2018-10-14 19:29:38.868749: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2018-10-14 19:29:38.868840: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 2909 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)

Saving...
saved to models/CNN/pretrained_model.ckpt-0
0 : loss:  2.523707 	 acc:  0.3
2 : loss:  1.113907 	 acc:  0.35
4 : loss:  1.0728161 	 acc:  0.47
6 : loss:  1.0834985 	 acc:  0.36
8 : loss:  1.0936515 	 acc:  0.34
10 : loss:  1.0928338 	 acc:  0.34
12 : loss:  1.0759722 	 acc:  0.38
14 : loss:  1.0904812 	 acc:  0.38
16 : loss:  1.1016818 	 acc:  0.41
18 : loss:  1.0609406 	 acc:  0.44
20 : loss:  1.0822965 	 acc:  0.39
22 : loss:  1.0741299 	 acc:  0.43
24 : loss:  1.0320141 	 acc:  0.44
26 : loss:  1.0024966 	 acc:  0.54
28 : loss:  1.0014312 	 acc:  0.47
30 : loss:  1.0748248 	 acc:  0.47
32 : loss:  0.97324693 	 acc:  0.49
34 : loss:  1.0852916 	 acc:  0.36
36 : loss:  1.0116521 	 acc:  0.47
38 : loss:  1.0300163 	 acc:  0.41
40 : loss:  1.0464306 	 acc:  0.46
42 : loss:  1.0131706 	 acc:  0.43
44 : loss:  1.0601869 	 acc:  0.37
46 : loss:  0.9999321 	 acc:  0.55
48 : loss:  1.031479 	 acc:  0.48
50 : loss:  0.9032216 	 acc:  0.49
52 : loss:  1.0083162 	 acc:  0.46
54 : loss:  1.0367823 	 acc:  0.5
56 : loss:  0.99206084 	 acc:  0.53
58 : loss:  1.0245755 	 acc:  0.42
60 : loss:  1.0176221 	 acc:  0.48
62 : loss:  0.9715976 	 acc:  0.55
64 : loss:  0.97369134 	 acc:  0.46
66 : loss:  0.9888136 	 acc:  0.56
68 : loss:  0.92947775 	 acc:  0.54
70 : loss:  1.0987855 	 acc:  0.37
72 : loss:  1.0089501 	 acc:  0.41
74 : loss:  0.995618 	 acc:  0.47
76 : loss:  0.9965498 	 acc:  0.45
78 : loss:  0.9945032 	 acc:  0.45
80 : loss:  1.0058421 	 acc:  0.47
82 : loss:  0.93405807 	 acc:  0.49
84 : loss:  0.98243356 	 acc:  0.59
86 : loss:  0.9060709 	 acc:  0.49
88 : loss:  0.9962597 	 acc:  0.45
90 : loss:  0.9423967 	 acc:  0.52
92 : loss:  0.98780817 	 acc:  0.53
94 : loss:  0.96811396 	 acc:  0.45
96 : loss:  0.9545473 	 acc:  0.5
98 : loss:  0.9458677 	 acc:  0.47
100 : loss:  0.958779 	 acc:  0.48
102 : loss:  1.0008658 	 acc:  0.52
104 : loss:  0.9857145 	 acc:  0.53
106 : loss:  0.96034926 	 acc:  0.52
108 : loss:  0.9862278 	 acc:  0.49
110 : loss:  0.9627588 	 acc:  0.54
112 : loss:  0.9386804 	 acc:  0.49
114 : loss:  1.0273575 	 acc:  0.45
116 : loss:  0.9656648 	 acc:  0.46
118 : loss:  0.94884676 	 acc:  0.47
120 : loss:  1.0148469 	 acc:  0.54
122 : loss:  0.94864184 	 acc:  0.52
124 : loss:  0.92480683 	 acc:  0.54
126 : loss:  0.99803436 	 acc:  0.47
128 : loss:  0.9005295 	 acc:  0.5
130 : loss:  1.042918 	 acc:  0.41
132 : loss:  0.9519049 	 acc:  0.5
134 : loss:  1.0677227 	 acc:  0.4
136 : loss:  1.0183735 	 acc:  0.39
138 : loss:  0.93668234 	 acc:  0.55
140 : loss:  0.8899874 	 acc:  0.54
142 : loss:  0.98415244 	 acc:  0.51
144 : loss:  0.95423365 	 acc:  0.49
146 : loss:  0.9619335 	 acc:  0.45
148 : loss:  0.91980255 	 acc:  0.5
150 : loss:  0.9177227 	 acc:  0.51
152 : loss:  0.9324434 	 acc:  0.52
154 : loss:  0.9532083 	 acc:  0.46
156 : loss:  1.0814507 	 acc:  0.43
158 : loss:  0.9720747 	 acc:  0.53
160 : loss:  0.91832376 	 acc:  0.48
162 : loss:  1.0149219 	 acc:  0.43
164 : loss:  0.8939769 	 acc:  0.52
166 : loss:  0.8618759 	 acc:  0.6
168 : loss:  1.0230324 	 acc:  0.54
170 : loss:  0.86480963 	 acc:  0.56
172 : loss:  0.9904706 	 acc:  0.42
174 : loss:  0.964689 	 acc:  0.49
176 : loss:  0.8895727 	 acc:  0.47
178 : loss:  0.95971674 	 acc:  0.58
180 : loss:  0.9113029 	 acc:  0.54
182 : loss:  0.99835753 	 acc:  0.37
184 : loss:  0.9995553 	 acc:  0.49
186 : loss:  1.0405145 	 acc:  0.49
188 : loss:  0.9519618 	 acc:  0.51
190 : loss:  0.97062385 	 acc:  0.51
192 : loss:  0.99470145 	 acc:  0.43
194 : loss:  0.9644542 	 acc:  0.53
196 : loss:  0.9617241 	 acc:  0.51
198 : loss:  1.0114408 	 acc:  0.43
200 : loss:  0.87439394 	 acc:  0.58
202 : loss:  0.88290054 	 acc:  0.45
204 : loss:  0.88532984 	 acc:  0.51
206 : loss:  1.007405 	 acc:  0.53
208 : loss:  0.9791665 	 acc:  0.47
210 : loss:  0.95468897 	 acc:  0.5
212 : loss:  0.9626588 	 acc:  0.54
214 : loss:  0.890555 	 acc:  0.56
216 : loss:  0.93871325 	 acc:  0.54
218 : loss:  0.970281 	 acc:  0.53
220 : loss:  0.9039119 	 acc:  0.51
222 : loss:  0.96046495 	 acc:  0.47
224 : loss:  0.8811545 	 acc:  0.6
226 : loss:  0.95516306 	 acc:  0.5
228 : loss:  0.9595987 	 acc:  0.49
230 : loss:  0.94461215 	 acc:  0.54
232 : loss:  0.8744231 	 acc:  0.52
234 : loss:  0.820578 	 acc:  0.64
236 : loss:  0.9281395 	 acc:  0.55
238 : loss:  0.9679675 	 acc:  0.52
240 : loss:  0.9064968 	 acc:  0.57
242 : loss:  1.0053738 	 acc:  0.46
244 : loss:  1.0059794 	 acc:  0.48
246 : loss:  0.9773478 	 acc:  0.55
248 : loss:  0.944382 	 acc:  0.46
250 : loss:  0.9515773 	 acc:  0.52
252 : loss:  0.9984958 	 acc:  0.46
254 : loss:  0.9031945 	 acc:  0.54
256 : loss:  0.91356117 	 acc:  0.54
258 : loss:  0.9329186 	 acc:  0.51
260 : loss:  0.88936406 	 acc:  0.59
262 : loss:  0.91760635 	 acc:  0.55
264 : loss:  1.0167428 	 acc:  0.47
266 : loss:  0.89712954 	 acc:  0.46
268 : loss:  0.9453044 	 acc:  0.59
270 : loss:  0.9710506 	 acc:  0.49
272 : loss:  0.9289398 	 acc:  0.57
274 : loss:  0.948051 	 acc:  0.45
276 : loss:  1.0142136 	 acc:  0.48
278 : loss:  1.0254432 	 acc:  0.51
280 : loss:  0.9134311 	 acc:  0.51
282 : loss:  0.93683875 	 acc:  0.5
284 : loss:  1.1347271 	 acc:  0.42
286 : loss:  0.90368056 	 acc:  0.59
288 : loss:  1.0021272 	 acc:  0.39
290 : loss:  0.9456737 	 acc:  0.45
292 : loss:  0.96632636 	 acc:  0.46
294 : loss:  0.9254915 	 acc:  0.5
296 : loss:  0.8796959 	 acc:  0.47
298 : loss:  0.99941957 	 acc:  0.46
300 : loss:  1.0201949 	 acc:  0.48
302 : loss:  0.98076874 	 acc:  0.5
304 : loss:  0.92642194 	 acc:  0.48
306 : loss:  0.9374152 	 acc:  0.51
308 : loss:  1.0147432 	 acc:  0.48
310 : loss:  0.95797575 	 acc:  0.54
312 : loss:  0.89282745 	 acc:  0.51
314 : loss:  0.9402735 	 acc:  0.51
316 : loss:  0.973721 	 acc:  0.59
318 : loss:  0.965252 	 acc:  0.54
320 : loss:  0.9471462 	 acc:  0.56
322 : loss:  1.0215935 	 acc:  0.42
324 : loss:  0.9437417 	 acc:  0.56
326 : loss:  0.9054781 	 acc:  0.58
328 : loss:  0.9000639 	 acc:  0.59
330 : loss:  0.91847044 	 acc:  0.5
332 : loss:  0.881815 	 acc:  0.53
334 : loss:  1.0134197 	 acc:  0.47
336 : loss:  0.8949278 	 acc:  0.61
338 : loss:  0.90814394 	 acc:  0.46
340 : loss:  0.96046185 	 acc:  0.51
342 : loss:  0.9196876 	 acc:  0.47
344 : loss:  0.9215674 	 acc:  0.48
346 : loss:  0.97107816 	 acc:  0.55
348 : loss:  0.92528594 	 acc:  0.45
350 : loss:  0.9261295 	 acc:  0.54
352 : loss:  0.96999896 	 acc:  0.46
354 : loss:  0.95941234 	 acc:  0.5
356 : loss:  0.93511444 	 acc:  0.5
358 : loss:  0.9706092 	 acc:  0.39
360 : loss:  0.98523027 	 acc:  0.49
362 : loss:  0.9714893 	 acc:  0.44
364 : loss:  0.91563815 	 acc:  0.53
366 : loss:  0.9782439 	 acc:  0.47
368 : loss:  0.9174835 	 acc:  0.51
370 : loss:  0.9127579 	 acc:  0.46

Saving...
saved to models/CNN/pretrained_model.ckpt-372

100 	 [64.32736135 56.52520933 25.34507256]
0 	val accuracy:  0.5106 	 f_! score:  [0.64327361 0.56525209 0.25345073]

372 : loss:  0.9081575 	 acc:  0.5
374 : loss:  0.9096189 	 acc:  0.5
376 : loss:  0.9305973 	 acc:  0.53
378 : loss:  0.9809494 	 acc:  0.47
380 : loss:  0.97040373 	 acc:  0.51
382 : loss:  1.0161576 	 acc:  0.48
384 : loss:  0.826797 	 acc:  0.6
386 : loss:  0.9015092 	 acc:  0.52
388 : loss:  0.87027633 	 acc:  0.57
390 : loss:  0.92095983 	 acc:  0.57
392 : loss:  0.9337749 	 acc:  0.53
394 : loss:  0.96887153 	 acc:  0.54
396 : loss:  0.87442476 	 acc:  0.58
398 : loss:  0.97060037 	 acc:  0.44
400 : loss:  0.9155236 	 acc:  0.52
402 : loss:  0.96300554 	 acc:  0.48
404 : loss:  0.97862756 	 acc:  0.49
406 : loss:  0.9269855 	 acc:  0.57
408 : loss:  1.0030885 	 acc:  0.5
410 : loss:  0.9296505 	 acc:  0.53
412 : loss:  0.90859026 	 acc:  0.54
414 : loss:  0.8670339 	 acc:  0.52
416 : loss:  1.0009307 	 acc:  0.44
418 : loss:  0.88782483 	 acc:  0.52
420 : loss:  0.9391409 	 acc:  0.52
422 : loss:  0.909278 	 acc:  0.53
424 : loss:  0.9143901 	 acc:  0.45
426 : loss:  0.9262281 	 acc:  0.56
428 : loss:  1.001122 	 acc:  0.49
430 : loss:  0.94582987 	 acc:  0.52
432 : loss:  0.9464449 	 acc:  0.5
434 : loss:  0.9304013 	 acc:  0.54
436 : loss:  0.9239906 	 acc:  0.53
438 : loss:  0.860968 	 acc:  0.52
440 : loss:  0.90989625 	 acc:  0.51
442 : loss:  0.79954743 	 acc:  0.61
444 : loss:  0.87072045 	 acc:  0.58
446 : loss:  0.8905092 	 acc:  0.53
448 : loss:  0.8279225 	 acc:  0.65
450 : loss:  0.8500644 	 acc:  0.57
452 : loss:  0.9302517 	 acc:  0.57
454 : loss:  0.7919139 	 acc:  0.63
456 : loss:  0.8801913 	 acc:  0.48
458 : loss:  1.0387678 	 acc:  0.51
460 : loss:  0.85621244 	 acc:  0.55
462 : loss:  0.88880074 	 acc:  0.53
464 : loss:  0.8953929 	 acc:  0.57
466 : loss:  0.8389573 	 acc:  0.62
468 : loss:  0.8780657 	 acc:  0.56
470 : loss:  0.9402438 	 acc:  0.53
472 : loss:  0.91103625 	 acc:  0.57
474 : loss:  0.8698651 	 acc:  0.57
476 : loss:  0.9102021 	 acc:  0.52
478 : loss:  1.082891 	 acc:  0.36
480 : loss:  0.95486665 	 acc:  0.52
482 : loss:  0.9799854 	 acc:  0.56
484 : loss:  0.9380663 	 acc:  0.58
486 : loss:  0.89361876 	 acc:  0.49
488 : loss:  0.97099113 	 acc:  0.46
490 : loss:  0.97011137 	 acc:  0.53
492 : loss:  0.9262617 	 acc:  0.5
494 : loss:  0.8429386 	 acc:  0.6
496 : loss:  0.86270934 	 acc:  0.62
498 : loss:  1.0464119 	 acc:  0.47
500 : loss:  0.9012851 	 acc:  0.56
502 : loss:  0.83479697 	 acc:  0.6
504 : loss:  0.9506113 	 acc:  0.51
506 : loss:  0.86553466 	 acc:  0.58
508 : loss:  0.90130645 	 acc:  0.58
510 : loss:  0.8420716 	 acc:  0.62
512 : loss:  0.85460275 	 acc:  0.58
514 : loss:  0.8983399 	 acc:  0.6
516 : loss:  0.8907544 	 acc:  0.58
518 : loss:  0.9559468 	 acc:  0.56
520 : loss:  1.0616161 	 acc:  0.41
522 : loss:  0.8975888 	 acc:  0.55
524 : loss:  0.85522944 	 acc:  0.63
526 : loss:  0.89546776 	 acc:  0.53
528 : loss:  0.8896376 	 acc:  0.5
530 : loss:  0.9280867 	 acc:  0.59
532 : loss:  0.8885472 	 acc:  0.51
534 : loss:  0.90558547 	 acc:  0.49
536 : loss:  0.9479361 	 acc:  0.55
538 : loss:  0.91959685 	 acc:  0.56
540 : loss:  0.86173326 	 acc:  0.58
542 : loss:  0.9525752 	 acc:  0.55
544 : loss:  0.8011797 	 acc:  0.63
546 : loss:  0.8316921 	 acc:  0.65
548 : loss:  0.96934974 	 acc:  0.53
550 : loss:  0.8747597 	 acc:  0.55
552 : loss:  0.8199768 	 acc:  0.61
554 : loss:  0.83529943 	 acc:  0.6
556 : loss:  0.82696474 	 acc:  0.59
558 : loss:  0.8823212 	 acc:  0.53
560 : loss:  0.91832536 	 acc:  0.52
562 : loss:  0.9322055 	 acc:  0.52
564 : loss:  0.960534 	 acc:  0.5
566 : loss:  0.89944047 	 acc:  0.54
568 : loss:  0.8408171 	 acc:  0.61
570 : loss:  0.85053134 	 acc:  0.62
572 : loss:  0.8322381 	 acc:  0.56
574 : loss:  0.8551017 	 acc:  0.6
576 : loss:  0.8886051 	 acc:  0.64
578 : loss:  0.92185885 	 acc:  0.49
580 : loss:  0.9404361 	 acc:  0.57
582 : loss:  0.9277946 	 acc:  0.56
584 : loss:  0.84725374 	 acc:  0.61
586 : loss:  0.79584265 	 acc:  0.56
588 : loss:  0.8256631 	 acc:  0.63
590 : loss:  0.8632557 	 acc:  0.54
592 : loss:  0.82090294 	 acc:  0.65
594 : loss:  0.85198057 	 acc:  0.62
596 : loss:  0.85063696 	 acc:  0.54
598 : loss:  0.9178148 	 acc:  0.57
600 : loss:  0.8644324 	 acc:  0.58
602 : loss:  0.8635515 	 acc:  0.54
604 : loss:  0.92781943 	 acc:  0.54
606 : loss:  0.88588 	 acc:  0.56
608 : loss:  0.867243 	 acc:  0.53
610 : loss:  0.8085177 	 acc:  0.59
612 : loss:  0.8474707 	 acc:  0.65
614 : loss:  0.90789586 	 acc:  0.6
616 : loss:  0.80413294 	 acc:  0.69
618 : loss:  0.754678 	 acc:  0.67
620 : loss:  0.82421243 	 acc:  0.59
622 : loss:  0.88987505 	 acc:  0.62
624 : loss:  0.9680936 	 acc:  0.59
626 : loss:  0.89411646 	 acc:  0.6
628 : loss:  0.8083112 	 acc:  0.64
630 : loss:  1.0028781 	 acc:  0.5
632 : loss:  0.79049385 	 acc:  0.61
634 : loss:  0.83648574 	 acc:  0.53
636 : loss:  0.99083865 	 acc:  0.51
638 : loss:  1.0003421 	 acc:  0.49
640 : loss:  0.9104013 	 acc:  0.54
642 : loss:  1.0190954 	 acc:  0.44
644 : loss:  0.81362396 	 acc:  0.57
646 : loss:  0.8107297 	 acc:  0.63
648 : loss:  0.90002805 	 acc:  0.5
650 : loss:  1.0255957 	 acc:  0.5
652 : loss:  0.90643305 	 acc:  0.58
654 : loss:  0.9246654 	 acc:  0.49
656 : loss:  0.7848748 	 acc:  0.64
658 : loss:  0.8873785 	 acc:  0.54
660 : loss:  0.8874139 	 acc:  0.58
662 : loss:  0.8058221 	 acc:  0.61
664 : loss:  0.88346726 	 acc:  0.57
666 : loss:  0.8337734 	 acc:  0.56
668 : loss:  0.90183413 	 acc:  0.57
670 : loss:  0.87568533 	 acc:  0.58
672 : loss:  0.8388165 	 acc:  0.61
674 : loss:  0.8137661 	 acc:  0.63
676 : loss:  0.8129485 	 acc:  0.65
678 : loss:  0.8436917 	 acc:  0.61
680 : loss:  0.88687044 	 acc:  0.51
682 : loss:  0.8824352 	 acc:  0.58
684 : loss:  0.928394 	 acc:  0.55
686 : loss:  0.8038477 	 acc:  0.64
688 : loss:  0.9403749 	 acc:  0.56
690 : loss:  0.8684588 	 acc:  0.53
692 : loss:  0.7060805 	 acc:  0.64
694 : loss:  0.84159225 	 acc:  0.66
696 : loss:  0.87003356 	 acc:  0.62
698 : loss:  0.7182077 	 acc:  0.75
700 : loss:  0.7927782 	 acc:  0.63
702 : loss:  0.8125273 	 acc:  0.62
704 : loss:  0.88022214 	 acc:  0.59
706 : loss:  0.8733539 	 acc:  0.61
708 : loss:  0.8808094 	 acc:  0.55
710 : loss:  0.90069145 	 acc:  0.54
712 : loss:  0.812367 	 acc:  0.64
714 : loss:  0.91659266 	 acc:  0.51
716 : loss:  0.82050997 	 acc:  0.61
718 : loss:  0.8642617 	 acc:  0.63
720 : loss:  0.8317032 	 acc:  0.62
722 : loss:  0.8998156 	 acc:  0.57
724 : loss:  0.9017346 	 acc:  0.58
726 : loss:  0.92745733 	 acc:  0.53
728 : loss:  0.8774108 	 acc:  0.55
730 : loss:  0.95379174 	 acc:  0.54
732 : loss:  0.84860367 	 acc:  0.68
734 : loss:  0.8691164 	 acc:  0.58
736 : loss:  0.8028801 	 acc:  0.65
738 : loss:  0.82969373 	 acc:  0.58
740 : loss:  0.8122717 	 acc:  0.68
742 : loss:  0.86995363 	 acc:  0.6

Saving...
saved to models/CNN/pretrained_model.ckpt-744

100 	 [68.50528895 57.70197881 52.62171882]
1 	val accuracy:  0.6038 	 f_! score:  [0.68505289 0.57701979 0.52621719]

744 : loss:  0.77374667 	 acc:  0.65
746 : loss:  0.83305585 	 acc:  0.61
748 : loss:  0.9552692 	 acc:  0.49
750 : loss:  0.8496785 	 acc:  0.64
752 : loss:  0.84221375 	 acc:  0.56
754 : loss:  0.81734705 	 acc:  0.66
756 : loss:  0.8577801 	 acc:  0.63
758 : loss:  1.0020677 	 acc:  0.48
760 : loss:  0.7368444 	 acc:  0.65
762 : loss:  0.8219046 	 acc:  0.61
764 : loss:  0.82240045 	 acc:  0.61
766 : loss:  0.8341083 	 acc:  0.61
768 : loss:  0.6994909 	 acc:  0.72
770 : loss:  0.75519043 	 acc:  0.68
772 : loss:  0.77667946 	 acc:  0.69
774 : loss:  0.8851924 	 acc:  0.54
776 : loss:  0.69760215 	 acc:  0.71
778 : loss:  0.92950976 	 acc:  0.57
780 : loss:  0.86517394 	 acc:  0.61
782 : loss:  0.81790024 	 acc:  0.6
784 : loss:  0.7819336 	 acc:  0.64
786 : loss:  0.7839811 	 acc:  0.56
788 : loss:  0.80750895 	 acc:  0.63
790 : loss:  0.88609433 	 acc:  0.58
792 : loss:  0.8442976 	 acc:  0.57
794 : loss:  0.87122416 	 acc:  0.61
796 : loss:  0.8660303 	 acc:  0.62
798 : loss:  0.8178623 	 acc:  0.62
800 : loss:  0.79119855 	 acc:  0.65
802 : loss:  0.9197857 	 acc:  0.52
804 : loss:  0.8534472 	 acc:  0.61
806 : loss:  0.7830206 	 acc:  0.67
808 : loss:  0.8361778 	 acc:  0.64
810 : loss:  0.89936274 	 acc:  0.52
812 : loss:  0.7595559 	 acc:  0.65
814 : loss:  0.85212445 	 acc:  0.63
816 : loss:  0.92969686 	 acc:  0.52
818 : loss:  0.98224264 	 acc:  0.5
820 : loss:  0.8738605 	 acc:  0.58
822 : loss:  0.82175064 	 acc:  0.59
824 : loss:  0.8667904 	 acc:  0.55
826 : loss:  0.88228154 	 acc:  0.54
828 : loss:  0.84839284 	 acc:  0.59
830 : loss:  0.9394901 	 acc:  0.62
832 : loss:  0.82861114 	 acc:  0.55
834 : loss:  0.84633833 	 acc:  0.58
836 : loss:  0.7012921 	 acc:  0.62
838 : loss:  0.8067998 	 acc:  0.63
840 : loss:  0.80770296 	 acc:  0.65
842 : loss:  0.8951127 	 acc:  0.58
844 : loss:  0.8325413 	 acc:  0.6
846 : loss:  0.7621773 	 acc:  0.63
848 : loss:  0.8312052 	 acc:  0.65
850 : loss:  0.8151213 	 acc:  0.59
852 : loss:  1.0131297 	 acc:  0.49
854 : loss:  0.8020887 	 acc:  0.62
856 : loss:  0.8877613 	 acc:  0.48
858 : loss:  0.8461939 	 acc:  0.58
860 : loss:  0.7871478 	 acc:  0.64
862 : loss:  0.86235595 	 acc:  0.59
864 : loss:  0.93310964 	 acc:  0.55
866 : loss:  0.86285615 	 acc:  0.66
868 : loss:  0.7907742 	 acc:  0.61
870 : loss:  0.7826836 	 acc:  0.64
872 : loss:  0.8631354 	 acc:  0.59
874 : loss:  0.83260894 	 acc:  0.61
876 : loss:  0.77349454 	 acc:  0.64
878 : loss:  0.7984027 	 acc:  0.59
880 : loss:  0.8261962 	 acc:  0.61
882 : loss:  0.8426761 	 acc:  0.63
884 : loss:  0.8365268 	 acc:  0.62
886 : loss:  0.9245993 	 acc:  0.53
888 : loss:  0.8908447 	 acc:  0.57
890 : loss:  0.7636046 	 acc:  0.64
892 : loss:  0.83004206 	 acc:  0.69
894 : loss:  0.8768041 	 acc:  0.6
896 : loss:  0.9173586 	 acc:  0.55
898 : loss:  0.91489196 	 acc:  0.52
900 : loss:  0.79527766 	 acc:  0.66
902 : loss:  0.8476578 	 acc:  0.57
904 : loss:  0.747407 	 acc:  0.7
906 : loss:  0.85238105 	 acc:  0.6
908 : loss:  0.934782 	 acc:  0.54
910 : loss:  0.7671852 	 acc:  0.62
912 : loss:  0.80723906 	 acc:  0.68
914 : loss:  0.885015 	 acc:  0.56
916 : loss:  0.8240889 	 acc:  0.65
918 : loss:  0.81072825 	 acc:  0.62
920 : loss:  0.7769696 	 acc:  0.65
922 : loss:  0.89202833 	 acc:  0.56
924 : loss:  0.784943 	 acc:  0.62
926 : loss:  0.88396955 	 acc:  0.59
928 : loss:  0.8122174 	 acc:  0.63
930 : loss:  0.8567508 	 acc:  0.58
932 : loss:  0.8453468 	 acc:  0.6
934 : loss:  0.69056433 	 acc:  0.71
936 : loss:  0.88221 	 acc:  0.55
938 : loss:  0.8271757 	 acc:  0.64
940 : loss:  0.71981406 	 acc:  0.64
942 : loss:  0.7409431 	 acc:  0.67
944 : loss:  0.8555722 	 acc:  0.61
946 : loss:  0.8904457 	 acc:  0.6
948 : loss:  0.8564172 	 acc:  0.62
950 : loss:  0.75033605 	 acc:  0.68
952 : loss:  0.84502006 	 acc:  0.61
954 : loss:  0.78514856 	 acc:  0.66
956 : loss:  0.8258511 	 acc:  0.64
958 : loss:  0.78171515 	 acc:  0.62
960 : loss:  0.88057727 	 acc:  0.55
962 : loss:  0.82156473 	 acc:  0.59
964 : loss:  0.8154652 	 acc:  0.68
966 : loss:  0.84000146 	 acc:  0.64
968 : loss:  0.7582264 	 acc:  0.65
970 : loss:  0.7312028 	 acc:  0.68
972 : loss:  0.7873487 	 acc:  0.65
974 : loss:  0.7435509 	 acc:  0.68
976 : loss:  0.87039316 	 acc:  0.56
978 : loss:  0.8134981 	 acc:  0.62
980 : loss:  0.9349903 	 acc:  0.53
982 : loss:  0.7333622 	 acc:  0.67
984 : loss:  0.7641948 	 acc:  0.7
986 : loss:  0.9148957 	 acc:  0.62
988 : loss:  0.9216813 	 acc:  0.59
990 : loss:  0.9296339 	 acc:  0.55
992 : loss:  0.6961281 	 acc:  0.69
994 : loss:  0.8144831 	 acc:  0.62
996 : loss:  0.7989418 	 acc:  0.67
998 : loss:  0.84183127 	 acc:  0.6
1000 : loss:  0.8474934 	 acc:  0.59
1002 : loss:  0.77054864 	 acc:  0.65
1004 : loss:  0.83405006 	 acc:  0.6
1006 : loss:  0.85200995 	 acc:  0.59
1008 : loss:  0.79661834 	 acc:  0.64
1010 : loss:  0.86069036 	 acc:  0.56
1012 : loss:  0.94245553 	 acc:  0.58
1014 : loss:  0.6893444 	 acc:  0.72
1016 : loss:  0.8986786 	 acc:  0.65
1018 : loss:  0.95564985 	 acc:  0.49
1020 : loss:  0.95075315 	 acc:  0.51
1022 : loss:  0.9075187 	 acc:  0.59
1024 : loss:  0.81519395 	 acc:  0.6
1026 : loss:  0.82677305 	 acc:  0.63
1028 : loss:  0.82167107 	 acc:  0.6
1030 : loss:  0.86420625 	 acc:  0.55
1032 : loss:  0.8735617 	 acc:  0.55
1034 : loss:  0.9461737 	 acc:  0.5
1036 : loss:  0.8606763 	 acc:  0.61
1038 : loss:  0.86331373 	 acc:  0.55
1040 : loss:  0.84994787 	 acc:  0.58
1042 : loss:  0.7487493 	 acc:  0.67
1044 : loss:  0.73445916 	 acc:  0.64
1046 : loss:  0.8169713 	 acc:  0.64
1048 : loss:  0.7609634 	 acc:  0.67
1050 : loss:  0.9614962 	 acc:  0.52
1052 : loss:  0.8581813 	 acc:  0.65
1054 : loss:  0.83414596 	 acc:  0.64
1056 : loss:  0.89733475 	 acc:  0.58
1058 : loss:  0.91310155 	 acc:  0.55
1060 : loss:  0.7976646 	 acc:  0.65
1062 : loss:  0.85578704 	 acc:  0.54
1064 : loss:  0.78022057 	 acc:  0.67
1066 : loss:  0.702982 	 acc:  0.7
1068 : loss:  0.8506886 	 acc:  0.55
1070 : loss:  0.78389466 	 acc:  0.63
1072 : loss:  0.78829527 	 acc:  0.62
1074 : loss:  0.845927 	 acc:  0.63
1076 : loss:  0.8194359 	 acc:  0.64
1078 : loss:  0.8197998 	 acc:  0.65
1080 : loss:  0.80304986 	 acc:  0.61
1082 : loss:  0.72830606 	 acc:  0.65
1084 : loss:  0.8211676 	 acc:  0.62
1086 : loss:  0.79561895 	 acc:  0.58
1088 : loss:  0.79021215 	 acc:  0.63
1090 : loss:  0.94944024 	 acc:  0.53
1092 : loss:  0.75491494 	 acc:  0.63
1094 : loss:  0.818894 	 acc:  0.6
1096 : loss:  0.83414006 	 acc:  0.55
1098 : loss:  0.82527864 	 acc:  0.61
1100 : loss:  0.7944417 	 acc:  0.66
1102 : loss:  0.80269873 	 acc:  0.66
1104 : loss:  0.7770367 	 acc:  0.63
1106 : loss:  0.8948563 	 acc:  0.63
1108 : loss:  0.92453504 	 acc:  0.57
1110 : loss:  0.70417285 	 acc:  0.71
1112 : loss:  0.8897979 	 acc:  0.57
1114 : loss:  0.8452514 	 acc:  0.59

Saving...
saved to models/CNN/pretrained_model.ckpt-1116

100 	 [65.27881592 63.03561383 54.88661992]
2 	val accuracy:  0.6155 	 f_! score:  [0.65278816 0.63035614 0.5488662 ]

1116 : loss:  0.8819025 	 acc:  0.56
1118 : loss:  0.7298887 	 acc:  0.66
1120 : loss:  0.83330613 	 acc:  0.55
1122 : loss:  0.8070267 	 acc:  0.66
1124 : loss:  0.7544679 	 acc:  0.62
1126 : loss:  0.8385389 	 acc:  0.63
1128 : loss:  0.74374247 	 acc:  0.74
1130 : loss:  0.79110354 	 acc:  0.64
1132 : loss:  0.7996497 	 acc:  0.64
1134 : loss:  0.885627 	 acc:  0.61
1136 : loss:  0.73888427 	 acc:  0.69
1138 : loss:  0.8202299 	 acc:  0.59
1140 : loss:  0.7248254 	 acc:  0.68
1142 : loss:  0.77166396 	 acc:  0.72
1144 : loss:  0.8516573 	 acc:  0.52
1146 : loss:  0.69029206 	 acc:  0.69
1148 : loss:  0.76369387 	 acc:  0.63
1150 : loss:  0.773853 	 acc:  0.68
1152 : loss:  0.77890044 	 acc:  0.65
1154 : loss:  0.77502924 	 acc:  0.63
1156 : loss:  0.80407774 	 acc:  0.58
1158 : loss:  0.78401613 	 acc:  0.64
1160 : loss:  0.75554603 	 acc:  0.6
1162 : loss:  0.7735113 	 acc:  0.65
1164 : loss:  0.8402134 	 acc:  0.63
1166 : loss:  0.81602377 	 acc:  0.67
1168 : loss:  0.81769633 	 acc:  0.59
1170 : loss:  0.6925066 	 acc:  0.71
1172 : loss:  0.75690126 	 acc:  0.58
1174 : loss:  0.6505266 	 acc:  0.75
1176 : loss:  0.67694336 	 acc:  0.72
1178 : loss:  0.75653535 	 acc:  0.68
1180 : loss:  0.7894861 	 acc:  0.62
1182 : loss:  0.8096793 	 acc:  0.66
1184 : loss:  0.8113296 	 acc:  0.65
1186 : loss:  0.9121904 	 acc:  0.5
1188 : loss:  0.7399499 	 acc:  0.66
1190 : loss:  0.81898415 	 acc:  0.6
1192 : loss:  0.8137401 	 acc:  0.67
1194 : loss:  0.74544954 	 acc:  0.69
1196 : loss:  0.68074286 	 acc:  0.7
1198 : loss:  0.9265905 	 acc:  0.54
1200 : loss:  0.82586837 	 acc:  0.66
1202 : loss:  0.8033112 	 acc:  0.63
1204 : loss:  0.7709606 	 acc:  0.68
1206 : loss:  0.81275046 	 acc:  0.61
1208 : loss:  0.7225683 	 acc:  0.71
1210 : loss:  0.74053437 	 acc:  0.66
1212 : loss:  0.7885782 	 acc:  0.64
1214 : loss:  0.7512414 	 acc:  0.66
1216 : loss:  0.89145195 	 acc:  0.55
1218 : loss:  0.7989795 	 acc:  0.67
1220 : loss:  0.8470304 	 acc:  0.63
1222 : loss:  0.6934192 	 acc:  0.69
1224 : loss:  0.85938156 	 acc:  0.62
1226 : loss:  0.7490872 	 acc:  0.67
1228 : loss:  0.72336984 	 acc:  0.67
1230 : loss:  0.6694192 	 acc:  0.73
1232 : loss:  0.7303078 	 acc:  0.68
1234 : loss:  0.7255045 	 acc:  0.72
1236 : loss:  0.8103487 	 acc:  0.59
1238 : loss:  0.78859353 	 acc:  0.68
1240 : loss:  0.9235549 	 acc:  0.6
1242 : loss:  0.9044381 	 acc:  0.54
1244 : loss:  0.8221945 	 acc:  0.6
1246 : loss:  0.80646515 	 acc:  0.63
1248 : loss:  0.79873705 	 acc:  0.61
1250 : loss:  0.83025634 	 acc:  0.68
1252 : loss:  0.88815963 	 acc:  0.64
1254 : loss:  0.72111297 	 acc:  0.64
1256 : loss:  0.74489534 	 acc:  0.65
1258 : loss:  0.82140845 	 acc:  0.65
1260 : loss:  0.856714 	 acc:  0.62
1262 : loss:  0.7757457 	 acc:  0.57
1264 : loss:  0.88231474 	 acc:  0.61
1266 : loss:  0.83628565 	 acc:  0.61
1268 : loss:  0.74895614 	 acc:  0.68
1270 : loss:  0.90212476 	 acc:  0.51
1272 : loss:  0.7248797 	 acc:  0.74
1274 : loss:  0.7340981 	 acc:  0.64
1276 : loss:  0.7508327 	 acc:  0.67
1278 : loss:  0.87512803 	 acc:  0.57
1280 : loss:  0.86023575 	 acc:  0.63
1282 : loss:  0.8792866 	 acc:  0.56
1284 : loss:  0.77742285 	 acc:  0.64
1286 : loss:  0.878004 	 acc:  0.56
1288 : loss:  0.82158136 	 acc:  0.59
1290 : loss:  0.7487376 	 acc:  0.65
1292 : loss:  0.72993684 	 acc:  0.68
1294 : loss:  0.8527601 	 acc:  0.59
1296 : loss:  0.90252423 	 acc:  0.59
1298 : loss:  0.6614446 	 acc:  0.74
1300 : loss:  0.7766159 	 acc:  0.63
1302 : loss:  0.8826763 	 acc:  0.56
1304 : loss:  0.9040918 	 acc:  0.52
1306 : loss:  0.7404886 	 acc:  0.63
1308 : loss:  0.8238902 	 acc:  0.65
1310 : loss:  0.8942398 	 acc:  0.58
1312 : loss:  0.85489845 	 acc:  0.55
1314 : loss:  0.85848594 	 acc:  0.58
1316 : loss:  0.73120356 	 acc:  0.67
1318 : loss:  0.8940812 	 acc:  0.61
1320 : loss:  0.8276673 	 acc:  0.57
1322 : loss:  0.76623183 	 acc:  0.63
1324 : loss:  0.8252869 	 acc:  0.68
1326 : loss:  0.7890061 	 acc:  0.66
1328 : loss:  0.6706817 	 acc:  0.69
1330 : loss:  0.77671903 	 acc:  0.65
1332 : loss:  0.84069043 	 acc:  0.57
1334 : loss:  0.8310498 	 acc:  0.61
1336 : loss:  0.8238558 	 acc:  0.62
1338 : loss:  0.71461886 	 acc:  0.66
1340 : loss:  0.68152183 	 acc:  0.63
1342 : loss:  0.78658223 	 acc:  0.62
1344 : loss:  0.794041 	 acc:  0.64
1346 : loss:  0.84003884 	 acc:  0.6
1348 : loss:  0.86367583 	 acc:  0.62
1350 : loss:  0.727524 	 acc:  0.65
1352 : loss:  0.81069666 	 acc:  0.64
1354 : loss:  0.6705406 	 acc:  0.7
1356 : loss:  0.75142264 	 acc:  0.7
1358 : loss:  0.7783099 	 acc:  0.57
1360 : loss:  0.8280666 	 acc:  0.59
1362 : loss:  0.7542939 	 acc:  0.63
1364 : loss:  0.7294047 	 acc:  0.68
1366 : loss:  0.7385231 	 acc:  0.66
1368 : loss:  0.67691123 	 acc:  0.71
1370 : loss:  0.78681874 	 acc:  0.6
1372 : loss:  0.8990127 	 acc:  0.58
1374 : loss:  0.8197267 	 acc:  0.65
1376 : loss:  0.836409 	 acc:  0.66
1378 : loss:  0.7808194 	 acc:  0.66
1380 : loss:  0.68057853 	 acc:  0.71
1382 : loss:  0.74452853 	 acc:  0.67
1384 : loss:  0.7254498 	 acc:  0.69
1386 : loss:  0.8062321 	 acc:  0.63
1388 : loss:  0.8194393 	 acc:  0.65
1390 : loss:  0.8608435 	 acc:  0.59
1392 : loss:  0.73814017 	 acc:  0.64
1394 : loss:  0.8676872 	 acc:  0.63
1396 : loss:  0.77341753 	 acc:  0.57
1398 : loss:  0.78520375 	 acc:  0.64
1400 : loss:  0.83543986 	 acc:  0.57
1402 : loss:  0.771418 	 acc:  0.69
1404 : loss:  0.7468698 	 acc:  0.66
1406 : loss:  0.9082116 	 acc:  0.57
1408 : loss:  0.72911644 	 acc:  0.68
1410 : loss:  0.7363647 	 acc:  0.62
1412 : loss:  0.9247543 	 acc:  0.55
1414 : loss:  0.80407804 	 acc:  0.58
1416 : loss:  0.85106796 	 acc:  0.59
1418 : loss:  0.8397432 	 acc:  0.58
1420 : loss:  0.93138015 	 acc:  0.5
1422 : loss:  0.74429053 	 acc:  0.69
1424 : loss:  0.7553957 	 acc:  0.69
1426 : loss:  0.8062916 	 acc:  0.64
1428 : loss:  0.72610164 	 acc:  0.68
1430 : loss:  0.79602045 	 acc:  0.62
1432 : loss:  0.75056875 	 acc:  0.67
1434 : loss:  0.8397888 	 acc:  0.61
1436 : loss:  0.75454056 	 acc:  0.72
1438 : loss:  0.8157667 	 acc:  0.64
1440 : loss:  0.7706474 	 acc:  0.63
1442 : loss:  0.80142075 	 acc:  0.62
1444 : loss:  0.82653344 	 acc:  0.58
1446 : loss:  0.8775401 	 acc:  0.62
1448 : loss:  0.7252522 	 acc:  0.7
1450 : loss:  0.8373384 	 acc:  0.56
1452 : loss:  0.87266 	 acc:  0.55
1454 : loss:  0.76420844 	 acc:  0.67
1456 : loss:  0.8266108 	 acc:  0.69
1458 : loss:  0.8922018 	 acc:  0.57
1460 : loss:  0.7357463 	 acc:  0.67
1462 : loss:  0.7367837 	 acc:  0.64
1464 : loss:  0.8149482 	 acc:  0.67
1466 : loss:  0.9259218 	 acc:  0.55
1468 : loss:  0.797005 	 acc:  0.66
1470 : loss:  0.70501906 	 acc:  0.69
1472 : loss:  0.8625031 	 acc:  0.52
1474 : loss:  0.81310946 	 acc:  0.56
1476 : loss:  0.6864874 	 acc:  0.68
1478 : loss:  0.81445956 	 acc:  0.62
1480 : loss:  0.8546829 	 acc:  0.59
1482 : loss:  0.7336767 	 acc:  0.65
1484 : loss:  0.74116427 	 acc:  0.68
1486 : loss:  0.8710025 	 acc:  0.62

Saving...
saved to models/CNN/pretrained_model.ckpt-1488

100 	 [70.28447941 63.1123877  55.63965625]
3 	val accuracy:  0.6376 	 f_! score:  [0.70284479 0.63112388 0.55639656]

1488 : loss:  0.76507974 	 acc:  0.67
1490 : loss:  0.80001473 	 acc:  0.66
1492 : loss:  0.70012033 	 acc:  0.65
1494 : loss:  0.9075574 	 acc:  0.59
1496 : loss:  0.7170412 	 acc:  0.64
1498 : loss:  0.76491815 	 acc:  0.66
1500 : loss:  0.8238933 	 acc:  0.68
1502 : loss:  0.7769543 	 acc:  0.63
1504 : loss:  0.7584652 	 acc:  0.63
1506 : loss:  0.7995213 	 acc:  0.59
1508 : loss:  0.74760497 	 acc:  0.64
1510 : loss:  0.74127114 	 acc:  0.69
1512 : loss:  0.75796735 	 acc:  0.63
1514 : loss:  0.94948435 	 acc:  0.53
1516 : loss:  0.7260075 	 acc:  0.74
1518 : loss:  0.6775986 	 acc:  0.72
1520 : loss:  0.73516566 	 acc:  0.73
1522 : loss:  0.7095849 	 acc:  0.72
1524 : loss:  0.7185418 	 acc:  0.69
1526 : loss:  0.75862694 	 acc:  0.67
1528 : loss:  0.8449938 	 acc:  0.62
1530 : loss:  0.9207826 	 acc:  0.53
1532 : loss:  0.8104133 	 acc:  0.53
1534 : loss:  0.8485614 	 acc:  0.6
1536 : loss:  0.80601954 	 acc:  0.62
1538 : loss:  0.821324 	 acc:  0.61
1540 : loss:  0.8448896 	 acc:  0.56
1542 : loss:  0.74643826 	 acc:  0.66
1544 : loss:  0.8840341 	 acc:  0.64
1546 : loss:  0.7542988 	 acc:  0.68
1548 : loss:  0.77089965 	 acc:  0.69
1550 : loss:  0.85182226 	 acc:  0.6
1552 : loss:  0.80821884 	 acc:  0.59
1554 : loss:  0.87322176 	 acc:  0.58
1556 : loss:  0.8040742 	 acc:  0.69
1558 : loss:  0.82159454 	 acc:  0.66
1560 : loss:  0.80813414 	 acc:  0.64
1562 : loss:  0.7807702 	 acc:  0.59
1564 : loss:  0.831226 	 acc:  0.68
1566 : loss:  0.8244155 	 acc:  0.62
1568 : loss:  0.76298493 	 acc:  0.63
1570 : loss:  0.90880966 	 acc:  0.56
1572 : loss:  0.7424136 	 acc:  0.67
1574 : loss:  0.8297503 	 acc:  0.69
1576 : loss:  0.75007707 	 acc:  0.67
1578 : loss:  0.66896147 	 acc:  0.7
1580 : loss:  0.7152363 	 acc:  0.7
1582 : loss:  0.6790095 	 acc:  0.67
1584 : loss:  0.82276773 	 acc:  0.56
1586 : loss:  0.73381525 	 acc:  0.7
1588 : loss:  0.67747384 	 acc:  0.72
1590 : loss:  0.84054303 	 acc:  0.59
1592 : loss:  0.74617696 	 acc:  0.69
1594 : loss:  0.7711113 	 acc:  0.6
1596 : loss:  0.7369859 	 acc:  0.61
1598 : loss:  0.75618285 	 acc:  0.65
1600 : loss:  0.67702556 	 acc:  0.71
1602 : loss:  0.6985435 	 acc:  0.71
1604 : loss:  0.8262329 	 acc:  0.64
1606 : loss:  0.7789996 	 acc:  0.57
1608 : loss:  0.7915797 	 acc:  0.64
1610 : loss:  0.72190076 	 acc:  0.71
1612 : loss:  0.8714569 	 acc:  0.65
1614 : loss:  0.83062935 	 acc:  0.6
1616 : loss:  0.833767 	 acc:  0.61
1618 : loss:  0.6467416 	 acc:  0.73
1620 : loss:  0.8770983 	 acc:  0.53
1622 : loss:  0.7495321 	 acc:  0.63
1624 : loss:  0.87073046 	 acc:  0.61
1626 : loss:  0.6909679 	 acc:  0.69
1628 : loss:  0.7253318 	 acc:  0.64
1630 : loss:  0.7861547 	 acc:  0.64
1632 : loss:  0.8533876 	 acc:  0.62
1634 : loss:  0.70930636 	 acc:  0.72
1636 : loss:  0.65780723 	 acc:  0.71
1638 : loss:  0.66010207 	 acc:  0.74
1640 : loss:  0.80659056 	 acc:  0.59
1642 : loss:  0.64657396 	 acc:  0.69
1644 : loss:  0.75981736 	 acc:  0.6
1646 : loss:  0.86078763 	 acc:  0.55
1648 : loss:  0.76573855 	 acc:  0.58
1650 : loss:  0.7605684 	 acc:  0.67
1652 : loss:  0.71970755 	 acc:  0.6
1654 : loss:  0.7362004 	 acc:  0.72
1656 : loss:  0.87273717 	 acc:  0.57
1658 : loss:  0.7446373 	 acc:  0.67
1660 : loss:  0.72435516 	 acc:  0.69
1662 : loss:  0.8466966 	 acc:  0.65
1664 : loss:  0.66802907 	 acc:  0.68
1666 : loss:  0.78561664 	 acc:  0.59
1668 : loss:  0.663242 	 acc:  0.69
1670 : loss:  0.80578524 	 acc:  0.56
1672 : loss:  0.7151487 	 acc:  0.69
1674 : loss:  0.7501646 	 acc:  0.64
1676 : loss:  0.74500215 	 acc:  0.68
1678 : loss:  0.6791703 	 acc:  0.73
1680 : loss:  0.7444339 	 acc:  0.7
1682 : loss:  0.66783243 	 acc:  0.69
1684 : loss:  0.6864976 	 acc:  0.7
1686 : loss:  0.79491174 	 acc:  0.59
1688 : loss:  0.80505514 	 acc:  0.68
1690 : loss:  0.7371172 	 acc:  0.59
1692 : loss:  0.7015569 	 acc:  0.7
1694 : loss:  0.8482499 	 acc:  0.56
1696 : loss:  0.7231751 	 acc:  0.7
1698 : loss:  0.79670626 	 acc:  0.62
1700 : loss:  0.6622721 	 acc:  0.68
1702 : loss:  0.703688 	 acc:  0.68
1704 : loss:  0.73044425 	 acc:  0.62
1706 : loss:  0.6448827 	 acc:  0.72
1708 : loss:  0.6881265 	 acc:  0.71
1710 : loss:  0.77989036 	 acc:  0.65
1712 : loss:  0.7490213 	 acc:  0.64
1714 : loss:  0.7117437 	 acc:  0.66
1716 : loss:  0.75400245 	 acc:  0.69
1718 : loss:  0.77276045 	 acc:  0.67
1720 : loss:  0.6731583 	 acc:  0.63
1722 : loss:  0.66216004 	 acc:  0.65
1724 : loss:  0.72510684 	 acc:  0.66
1726 : loss:  0.743827 	 acc:  0.69
1728 : loss:  0.77559394 	 acc:  0.68
1730 : loss:  0.8474071 	 acc:  0.61
1732 : loss:  0.7327534 	 acc:  0.71
1734 : loss:  0.71036667 	 acc:  0.69
1736 : loss:  0.76558334 	 acc:  0.69
1738 : loss:  0.7818688 	 acc:  0.61
1740 : loss:  0.8295884 	 acc:  0.65
1742 : loss:  0.72112393 	 acc:  0.68
1744 : loss:  0.67549497 	 acc:  0.7
1746 : loss:  0.7601687 	 acc:  0.61
1748 : loss:  0.8004206 	 acc:  0.64
1750 : loss:  0.8032526 	 acc:  0.67
1752 : loss:  0.79167455 	 acc:  0.65
1754 : loss:  0.84521705 	 acc:  0.57
1756 : loss:  0.79089475 	 acc:  0.66
1758 : loss:  0.79646224 	 acc:  0.61
1760 : loss:  0.70228213 	 acc:  0.7
1762 : loss:  0.77496415 	 acc:  0.67
1764 : loss:  0.71742874 	 acc:  0.68
1766 : loss:  0.9223279 	 acc:  0.59
1768 : loss:  0.6809391 	 acc:  0.71
1770 : loss:  0.80452704 	 acc:  0.62
1772 : loss:  0.59924215 	 acc:  0.71
1774 : loss:  0.6568696 	 acc:  0.72
1776 : loss:  0.78043664 	 acc:  0.61
1778 : loss:  0.7750832 	 acc:  0.6
1780 : loss:  0.7695443 	 acc:  0.62
1782 : loss:  0.8433865 	 acc:  0.64
1784 : loss:  0.79566026 	 acc:  0.67
1786 : loss:  0.892606 	 acc:  0.59
1788 : loss:  0.7283477 	 acc:  0.63
1790 : loss:  0.76828694 	 acc:  0.62
1792 : loss:  0.7093143 	 acc:  0.67
1794 : loss:  0.8364669 	 acc:  0.61
1796 : loss:  0.766944 	 acc:  0.6
1798 : loss:  0.82130975 	 acc:  0.62
1800 : loss:  0.76997024 	 acc:  0.71
1802 : loss:  0.90524054 	 acc:  0.56
1804 : loss:  0.739997 	 acc:  0.67
1806 : loss:  0.65312225 	 acc:  0.65
1808 : loss:  0.7541493 	 acc:  0.65
1810 : loss:  0.9677353 	 acc:  0.59
1812 : loss:  0.8417239 	 acc:  0.6
1814 : loss:  0.6941195 	 acc:  0.71
1816 : loss:  0.9058931 	 acc:  0.54
1818 : loss:  0.71652275 	 acc:  0.65
1820 : loss:  0.71357954 	 acc:  0.67
1822 : loss:  0.71390116 	 acc:  0.68
1824 : loss:  0.7693781 	 acc:  0.65
1826 : loss:  0.666476 	 acc:  0.75
1828 : loss:  0.82420915 	 acc:  0.59
1830 : loss:  0.7968699 	 acc:  0.63
1832 : loss:  0.70336133 	 acc:  0.7
1834 : loss:  0.80264366 	 acc:  0.62
1836 : loss:  0.8027978 	 acc:  0.62
1838 : loss:  0.77829367 	 acc:  0.65
1840 : loss:  0.6854189 	 acc:  0.68
1842 : loss:  0.9510962 	 acc:  0.54
1844 : loss:  0.7838663 	 acc:  0.63
1846 : loss:  0.76605815 	 acc:  0.62
1848 : loss:  0.6928673 	 acc:  0.63
1850 : loss:  0.79027194 	 acc:  0.55
1852 : loss:  0.6368218 	 acc:  0.76
1854 : loss:  0.73272246 	 acc:  0.72
1856 : loss:  0.8572312 	 acc:  0.57
1858 : loss:  0.76579595 	 acc:  0.63

Saving...
saved to models/CNN/pretrained_model.ckpt-1860

100 	 [69.87245449 61.66957639 56.53355832]
4 	val accuracy:  0.6335 	 f_! score:  [0.69872454 0.61669576 0.56533558]

1860 : loss:  0.8708971 	 acc:  0.64
1862 : loss:  0.7664455 	 acc:  0.66
1864 : loss:  0.7382796 	 acc:  0.66
1866 : loss:  0.6684076 	 acc:  0.73
1868 : loss:  0.65538687 	 acc:  0.65
1870 : loss:  0.76605934 	 acc:  0.67
1872 : loss:  0.70049036 	 acc:  0.65
1874 : loss:  0.88016164 	 acc:  0.59
1876 : loss:  0.7455052 	 acc:  0.67
1878 : loss:  0.9143518 	 acc:  0.62
1880 : loss:  0.68887985 	 acc:  0.65
1882 : loss:  0.70081 	 acc:  0.67
1884 : loss:  0.7347692 	 acc:  0.65
1886 : loss:  0.788099 	 acc:  0.61
1888 : loss:  0.712046 	 acc:  0.65
1890 : loss:  0.879463 	 acc:  0.57
1892 : loss:  0.7316977 	 acc:  0.65
1894 : loss:  0.7423269 	 acc:  0.72
1896 : loss:  0.8070762 	 acc:  0.65
1898 : loss:  0.7133656 	 acc:  0.69
1900 : loss:  0.81947654 	 acc:  0.65
1902 : loss:  0.78234184 	 acc:  0.66
1904 : loss:  0.8351197 	 acc:  0.6
1906 : loss:  0.6920893 	 acc:  0.67
1908 : loss:  0.8269478 	 acc:  0.57
1910 : loss:  0.6615777 	 acc:  0.69
1912 : loss:  0.7756164 	 acc:  0.61
1914 : loss:  0.7832768 	 acc:  0.62
1916 : loss:  0.76584023 	 acc:  0.65
1918 : loss:  0.69121665 	 acc:  0.71
1920 : loss:  0.7540867 	 acc:  0.7
1922 : loss:  0.76260304 	 acc:  0.61
1924 : loss:  0.7751974 	 acc:  0.57
1926 : loss:  0.72860664 	 acc:  0.66
1928 : loss:  0.84290826 	 acc:  0.59
1930 : loss:  0.7516587 	 acc:  0.65
1932 : loss:  0.7795055 	 acc:  0.58
1934 : loss:  0.71113855 	 acc:  0.72
1936 : loss:  0.8832896 	 acc:  0.57
1938 : loss:  0.8791273 	 acc:  0.58
1940 : loss:  0.76039606 	 acc:  0.64
1942 : loss:  0.66352105 	 acc:  0.69
1944 : loss:  0.71237683 	 acc:  0.72
1946 : loss:  0.72472227 	 acc:  0.66
1948 : loss:  0.7743214 	 acc:  0.61
1950 : loss:  0.79756516 	 acc:  0.67
1952 : loss:  0.72130203 	 acc:  0.69
1954 : loss:  0.8542129 	 acc:  0.62
1956 : loss:  0.7187377 	 acc:  0.62
1958 : loss:  0.6605492 	 acc:  0.72
1960 : loss:  0.6937871 	 acc:  0.66
1962 : loss:  0.73544526 	 acc:  0.69
1964 : loss:  0.90535307 	 acc:  0.53
1966 : loss:  0.74750197 	 acc:  0.63
1968 : loss:  0.8003339 	 acc:  0.64
1970 : loss:  0.7396684 	 acc:  0.64
1972 : loss:  0.7818419 	 acc:  0.66
1974 : loss:  0.707665 	 acc:  0.68
1976 : loss:  0.69913566 	 acc:  0.72
1978 : loss:  0.81459856 	 acc:  0.65
1980 : loss:  0.7205972 	 acc:  0.67
1982 : loss:  0.7781482 	 acc:  0.66
1984 : loss:  0.76949906 	 acc:  0.67
1986 : loss:  0.76316434 	 acc:  0.69
1988 : loss:  0.80269814 	 acc:  0.61
1990 : loss:  0.5485037 	 acc:  0.79
1992 : loss:  0.75246626 	 acc:  0.65
1994 : loss:  0.7986384 	 acc:  0.66
1996 : loss:  0.6701273 	 acc:  0.72
1998 : loss:  0.8046688 	 acc:  0.65
2000 : loss:  0.7862735 	 acc:  0.63
2002 : loss:  0.7292728 	 acc:  0.62
2004 : loss:  0.6434602 	 acc:  0.71
2006 : loss:  0.65608925 	 acc:  0.69
2008 : loss:  0.8545823 	 acc:  0.58
2010 : loss:  0.7671857 	 acc:  0.67
2012 : loss:  0.89705986 	 acc:  0.55
2014 : loss:  0.6902454 	 acc:  0.68
2016 : loss:  0.72156614 	 acc:  0.67
2018 : loss:  0.6557866 	 acc:  0.71
2020 : loss:  0.80875266 	 acc:  0.65
2022 : loss:  0.7818274 	 acc:  0.61
2024 : loss:  0.87341875 	 acc:  0.54
2026 : loss:  0.83670306 	 acc:  0.59
2028 : loss:  0.71652055 	 acc:  0.68
2030 : loss:  0.73691475 	 acc:  0.63
2032 : loss:  0.7236142 	 acc:  0.65
2034 : loss:  0.795208 	 acc:  0.66
2036 : loss:  0.6742189 	 acc:  0.7
2038 : loss:  0.75554824 	 acc:  0.69
2040 : loss:  0.74550307 	 acc:  0.66
2042 : loss:  0.809015 	 acc:  0.62
2044 : loss:  0.7234976 	 acc:  0.7
2046 : loss:  0.83735853 	 acc:  0.62
2048 : loss:  0.6843934 	 acc:  0.69
2050 : loss:  0.76306725 	 acc:  0.61
2052 : loss:  0.66024417 	 acc:  0.66
2054 : loss:  0.69579977 	 acc:  0.7
2056 : loss:  0.7901771 	 acc:  0.64
2058 : loss:  0.7054825 	 acc:  0.68
2060 : loss:  0.78402483 	 acc:  0.7
2062 : loss:  0.78749937 	 acc:  0.67
2064 : loss:  0.72762537 	 acc:  0.69
2066 : loss:  0.7772781 	 acc:  0.62
2068 : loss:  0.58294713 	 acc:  0.77
2070 : loss:  0.7918722 	 acc:  0.59
2072 : loss:  0.7655886 	 acc:  0.66
2074 : loss:  0.74891734 	 acc:  0.66
2076 : loss:  0.7398362 	 acc:  0.69
2078 : loss:  0.69852996 	 acc:  0.69
2080 : loss:  0.9186666 	 acc:  0.54
2082 : loss:  0.70317644 	 acc:  0.7
2084 : loss:  0.77305394 	 acc:  0.67
2086 : loss:  0.7666266 	 acc:  0.68
2088 : loss:  0.8291438 	 acc:  0.61
2090 : loss:  0.7580446 	 acc:  0.62
2092 : loss:  0.7188452 	 acc:  0.65
2094 : loss:  0.8421196 	 acc:  0.65
2096 : loss:  0.91354346 	 acc:  0.62
2098 : loss:  0.5905196 	 acc:  0.75
2100 : loss:  0.7865225 	 acc:  0.62
2102 : loss:  0.800064 	 acc:  0.71
2104 : loss:  0.72520185 	 acc:  0.67
2106 : loss:  0.6703312 	 acc:  0.74
2108 : loss:  0.73059636 	 acc:  0.64
2110 : loss:  0.72095 	 acc:  0.7
2112 : loss:  0.80094445 	 acc:  0.61
2114 : loss:  0.7325428 	 acc:  0.69
2116 : loss:  0.7345861 	 acc:  0.67
2118 : loss:  0.7254383 	 acc:  0.66
2120 : loss:  0.86043024 	 acc:  0.6
2122 : loss:  0.8016979 	 acc:  0.62
2124 : loss:  0.758627 	 acc:  0.67
2126 : loss:  0.8810432 	 acc:  0.64
2128 : loss:  0.70383817 	 acc:  0.67
2130 : loss:  0.6564054 	 acc:  0.77
2132 : loss:  0.75976485 	 acc:  0.65
2134 : loss:  0.67316025 	 acc:  0.74
2136 : loss:  0.8535262 	 acc:  0.59
2138 : loss:  0.68319046 	 acc:  0.75
2140 : loss:  0.7472207 	 acc:  0.69
2142 : loss:  0.80633485 	 acc:  0.63
2144 : loss:  0.8261176 	 acc:  0.62
2146 : loss:  0.6921772 	 acc:  0.71
2148 : loss:  0.7325917 	 acc:  0.68
2150 : loss:  0.7404538 	 acc:  0.72
2152 : loss:  0.74377334 	 acc:  0.69
2154 : loss:  0.728701 	 acc:  0.67
2156 : loss:  0.65572906 	 acc:  0.75
2158 : loss:  0.6638014 	 acc:  0.73
2160 : loss:  0.780545 	 acc:  0.63
2162 : loss:  0.7302826 	 acc:  0.66
2164 : loss:  0.6172862 	 acc:  0.68
2166 : loss:  0.64642334 	 acc:  0.69
2168 : loss:  0.60516477 	 acc:  0.76
2170 : loss:  0.6289109 	 acc:  0.68
2172 : loss:  0.6672112 	 acc:  0.73
2174 : loss:  0.76920736 	 acc:  0.64
2176 : loss:  0.675103 	 acc:  0.69
2178 : loss:  0.7729454 	 acc:  0.62
2180 : loss:  0.71627325 	 acc:  0.65
2182 : loss:  0.7494314 	 acc:  0.62
2184 : loss:  0.72982264 	 acc:  0.69
2186 : loss:  0.8140018 	 acc:  0.58
2188 : loss:  0.81152964 	 acc:  0.65
2190 : loss:  0.68602157 	 acc:  0.66
2192 : loss:  0.8015948 	 acc:  0.67
2194 : loss:  0.78418654 	 acc:  0.64
2196 : loss:  0.5940257 	 acc:  0.73
2198 : loss:  0.67288244 	 acc:  0.74
2200 : loss:  0.62071717 	 acc:  0.72
2202 : loss:  0.70438933 	 acc:  0.71
2204 : loss:  0.683199 	 acc:  0.74
2206 : loss:  0.6990334 	 acc:  0.62
2208 : loss:  0.77125704 	 acc:  0.64
2210 : loss:  0.8343631 	 acc:  0.61
2212 : loss:  0.80183375 	 acc:  0.66
2214 : loss:  0.7040902 	 acc:  0.66
2216 : loss:  0.72917855 	 acc:  0.69
2218 : loss:  0.63850707 	 acc:  0.76
2220 : loss:  0.77381706 	 acc:  0.66
2222 : loss:  0.8134453 	 acc:  0.64
2224 : loss:  0.7636872 	 acc:  0.62
2226 : loss:  0.82426107 	 acc:  0.58
2228 : loss:  0.7117027 	 acc:  0.7
2230 : loss:  0.7169061 	 acc:  0.72

Saving...
saved to models/CNN/pretrained_model.ckpt-2232

100 	 [69.84992762 63.88632865 50.95001661]
5 	val accuracy:  0.6295 	 f_! score:  [0.69849928 0.63886329 0.50950017]

2232 : loss:  0.79499406 	 acc:  0.64
2234 : loss:  0.6799384 	 acc:  0.63
2236 : loss:  0.7584076 	 acc:  0.63
2238 : loss:  0.84426534 	 acc:  0.57
2240 : loss:  0.71890223 	 acc:  0.65
2242 : loss:  0.6388086 	 acc:  0.77
2244 : loss:  0.6558602 	 acc:  0.71
2246 : loss:  0.82926697 	 acc:  0.62
2248 : loss:  0.68101346 	 acc:  0.74
2250 : loss:  0.69392973 	 acc:  0.68
2252 : loss:  0.8838699 	 acc:  0.6
2254 : loss:  0.7856769 	 acc:  0.64
2256 : loss:  0.57216036 	 acc:  0.65
2258 : loss:  0.86045814 	 acc:  0.61
2260 : loss:  0.63315016 	 acc:  0.72
2262 : loss:  0.7435289 	 acc:  0.62
2264 : loss:  0.64868873 	 acc:  0.72
2266 : loss:  0.6180178 	 acc:  0.69
2268 : loss:  0.66149354 	 acc:  0.74
2270 : loss:  0.72896105 	 acc:  0.67
2272 : loss:  0.79619443 	 acc:  0.6
2274 : loss:  0.670654 	 acc:  0.71
2276 : loss:  0.8234105 	 acc:  0.63
2278 : loss:  0.5802493 	 acc:  0.78
2280 : loss:  0.775765 	 acc:  0.63
2282 : loss:  0.6962305 	 acc:  0.7
2284 : loss:  0.7693844 	 acc:  0.69
2286 : loss:  0.5898223 	 acc:  0.74
2288 : loss:  0.6739668 	 acc:  0.72
2290 : loss:  0.8135041 	 acc:  0.64
2292 : loss:  0.69668067 	 acc:  0.68
2294 : loss:  0.6478107 	 acc:  0.68
2296 : loss:  0.806273 	 acc:  0.59
2298 : loss:  0.73979646 	 acc:  0.66
2300 : loss:  0.7756453 	 acc:  0.62
2302 : loss:  0.6299094 	 acc:  0.74
2304 : loss:  0.8943034 	 acc:  0.54
2306 : loss:  0.7052824 	 acc:  0.7
2308 : loss:  0.6555396 	 acc:  0.7
2310 : loss:  0.71681 	 acc:  0.7
2312 : loss:  0.68175364 	 acc:  0.7
2314 : loss:  0.63137347 	 acc:  0.78
2316 : loss:  0.8040891 	 acc:  0.6
2318 : loss:  0.8232855 	 acc:  0.58
2320 : loss:  0.8759434 	 acc:  0.56
2322 : loss:  0.6898145 	 acc:  0.68
2324 : loss:  0.6799698 	 acc:  0.72
2326 : loss:  0.66187996 	 acc:  0.68
2328 : loss:  0.6859788 	 acc:  0.75
2330 : loss:  0.6942297 	 acc:  0.69
2332 : loss:  0.62010497 	 acc:  0.75
2334 : loss:  0.62990546 	 acc:  0.74
2336 : loss:  0.62281585 	 acc:  0.72
2338 : loss:  0.6229993 	 acc:  0.73
2340 : loss:  0.8020946 	 acc:  0.61
2342 : loss:  0.9000623 	 acc:  0.64
2344 : loss:  0.8004786 	 acc:  0.64
2346 : loss:  0.76694024 	 acc:  0.68
2348 : loss:  0.6971594 	 acc:  0.64
2350 : loss:  0.70021766 	 acc:  0.73
2352 : loss:  0.7990665 	 acc:  0.61
2354 : loss:  0.60011184 	 acc:  0.8
2356 : loss:  0.6446326 	 acc:  0.75
2358 : loss:  0.65629226 	 acc:  0.7
2360 : loss:  0.7344571 	 acc:  0.63
2362 : loss:  0.7164637 	 acc:  0.64
2364 : loss:  0.76352745 	 acc:  0.64
2366 : loss:  0.69634473 	 acc:  0.66
2368 : loss:  0.6863992 	 acc:  0.65
2370 : loss:  0.75697345 	 acc:  0.72
2372 : loss:  0.7016203 	 acc:  0.66
2374 : loss:  0.80411804 	 acc:  0.63
2376 : loss:  0.7420807 	 acc:  0.63
2378 : loss:  0.6694387 	 acc:  0.7
2380 : loss:  0.73176306 	 acc:  0.66
2382 : loss:  0.60732895 	 acc:  0.71
2384 : loss:  0.88644224 	 acc:  0.61
2386 : loss:  0.7378154 	 acc:  0.69
2388 : loss:  0.82314306 	 acc:  0.66
2390 : loss:  0.71477425 	 acc:  0.67
2392 : loss:  0.69699365 	 acc:  0.69
2394 : loss:  0.74561673 	 acc:  0.66
2396 : loss:  0.71935916 	 acc:  0.67
2398 : loss:  0.7620248 	 acc:  0.61
2400 : loss:  0.6455361 	 acc:  0.74
2402 : loss:  0.69673544 	 acc:  0.67
2404 : loss:  0.8242506 	 acc:  0.68
2406 : loss:  0.68819016 	 acc:  0.65
2408 : loss:  0.73834646 	 acc:  0.63
2410 : loss:  0.81274086 	 acc:  0.63
2412 : loss:  0.64324546 	 acc:  0.73
2414 : loss:  0.78185564 	 acc:  0.62
2416 : loss:  0.59790015 	 acc:  0.75
2418 : loss:  0.69602615 	 acc:  0.68
2420 : loss:  0.7447275 	 acc:  0.65
2422 : loss:  0.9110981 	 acc:  0.58
2424 : loss:  0.7279524 	 acc:  0.63
2426 : loss:  0.72948545 	 acc:  0.63
2428 : loss:  0.7297673 	 acc:  0.7
2430 : loss:  0.73890847 	 acc:  0.69
2432 : loss:  0.8180079 	 acc:  0.56
2434 : loss:  0.63538045 	 acc:  0.73
2436 : loss:  0.9008383 	 acc:  0.55
2438 : loss:  0.6251136 	 acc:  0.7
2440 : loss:  0.8483718 	 acc:  0.6
2442 : loss:  0.5991966 	 acc:  0.75
2444 : loss:  0.78126967 	 acc:  0.6
2446 : loss:  0.83103186 	 acc:  0.64
2448 : loss:  0.87595236 	 acc:  0.61
2450 : loss:  0.79364747 	 acc:  0.61
2452 : loss:  0.7111409 	 acc:  0.67
2454 : loss:  0.70695704 	 acc:  0.72
2456 : loss:  0.67238396 	 acc:  0.69
2458 : loss:  0.736094 	 acc:  0.66
2460 : loss:  0.8191173 	 acc:  0.62
2462 : loss:  0.7429786 	 acc:  0.7
2464 : loss:  0.6858471 	 acc:  0.64
2466 : loss:  0.73890495 	 acc:  0.69
2468 : loss:  0.6917427 	 acc:  0.72
2470 : loss:  0.77213776 	 acc:  0.63
2472 : loss:  0.873772 	 acc:  0.59
2474 : loss:  0.658713 	 acc:  0.69
2476 : loss:  0.74211603 	 acc:  0.66
2478 : loss:  0.7454203 	 acc:  0.61
2480 : loss:  0.80185926 	 acc:  0.62
2482 : loss:  0.68094844 	 acc:  0.73
2484 : loss:  0.77750105 	 acc:  0.66
2486 : loss:  0.7419054 	 acc:  0.67
2488 : loss:  0.73090655 	 acc:  0.72
2490 : loss:  0.79103523 	 acc:  0.63
2492 : loss:  0.55269015 	 acc:  0.79
2494 : loss:  0.7821631 	 acc:  0.61
2496 : loss:  0.69708556 	 acc:  0.71
2498 : loss:  0.56077594 	 acc:  0.7
2500 : loss:  0.81898504 	 acc:  0.58
2502 : loss:  0.89177895 	 acc:  0.57
2504 : loss:  0.5804237 	 acc:  0.74
2506 : loss:  0.5753242 	 acc:  0.79
2508 : loss:  0.8533856 	 acc:  0.57
2510 : loss:  0.7703342 	 acc:  0.65
2512 : loss:  0.69543815 	 acc:  0.67
2514 : loss:  0.695745 	 acc:  0.69
2516 : loss:  0.56731236 	 acc:  0.73
2518 : loss:  0.7819372 	 acc:  0.66
2520 : loss:  0.77426445 	 acc:  0.61
2522 : loss:  0.8536516 	 acc:  0.55
2524 : loss:  0.646427 	 acc:  0.71
2526 : loss:  0.7002547 	 acc:  0.69
2528 : loss:  0.7377813 	 acc:  0.72
2530 : loss:  0.9081581 	 acc:  0.6
2532 : loss:  0.75505304 	 acc:  0.65
2534 : loss:  0.76105535 	 acc:  0.65
2536 : loss:  0.7406243 	 acc:  0.66
2538 : loss:  0.6801542 	 acc:  0.67
2540 : loss:  0.6600124 	 acc:  0.69
2542 : loss:  0.709752 	 acc:  0.73
2544 : loss:  0.8057859 	 acc:  0.64
2546 : loss:  0.761395 	 acc:  0.64
2548 : loss:  0.6684513 	 acc:  0.7
2550 : loss:  0.8313334 	 acc:  0.62
2552 : loss:  0.66955817 	 acc:  0.67
2554 : loss:  0.6567143 	 acc:  0.73
2556 : loss:  0.66857344 	 acc:  0.74
2558 : loss:  0.8181473 	 acc:  0.63
2560 : loss:  0.78548014 	 acc:  0.56
2562 : loss:  0.7241914 	 acc:  0.66
2564 : loss:  0.83039427 	 acc:  0.67
2566 : loss:  0.80641294 	 acc:  0.58
2568 : loss:  0.691919 	 acc:  0.68
2570 : loss:  0.6507372 	 acc:  0.73
2572 : loss:  0.67867684 	 acc:  0.7
2574 : loss:  0.40099975 	 acc:  0.9
2576 : loss:  0.8497514 	 acc:  0.57
2578 : loss:  0.8040087 	 acc:  0.61
2580 : loss:  0.7272917 	 acc:  0.7
2582 : loss:  0.7046334 	 acc:  0.62
2584 : loss:  0.8306259 	 acc:  0.58
2586 : loss:  0.6282663 	 acc:  0.71
2588 : loss:  0.75692457 	 acc:  0.65
2590 : loss:  0.9171363 	 acc:  0.61
2592 : loss:  0.7193393 	 acc:  0.67
2594 : loss:  0.74055654 	 acc:  0.68
2596 : loss:  0.7057241 	 acc:  0.64
2598 : loss:  0.78170633 	 acc:  0.65
2600 : loss:  0.73511386 	 acc:  0.69
2602 : loss:  0.7917723 	 acc:  0.63

Saving...
saved to models/CNN/pretrained_model.ckpt-2604

100 	 [70.57326314 65.07821634 52.658453  ]
6 	val accuracy:  0.6392 	 f_! score:  [0.70573263 0.65078216 0.52658453]

100 	 [70.57326314 65.07821634 52.658453  ]
100 	 [67.11136048 59.15556121 67.62910182]
100 	 [75.03040139 73.07219499 43.6301256 ]
100 	 [3362. 3315. 3323.]
--- Test   Twitter ---
0.6392
f1:  [0.70573263 0.65078216 0.52658453]
--- 195.17237329483032 seconds ---

"""


