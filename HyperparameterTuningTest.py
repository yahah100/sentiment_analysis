from TestHelper import TestHelper


net_name = "CNN"

testhelper = TestHelper()

learn_rates = [0.001,0.0008,0.0005,0.0001,0.00008]
mean_test_acc = []
for learn_rate in learn_rates:
    acc = testhelper.hyper_tuning("Review", net_name, learn_rate)
    mean_test_acc.append(acc)

print("Final Result: \n \n", mean_test_acc)
"""
Final Result: 
 
 [0.69018006, 0.68802, 0.68324, 0.64695996, 0.64479995]

"""
mean_test_acc = []
for learn_rate in learn_rates:
    acc = testhelper.hyper_tuning("Twitter", net_name, learn_rate)
    mean_test_acc.append(acc)

print("Final Result: \n \n", mean_test_acc)
"""
Final Result: 
 
 [0.597, 0.62399995, 0.62310004, 0.5873, 0.5779]

"""
net_name = "LSTM"

mean_test_acc = []
for learn_rate in learn_rates:
    acc = testhelper.hyper_tuning("Review", net_name, learn_rate)
    mean_test_acc.append(acc)

print("Final Result: \n \n", mean_test_acc)
"""
Final Result: 
 
 [0.6764, 0.6781599, 0.67274, 0.61873996, 0.60407996]

"""
mean_test_acc = []
for learn_rate in learn_rates:
    acc = testhelper.hyper_tuning("Twitter", net_name, learn_rate)
    mean_test_acc.append(acc)

print("Final Result: \n \n", mean_test_acc)

"""
Final Result: 
 
 [0.601, 0.59970003, 0.59610003, 0.57879996, 0.56920004]

"""