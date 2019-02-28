from TestHelper import TestHelper


net_name = "CNN"

testhelper = TestHelper()
review_loss, review_acc = testhelper.train_input("Medical", net_name, epochs=1)