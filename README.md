# classification
#The application of some classifier in python
#add svm function to the plateform, just  for test!

def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0, kernel="rbf", cache_size=600)
    svcClf.fit(traindata, trainlabel)
    
    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i] == pred_testlabel[i]])/float(num)
    cm = confusion_matrix(testlabel, pred_testlabel)
    print (cm)
    print("cnn-svm Accuracy:", accuracy)
    AnalyseClassification(pred_testlabel, testlabel)
