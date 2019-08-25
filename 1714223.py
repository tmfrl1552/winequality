
import numpy as np
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import numpy as np
test = []

def printmenu():
    print(" [ Student ID: 1714223 ]")
    print(" [ Name: Lee Seul Gi ]\n")
    print("1. Predict wine quality\n2. Evaluate wine prediction model")
    print("3. Cluster wines\n4. Quit\n\n")

def Decision_Tree():
    global classifier
    data = np.genfromtxt('winequality-red.csv', dtype = np.float32, delimiter = ";", skip_header = 1, usecols = range(0,12))
    x = data[:, 0:11]
    y = data[:, 11]

    # model building
    random_state = 0
    classifier = tree. DecisionTreeClassifier()
    classifier = classifier.fit(x, y)
    #test_sample을 배열로 받아야 함
    test_sample = np.array(test, dtype = "float64").reshape(1, -1)
    #predict the class of the test sample
    predicted_class = classifier.predict(test_sample)
    print("1. Decision tree: %d" % predicted_class)
    
def SVM():
    global classifier
    data = np.genfromtxt('winequality-red.csv', dtype = np.float32, delimiter = ";", skip_header = 1, usecols = range(0,12))
    x = data[:, 0:11]
    y = data[:, 11]

    #model building
    random_state = 0
    classifier = svm.SVC()
    classifier = classifier.fit(x, y)

    test_sample = np.array(test, dtype = "float64").reshape(1, -1)

    #predict the class of the test sample
    predicted_class = classifier.predict(test_sample)
    print("2. Support vector machine: %d" % predicted_class)

def Logistic():
    global classifier
    data = np.genfromtxt('winequality-red.csv', dtype = np.float32, delimiter = ";", skip_header = 1, usecols = range(0,12))
    x = data[:, 0:11]
    y = data[:, 11]

    #model building
    random_state = 0
    classifier = linear_model.LogisticRegression()
    classifier = classifier.fit(x, y)

    test_sample = np.array(test, dtype = "float64").reshape(1, -1)

    #predict the class of the test sample
    predicted_class = classifier.predict(test_sample)
    print("3. Logistic regression: %d" % predicted_class)

def Neighbors():
    global classifier
    data = np.genfromtxt('winequality-red.csv', dtype = np.float32, delimiter = ";", skip_header = 1, usecols = range(0,12))
    x = data[:, 0:11]
    y = data[:, 11]

    #model building
    classifier = neighbors.KNeighborsClassifier(n_neighbors = 5)
    classifier = classifier.fit(x, y)

    test_sample = np.array(test, dtype = "float64").reshape(1, -1)

    #predict the class of the test sample
    
    predicted_class = classifier.predict(test_sample)
    print("4. K-NN classifier: %d" % predicted_class)

def Decision_Eval():
    data = np.genfromtxt('winequality-red.csv', dtype = np.float32, delimiter = ";", skip_header = 1, usecols = range(0,12))
    y_true = data[:, 11]
    #y_pred 만들기
    x = data[:, 0:11]
    y = data[:, 11]

    # model building
    random_state = 0
    classifier = tree. DecisionTreeClassifier()
    classifier = classifier.fit(x, y)
    y_pred = classifier.predict(x)

    print("Decision tree:")
    print("1. confusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("2. Accuracy: ")
    print(accuracy_score(y_true, y_pred))
    print("3. Precision: ")
    print(precision_score(y_true, y_pred, average=None))
    print("4. Recall: ")
    print(recall_score(y_true, y_pred, average=None))
    print("5. F-measure: ")
    print(f1_score(y_true, y_pred, average=None))
    print("\n")

def SVM_Eval():
    data = np.genfromtxt('winequality-red.csv', dtype = np.float32, delimiter = ";", skip_header = 1, usecols = range(0,12))
    y_true = data[:, 11]
    #y_pred 만들기
    x = data[:, 0:11]
    y = data[:, 11]

    # model building
    random_state = 0
    classifier = svm.SVC()
    classifier = classifier.fit(x, y)
    y_pred = classifier.predict(x)
        
    print("Support vector machine:")
    print("1. confusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("2. Accuracy: ")
    print(accuracy_score(y_true, y_pred))
    print("3. Precision: ")
    print(precision_score(y_true, y_pred, average=None))
    print("4. Recall: ")
    print(recall_score(y_true, y_pred, average=None))
    print("5. F-measure: ")
    print(f1_score(y_true, y_pred, average=None))
    print("\n")

def Log_Eval():
    data = np.genfromtxt('winequality-red.csv', dtype = np.float32, delimiter = ";", skip_header = 1, usecols = range(0,12))
    y_true = data[:, 11]
    #y_pred 만들기
    x = data[:, 0:11]
    y = data[:, 11]

    # model building
    random_state = 0
    classifier = linear_model.LogisticRegression()
    classifier = classifier.fit(x, y)
    y_pred = classifier.predict(x)
        
    print("Logistic regression:")
    print("1. confusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("2. Accuracy: ")
    print(accuracy_score(y_true, y_pred))
    print("3. Precision: ")
    print(precision_score(y_true, y_pred, average=None))
    print("4. Recall: ")
    print(recall_score(y_true, y_pred, average=None))
    print("5. F-measure: ")
    print(f1_score(y_true, y_pred, average=None))
    print("\n")

def NN_Eval():
    data = np.genfromtxt('winequality-red.csv', dtype = np.float32, delimiter = ";", skip_header = 1, usecols = range(0,12))
    y_true = data[:, 11]
    #y_pred 만들기
    x = data[:, 0:11]
    y = data[:, 11]

    # model building
    random_state = 0
    classifier = neighbors.KNeighborsClassifier(n_neighbors = 5)
    classifier = classifier.fit(x, y)
    y_pred = classifier.predict(x)
        
    print("K-NN classifier:")
    print("1. confusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("2. Accuracy: ")
    print(accuracy_score(y_true, y_pred))
    print("3. Precision: ")
    print(precision_score(y_true, y_pred, average=None))
    print("4. Recall: ")
    print(recall_score(y_true, y_pred, average=None))
    print("5. F-measure: ")
    print(f1_score(y_true, y_pred, average=None))
    print("\n")
    
    
def hier_C(c_num):
    num = 0
    hier_array = []
    for i in range(c_num):
        hier_array.append(0)    
    data = np.genfromtxt('winequality-red.csv', dtype = np.float32, delimiter = ";", skip_header = 1, usecols = range(0,12))
    X = np.array(data[:, 0:11])
    model = AgglomerativeClustering(n_clusters = c_num)
    model.fit(X)
    
    for i in model.labels_:
        hier_array[i] += 1

    print("<hierarchical clustering>")   
    for i in hier_array:
        print("Cluster %d: %d" % (num, i))
        num += 1
    print("\n")

def Kmeans(c_num):
    num = 0
    K_array = []
    for i in range(c_num):
        K_array.append(0)
    data = np.genfromtxt('winequality-red.csv', dtype = np.float32, delimiter = ";", skip_header = 1, usecols = range(0,12))
    X = np.array(data[:, 0:11])
    model = KMeans(n_clusters = c_num, random_state = 0)
    model.fit(X)

    for i in model.labels_:
        K_array[i] += 1

    print("<K-means clustering>")   
    for i in K_array:
        print("Cluster %d: %d" % (num, i))
        num += 1
    print("\n")

while True:
    printmenu()
    select = int(input())
    print("\n")

    if(select == 1):
        print("Input the values of a wine:")
        fixed_acidity = float(input("1. fixed acidity: "))
        test.append(fixed_acidity)
        volatile_acidity = float(input("2.volatile_acidity: "))
        test.append(volatile_acidity)
        critric_acid = float(input("3. critric acid: "))
        test.append(critric_acid)
        residual_sugar = float(input("4.residual_sugar: "))
        test.append(residual_sugar)
        chlorides = float(input("5. chlorides: "))
        test.append(chlorides)
        free_s_d = float(input("6. free sulfur dioxide: "))
        test.append(free_s_d)
        total_s_d = float(input("7. total sulfur dioxide: "))
        test.append(total_s_d)
        density = float(input("8. density: "))
        test.append(density)
        pH = float(input("9. pH: "))
        test.append(pH)
        sulphates = float(input("10. sulphates: "))
        test.append(sulphates)
        alcohol = float(input("11. alcohol: "))
        test.append(alcohol)
        print("\nPredicted wine quality:")
        Decision_Tree()
        SVM()
        Logistic()
        Neighbors()
        print("\n")
        for i in range(11):
            del test[0]
        

    elif (select == 2):
        Decision_Eval()
        SVM_Eval()
        Log_Eval()
        NN_Eval()
        
    elif (select == 3):
        c_num = int(input("Input the number of clusters: "))
        print("\nThe number of wines in each cluster:\n\n")
        hier_C(c_num)
        Kmeans(c_num)
        
    elif (select == 4):
        break
        
        
