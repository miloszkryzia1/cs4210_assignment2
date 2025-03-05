#-------------------------------------------------------------------------
# AUTHOR: Milosz Kryzia
# FILENAME: knn.py
# SPECIFICATION: this program finds the LOO-CV error rate for a 1NN classifier
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

wrong_predictions = 0
predictions = 0

#Loop your data to allow each instance to be your test set
for i in range(len(db)):

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    X = []
    for j in range(len(db)):
       if j != i:
          instc = db[j][:len(db[j])-1]
          for k in range(len(instc)):
             instc[k] = float(instc[k])
          X.append(instc)

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    Y = []
    for j in range(len(db)):
       if j != i:
          c = 1 if db[j][-1] == 'ham' else 0
          Y.append(c)

    #Store the test sample of this iteration in the vector testSample
    testSample = db[i][:-1]
    for j in range(len(testSample)):
       testSample[j] = float(testSample[j])

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    test_c = 1 if db[i][-1] == 'ham' else 0
    if class_predicted != test_c:
       wrong_predictions += 1
    predictions += 1

#Print the error rate
error_rate = wrong_predictions / predictions
print(f'Error rate: {error_rate}')





