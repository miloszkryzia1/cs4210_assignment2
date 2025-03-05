# -------------------------------------------------------------------------
# AUTHOR: Milosz Kryzia
# FILENAME: decision_tree_2.py
# SPECIFICATION: This program creates decision trees using different test sets and compares their accuracy
# FOR: CS 4210- Assignment #2
# TIME SPENT:
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    # Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, data in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(data)

    # Transform the original categorical training features to numbers and add to the 4D array X.
    for data in dbTraining:
        x_row = []
        # age
        if data[0] == 'Young':
            x_row.append(0)
        elif data[0] == 'Prepresbyopic':
            x_row.append(1)
        else:
            x_row.append(2)
        # spec pres
        if data[1] == 'Myope':
            x_row.append(0)
        else:
            x_row.append(1)
        # astigmatism
        if data[2] == 'Yes':
            x_row.append(0)
        else:
            x_row.append(1)
        # TPR
        if data[3] == 'Normal':
            x_row.append(0)
        else:
            x_row.append(1)
        X.append(x_row)

    # Transform the original categorical training classes to numbers and add to the vector Y.
    for data in dbTraining:
        if data[4] == 'Yes':
            Y.append(1)
        else:
            Y.append(0)


    accuracies = [] #append accuracy after each of the 10 runs

    # Loop your training and test tasks 10 times here
    for i in range(10):
        # Fitting the decision tree to the data setting max_depth=5
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

        # Read the test data and add this data to dbTest
        dbTest = []
        with open('contact_lens_test.csv', 'r') as file:
            reader = csv.reader(file)
            for i, data in enumerate(reader):
                if i > 0:  # skipping the header
                    dbTest.append(data)

        tp = 0
        tn = 0
        predictions = len(dbTest)

        for data in dbTest:
            # Transform the features of the test instances to numbers following the same strategy done during training
            x = []
            if data[0] == 'Young':
                x.append(0)
            elif data[0] == 'Prepresbyopic':
                x.append(1)
            else:
                x.append(2)
            
            if data[1] == 'Myope':
                x.append(0)
            else:
                x.append(1)
            
            if data[2] == 'Yes':
                x.append(0)
            else:
                x.append(1)
            
            if data[3] == 'Normal':
                x.append(0)
            else:
                x.append(1)
            
            y = 1 if data[4] == 'Yes' else 0

            class_predicted = clf.predict([x])[0]

            # Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            if class_predicted == y:
                if y == 1:
                    tp += 1
                else:
                    tn += 1
            
        accuracy = (tp+tn) / predictions
        accuracies.append(accuracy)

    # Find the average of this model during the 10 runs (training and test set)
    avg_accuracy = sum(accuracies) / len(accuracies)

    # Print the average accuracy of this model during the 10 runs (training and test set).
    # Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f"Final accuracy when training {ds}: {avg_accuracy}")
