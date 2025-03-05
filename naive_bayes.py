#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
db = []
with open('weather_training.csv', 'r') as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        if i > 0:
            db.append(row[1:]) #get rid of day ID

X = []
Y = []

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
for data in db:
    #outlook
    if data[0] == 'Sunny':
        data[0] = 1
    elif data[0] == 'Overcast':
        data[0] = 2
    else:
        data[0] = 3
    #temp
    if data[1] == 'Cool':
        data[1] = 1
    elif data[1] == 'Mild':
        data[1] = 2
    else:
        data[1] = 3
    #humidity
    if data[2] == 'Normal':
        data[2] = 1
    else:
        data[2] = 2
    #wind
    if data[3] == 'Weak':
        data[3] = 1
    else:
        data[3] = 2

for data in db:
    X.append(data[:-1])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
for data in db:
    Y.append(1 if data[-1] == 'Yes' else 2)

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
day_labels = []
test_db = []
with open('weather_test.csv', 'r') as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        if i > 0:
            row.pop() #get rid of class label
            day_labels.append(row[0])
            test_db.append(row[1:]) #get rid of day ID

#transform like before
X_TEST = []
for data in test_db:
    x = [0]*4
    #outlook
    if data[0] == 'Sunny':
        x[0] = 1
    elif data[0] == 'Overcast':
        x[0] = 2
    else:
        x[0] = 3
    #temp
    if data[1] == 'Cool':
        x[1] = 1
    elif data[1] == 'Mild':
        x[1] = 2
    else:
        x[1] = 3
    #humidity
    if data[2] == 'Normal':
        x[2] = 1
    else:
        x[2] = 2
    #wind
    if data[3] == 'Weak':
        x[3] = 1
    else:
        x[3] = 2
    X_TEST.append(x)

#Printing the header os the solution
print('Day    Outlook    Temperature    Humidity    Wind    PlayTennis    Confidence')

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for i in range(len(X_TEST)):
    prediction_array = clf.predict_proba([X_TEST[i]])[0]
    if prediction_array[0] >= 0.75:
        prediction = 'Yes'
        confidence = round(prediction_array[0], 2)
    elif prediction_array[1] >= 0.75:
        prediction = 'No'
        confidence = round(prediction_array[1], 2)
    else:
        continue
    print(f'{day_labels[i]}    {test_db[i][0]}    {test_db[i][1]}    {test_db[i][2]}    {test_db[i][3]}    {prediction}    {confidence}')

