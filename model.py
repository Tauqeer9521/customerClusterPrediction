# Prepare a cluster of customers to predict the purchase power based on their income and spending score
#Importing Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
from sklearn.cluster import KMeans

#Loading the Dataset into the DataFrame
df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)" , "Spending Score (1-100)"]]
wcss_list = []
for i in range(1 , 11):
    kmeans = KMeans(n_clusters = i , init = "k-means++" , random_state = 1)
    kmeans.fit(X)
    wcss_list.append(kmeans.inertia_)

#Visualize the results
#plt.plot(range(1,11) , wcss_list)
#plt.title("Elbow Method Graph")
#plt.xlabel("Number of Cluster")
#plt.ylabel("WCSS List")
#plt.show()

#Training the model on our dataset
model = KMeans(n_clusters = 5 , init = "k-means++" , random_state = 1)
y_predict = model.fit_predict(X)

print(y_predict) 

#Converting the DataFrame X into a numpy array
X_array = X.values
#plotting the graph of clusters
plt.scatter(X_array[y_predict == 0 , 0] , X_array[y_predict == 0 , 1], s = 100 , color = "Green")
plt.scatter(X_array[y_predict == 1 , 0] , X_array[y_predict == 1 , 1], s = 100 , color = "Red")
plt.scatter(X_array[y_predict == 2 , 0] , X_array[y_predict == 2 , 1], s = 100 , color = "Yellow")
plt.scatter(X_array[y_predict == 3 , 0] , X_array[y_predict == 3 , 1], s = 100 , color = "Blue")
plt.scatter(X_array[y_predict == 4 , 0] , X_array[y_predict == 4 , 1], s = 100 , color = "Orange")

plt.title("Customer Segmentation Graph")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

joblib.dump(model , "Model.pkl")
print("Model is Saved")