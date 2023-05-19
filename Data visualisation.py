import matplotlib.pyplot as plt
import numpy
import pandas
#On one axis number from 1 to 15
a = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00591/name_gender_dataset.csv')

#On other axis generate random integers with mean and sigma
mean = 50
sigma = 10
b = numpy.random.normal(mean, sigma, 15).astype(int)
z1=a['Count'][0:15]
z2=a['Probability'][0:15]
z3=a['Gender'][0:15]
plt.plot(z1,b,color='Red',ls='--',lw=4,marker='3',mew=10)
sales = pandas.DataFrame({'X':z1,'Y':z2,'Z': z3})
colors = ['Red','Green','Black']
sales.plot(xticks=range(1,5),yticks=range(0,100,20),color=colors)

plt.bar(z2,z3)

sales.plot(kind='bar')

z11=a['Count'][0:5]
color_list = ['Red','Green','Blue','Yellow','Grey']
plt.pie(z11,labels=['AA','BB','CC','DD','EE'],colors=color_list)
plt.hist(z1)

plt.scatter(z1,z2)


plt.boxplot(z1)
mean = 20
sigma = 5
c = numpy.random.normal(mean, sigma, 15).astype(int)
#Create figure object
fig_sub_object = plt.figure()
#Two axes inside figure object. 
number_of_rows= 1
number_of_cols = 2
fig_sub_object, (axes1,axes2) = plt.subplots(number_of_rows,number_of_cols)

axes1.plot(z1,b)
axes2.plot(z1,c)
import seaborn
from os import name
#Generate random integers with mean and sigma
mean = 25
sigma = 10
dist_data_1 = numpy.random.normal(mean, sigma, 500).astype(int)
dist_data_2 = numpy.random.normal(mean+5, sigma-4, 500).astype(int)
dist_data_3 = numpy.random.normal(mean-5, sigma+2, 500).astype(int)
dist_data = pandas.DataFrame({"A" :dist_data_1,"B":dist_data_2,"C":dist_data_3})
z4=a['Name'][0:15]
seaborn.distplot(z1,bins=10)
seaborn.jointplot(x=z2, y=z3);
seaborn.jointplot(x=z2, y=z1,kind="kde")
dist_data = pandas.DataFrame({"A" :z1,"B":z2,"C":z3})
seaborn.pairplot(dist_data)
seaborn.stripplot(x="B", y="A", data=dist_data, jitter=True)
seaborn.violinplot(x="B", y="A", data=dist_data)
seaborn.pointplot(x="A", y="B",data=dist_data)
