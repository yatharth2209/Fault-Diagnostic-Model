import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

gear_data=pd.read_csv("data_predict.csv")
print(gear_data.head())

cols_to_norm = ['SR', 'GR','Load']

gear_data[cols_to_norm] = gear_data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

sr = tf.feature_column.numeric_column('SR')
rate = tf.feature_column.numeric_column('Rate')
gr = tf.feature_column.numeric_column('GR')
load = tf.feature_column.numeric_column('Load')

print(gear_data.head())

feat_cols = [sr,gr,load]

x_data = gear_data.drop(['Label','Metric','Timestamp','Rate'],axis=1)

labels = gear_data['Label']

x_train, x_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.10, random_state=101)


x_data=tf.keras.utils.normalize(x_data,axis=1)
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)


new_model = tf.keras.models.load_model('trained_model.h5')
new_predictions = new_model.predict(x_data.values)

count0, count1, count2 = 0, 0, 0
for i in new_predictions:
	cl = np.argmax(i)
	if cl==0:
		count0+=1
	elif cl==1:
		count1+=1
	elif cl==2:
		count2+=1
val_loss,val_acc=new_model.evaluate(x_test,y_test)
print(x_train.values.shape)
new_model.fit(x_train.values,y_train.values,epochs=1)
new_model.save('trained_model.h5')
print(val_acc)
print(count0,count1,count2)
print("This batch is classified as class: "+str(np.argmax(np.array([count0,count1,count2]))))
