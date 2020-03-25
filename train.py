import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


gear_data=pd.read_csv("data.csv")
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

x_train, x_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.3, random_state=101)

print(len(x_train), 'train examples')
print(len(x_test), 'test examples')

print(x_train.shape)
print(x_test.shape)
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(512,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(3,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train.values,y_train.values,epochs=10)

model.save('trained_model.h5')

val_loss,val_acc=model.evaluate(x_test,y_test)
print(val_loss,val_acc)