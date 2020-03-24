import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.ERROR)

def create_csv():
	file = open('offline-train.txt','r')

	with open('data.csv','w') as fcsv:
		fcsv.write("Metric,Timestamp,Label,SR,Rate,GR,Load\n")
		for f in file:
			l= f.split(',')

			m = l[0].split(":")[1].split('"')[1]	
			#metric.append(m)

			t =l[1].split(":")[1]
			#timestamp.append(t)

			la =l[2].split(":")[1]
			#label.append(la)

			s = l[3].split(":")[1]
			#sr.append(s)

			r = l[4].split(":")[1]
			#rate.append(r)

			g = l[5].split(":")[1]
			#gr.append(g)

			lo = l[6].split(":")[1].split('"')[1]	
			#load.append(lo)
			fcsv.write(m+","+t+","+la+","+s+","+r+","+g+","+lo+"\n")

#create_csv()
gear_data = pd.read_csv('data.csv',index_col=None)
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

X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.3, random_state=101)

print(len(X_train), 'train examples')
print(len(X_test), 'test examples')

inputFunction = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=200,num_epochs=1000,shuffle=True)

dnnClassifierModel = tf.estimator.DNNClassifier(hidden_units=[1024, 512, 256],
                                                feature_columns=feat_cols,
                                                n_classes=3,
                                                activation_fn=tf.nn.relu,
                                                optimizer=lambda: tf.train.AdamOptimizer(
                                                    learning_rate=tf.train.exponential_decay(learning_rate=0.001,
                                                    global_step=tf.train.get_global_step(),
                                                    decay_steps=1000,
                                                    decay_rate=0.96)))

dnnClassifierModel.train(input_fn=inputFunction,steps=1000)

evaluateInputFunction = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)
results = dnnClassifierModel.evaluate(evaluateInputFunction)

print(results)