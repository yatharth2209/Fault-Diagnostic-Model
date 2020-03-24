def create_csv():
	file = open('offline-train.txt','r')

	with open('data1.csv','w') as fcsv:
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

create_csv()