def create_csv():
	file = open('offline-train.txt','r')

	with open('data_predict.csv','w') as fcsv:
		fcsv.write("Metric,Timestamp,Label,SR,Rate,GR,Load\n")
		for f in file:
			if f == "\n":
				print("Blank found")
				continue
			
			l= f.split(',')

			m = l[0].split(":")[1].split('"')[1]	
			
			t =l[1].split(":")[1]
			
			la =l[2].split(":")[1]
			
			s = l[3].split(":")[1]
			
			r = l[4].split(":")[1]
			
			g = l[5].split(":")[1]
			
			lo = l[6].split(":")[1].split('"')[1]	

			fcsv.write(m+","+t+","+la+","+s+","+r+","+g+","+lo+"\n")

create_csv()