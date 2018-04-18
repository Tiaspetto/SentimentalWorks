from pyclustering.cluster.cure import cure
import csv;

# -*- coding:uft-8 -*-
def r_python_cure_iterface(filepath):
	samples = []
	with open(filepath, 'r') as csvfile:
		csvreader = csv. reader(csvfile, delimiter=",")
		count = 0
		for row in csvreader:
			if count != 0:
			    sample = row
			    sample = [float(x) for x in sample]
			    samples.append(sample)
			count = count + 1

	cure_instance = cure(samples, 8, number_represent_points = 4, compression=1.0);
	cure_instance.process();
	clusters = cure_instance.get_clusters();

	print(clusters)
	#print(samples)

r_python_cure_iterface("samples.csv")
