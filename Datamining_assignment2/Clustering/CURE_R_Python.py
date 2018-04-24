from pyclustering.cluster.cure import cure
from pyclustering.cluster import cluster_visualizer
import csv

# -*- coding:uft-8 -*-
result = []


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

    cure_instance = cure(
        samples, 8, number_represent_points=8, compression=0.25)
    cure_instance.process()
    clusters = cure_instance.get_clusters()

    return clusters


result = r_python_cure_iterface("Clustering/samples.csv")
