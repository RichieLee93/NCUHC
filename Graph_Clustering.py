from collections import Counter
import datetime
from qsr.Distance_neighborhood import calculate_neighborhood_distance


def graph_clustering(data, graph, num, constraint):

	flag_match = False
	for i in graph:
		if data[0][-1] == i[-1]:
			count = 0
			for j in range(len(data[0])):
				if data[0][j] == i[j]:
					count += 1
			if num == 5 and count >= num:
				if calculate_neighborhood_distance(data[0], i) <= constraint:
					flag_match = True
					break
			if num == 6 and count >= num:
				flag_match = True
				break

	return flag_match
	

def get_cluster(data, num, constraint):
	graph = {}
	# thres = 7
	# number_qsr = 5
	for i in range(len(data)):
		if i == 0:
			graph[0] = data[i]
		else:
			flag = True
			for j in graph.keys():
				if graph_clustering(data[i], graph[j], num, constraint):
					graph[j].append(data[i][0])
					flag = False
					break
			if flag:
				graph[len(graph)] = data[i]
		if i%100000 == 0:
			print("Finished {}th data, time is {}".format(i, datetime.datetime.now()))
	print("After processing {}th data, the number of clusters is {}".format(len(data), len(graph)))

	return list(graph.values())
		

		
	
