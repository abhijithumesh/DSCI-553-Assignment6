import math
import os
import random
import sys
import time
import json

from pyspark import SparkConf, SparkContext
from collections import OrderedDict


def calculate_eucledian_distance(point1, point2):
    sum_res = 0
    
    for i in range(len(point1)):
        sum_res += (point1[i] - point2[i])**2
        
    return math.sqrt(sum_res)


def generate_centroids(indexes_rdd, n_cluster, idx_points_map):

    indexes = indexes_rdd.collect()
    count = 0
    
    random_idx = random.choice(indexes)
    centroids = [(count, idx_points_map[random_idx])]
    
    eucledian_vector = dict()
    
    for index in indexes:
        eucledian_vector[index] = calculate_eucledian_distance(centroids[0][1], idx_points_map[index])
        
    while count < n_cluster-1:

        max_distance = float("-inf")
        
        for key, val in eucledian_vector.items():
            
            if val > max_distance:
                max_distance = val
                subsequent_centroid = key
        
        next_centroid_points = idx_points_map[subsequent_centroid]
        
        count += 1
        centroids.append((count, next_centroid_points))
        
        for index in indexes:
            next_distance = calculate_eucledian_distance(next_centroid_points, idx_points_map[index])
            eucledian_vector[index] = min(eucledian_vector[index], next_distance)
            
    return centroids
    

def centroid_index_mapping(index, coordinates_xyz, centroids):

    min_distance = float("inf")
    
    for cent_val in centroids:
    
        distance = calculate_eucledian_distance(cent_val[1], coordinates_xyz)
        
        if min_distance > distance:
            nearest_cluster = cent_val[0]
            min_distance = distance
            
    return (nearest_cluster, index)


def find_new_centers(cluster_list, idx_points_map, dimension):

    new_centroids = [float(0) for i in range(dimension)]
    cluster_size = len(cluster_list)
    
    for index in cluster_list:
        coordinates = idx_points_map[index]
        
        for i in range(dimension):
        
            new_centroids[i] += coordinates[i]/cluster_size

    return new_centroids
    

def k_means_clustering(indexes_rdd, n_cluster, idx_points_map):

    indexes = indexes_rdd.collect()
    centroids = generate_centroids(indexes_rdd, n_cluster, idx_points_map)

    dimension = len(centroids[0][1])
    
    point_num = indexes_rdd.count()
    total = point_num
    
    cnt = 0
    
    #initially all points are in their own clusters
    cluster_dict = {} 
    for index in indexes:
        cluster_dict[index] = index
   
    while(total > 0 and cnt < 20):
        total = 0
        cluster = indexes_rdd.map(lambda x: centroid_index_mapping(x, idx_points_map[x], centroids))

        centroids = cluster.groupByKey().mapValues(lambda clus_list: find_new_centers(clus_list, idx_points_map, dimension)).collect()

        for clus_idx in cluster.collect():
            if cluster_dict[clus_idx[1]] != clus_idx[0]:
                total += 1
            
            cluster_dict[clus_idx[1]] = clus_idx[0]
            
        cnt += 1
        
    cluster = cluster.groupByKey().map(lambda x: list(x[1])).collect()
    
    return cluster


if __name__ == "__main__":

    start = time.time()
    
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
    
    input_path  = sys.argv[1]
    n_cluster = int(sys.argv[2])
    out_file1 = sys.argv[3]
    
    conf = SparkConf().setAppName("HW6kmeans")
    sc = SparkContext()
    sc.setLogLevel("ERROR")
    
    input_rdd = sc.textFile(input_path).map(lambda x: x.split(','))
    
    idx_points_map = input_rdd.map(lambda x: [int(x[0]), list(map(float, x[1:]))]).collectAsMap()

    indexes = input_rdd.map(lambda x: int(x[0]))
    
    clusters = k_means_clustering(indexes, n_cluster, idx_points_map)

    number_of_clusters = 0
    result = {}
    for cluster in clusters:
        
        for index in cluster:
            result[str(index)] = number_of_clusters
            
        number_of_clusters += 1
        
    result = OrderedDict(sorted(result.items(), key = lambda i: int(i[0])))
        
    with open(out_file1, "w") as fp:
        json.dump(result, fp)
        
    end = time.time()
    
    print("Duration:", end-start)
        