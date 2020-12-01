import csv
import json
import math
import os
import random
import sys
import time

from pyspark import SparkConf, SparkContext

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
   
    while(total > point_num/50 and total > 100 and cnt < 10):
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
    

def calculate_mahalanobis_distance(coord, clus_dict):

    dimension = len(coord)
    two_std_dev = 2 * math.sqrt(dimension)
    
    cluster_number = -1
    cluster_minimum_dis = float("inf")
    
    for i in range(len(clus_dict)):
    
        if clus_dict[i][0] != 1:
            # Calculating standard deviation.
            total_num, some, some_sq = clus_dict[i]
            deviation = []
            
            for d in range(0, dimension):
                deviation.append(math.sqrt(max(1e-10, (some_sq[d]/total_num)-(some[d]/total_num)**2)))
            
            mahalanobis_dis = 0 
            
            #finding the cluster center = sum/N
            madhya = list(map(lambda x: x/clus_dict[i][0], clus_dict[i][1]))
            
            for j in range(len(deviation)):
                mahalanobis_dis += ((coord[j] - madhya[j])/deviation[j])**2
                
            mahalanobis_dis = math.sqrt(mahalanobis_dis)
            
            if mahalanobis_dis < two_std_dev and mahalanobis_dis < cluster_minimum_dis:
                cluster_minimum_dis = mahalanobis_dis
                cluster_number = i
                
    return cluster_number
    
def statistics_sarimadko(index, information, dim):
    
    total_num = information[0]+1
    some = information[1]
    some_sq = information[2]
    
    for i in range(dim):
    
        val = index[i]
        some[i] += val
        some_sq[i] += val**2
    
    return (total_num, some, some_sq)

    
def calculate_stat_information(index_list, idx_coordinates_map, dimension):

    total_num = 0
    some = [0 for val in range(dimension)]
    sq_some = [0 for val in range(dimension)]
    
    for index in index_list:
        coordinates = idx_coordinates_map[index]
        
        for val in range(0, dimension):
            some[val] += coordinates[val]
            sq_some[val] += pow(coordinates[val], 2)
            
        total_num += 1
        
    return (total_num, some, sq_some)
    
    
if __name__ == "__main__":

    # python3 bfr.py <input_path> <n_cluster> <out_file1> <out_file2>
    
    start = time.time()
    
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
    
    input_path = sys.argv[1]
    n_cluster = int(sys.argv[2])
    out_file1 = sys.argv[3]
    out_file2 = sys.argv[4]
    
    conf = SparkConf().setAppName("HW6kmeans")
    sc = SparkContext()
    sc.setLogLevel("ERROR")
    
    files = sorted(os.listdir(input_path))
    
    #print(files)
    
    result_dict = {}
    
    intermittent_content = []
    discard_info = {}
    combined_info = {}
    combined_clus = {}
    residual_clus = {}
    residual_pts = []
    discard_cnt = 0
    residual_cnt = 0
    combined_cnt = 0
    discard_clus_no = 0
    combined_clus_no = 0
    dimension = 0
    FIFTY = 50
    HUNDRED = 100
    
    for i in range(len(files)):
    
        input_rdd = sc.textFile(input_path+"/"+files[i]).map(lambda line: line.split(',')).map(lambda x: [int(x[0]), list(map(float, x[1:]))])
        
        idx_coordinates_map = input_rdd.collectAsMap()
        
        all_idxs = list(idx_coordinates_map.keys())
        
        dimension = len(idx_coordinates_map[all_idxs[0]])
        
        for index in all_idxs:
            result_dict[str(index)] = -1
        
        #print(i, time.time()- start)
        
        # For the first file in the input
        if i == 0:
            idx_rdd = input_rdd.map(lambda x: x[0])
            sample_idxs = idx_rdd.sample(False, 0.1)
            leftover_idxs = idx_rdd.subtract(sample_idxs)
            
            clusters = k_means_clustering(sample_idxs, n_cluster, idx_coordinates_map)
            
            for cluster in clusters:
            
                discard_info[discard_clus_no] = calculate_stat_information(cluster, idx_coordinates_map, dimension)
                discard_cnt += len(cluster)
                
                for index in cluster:
                    result_dict[index] = discard_clus_no

                discard_clus_no += 1
                
            magnify_n_cluster = 5*n_cluster
            leftover_cluster = k_means_clustering(leftover_idxs, magnify_n_cluster, idx_coordinates_map)
            
            for cluster in leftover_cluster:
            
                if len(cluster) > 1:
                
                    combined_info[combined_clus_no] = calculate_stat_information(cluster, idx_coordinates_map, dimension)
                    combined_clus[combined_clus_no] = cluster
                    combined_clus_no += 1
                    combined_cnt += len(cluster)
                else:
                    # There is just one index in the list. Hence can be indexed with 0
                    residual_clus[cluster[0]] = idx_coordinates_map[cluster[0]]
                    residual_pts.append(cluster[0])
                    residual_cnt += 1
                    
        else:
            
            for idx, coord in idx_coordinates_map.items():
                cluster_idx = calculate_mahalanobis_distance(coord, discard_info)
                if cluster_idx != -1:
                    discard_cnt += 1
                    result_dict[str(idx)] = cluster_idx
                    discard_info[cluster_idx] = statistics_sarimadko(coord, discard_info[cluster_idx], dimension)
                else:
                    cluster_idx = calculate_mahalanobis_distance(coord, combined_info)
                    if cluster_idx != -1:
                        combined_cnt += 1
                        combined_clus[cluster_idx].append(idx)
                        combined_info[cluster_idx] = statistics_sarimadko(coord, combined_info[cluster_idx], dimension)
                    else:
                        residual_clus[idx] = coord
                        residual_pts.append(idx)
                        residual_cnt += 1
                        
                        
            if ((len(residual_pts) > n_cluster * FIFTY) and (len(residual_pts) > len(result_dict)/HUNDRED)):
            
                RS_rdd = sc.parallelize(residual_pts)
                magnify_n_cluster = n_cluster * 5
                kmeans_on_resudial = k_means_clustering(RS_rdd, magnify_n_cluster, residual_clus)
                new_RS_cluster = {}
                new_RS_point = []
                
                
                for cluster in kmeans_on_resudial:
                
                    if len(cluster) > 1:
                        cluster_centroids = find_new_centers(cluster, residual_clus, dimension)
                        cluster_idx = calculate_mahalanobis_distance(cluster_centroids, combined_info)
                        combined_cnt += len(cluster)
                        
                        if cluster_idx == -1:
                            combined_clus[combined_clus_no] = cluster
                            combined_info[combined_clus_no] = calculate_stat_information(cluster, residual_clus, dimension)
                            combined_clus_no += 1
                            
                        else:
                            combined_clus[cluster_idx] += cluster
                            for idx in cluster:
                                combined_info[cluster_idx] = statistics_sarimadko(residual_clus[point], combined_info[cluster_idx], dimension)

                    else:
                    
                        new_RS_point += cluster
                        for point in cluster:
                            new_RS_cluster[point] = residual_clus[point]
                            
                residual_clus = new_RS_cluster
                residual_pts = new_RS_point
                residual_cnt = len(residual_pts)
                        
        intermittent_content.append([i+1, discard_clus_no, discard_cnt, combined_clus_no, combined_cnt, residual_cnt])
        
    for cluster_index in range(len(combined_clus)):
    
        cs_cluster_inf = combined_info[cluster_index]
        cs_center = list(map(lambda x: x/cs_cluster_inf[0], cs_cluster_inf[1]))
        merge_cluster_num = calculate_mahalanobis_distance(cs_center, discard_info)
        
        if merge_cluster_num != -1:
            for point in combined_clus[cluster_index]:
                result_dict[str(point)] = merge_cluster_num
        else:
            for point in combined_clus[cluster_index]:
                result_dict[str(point)] = discard_clus_no
            discard_clus_no+= 1

    
    for point in residual_pts:
        result_dict[str(point)] = -1
        
    output = json.dumps(result_dict)
    
    with open(out_file1, "w") as fp:
        fp.write(output)
    
    with open(out_file2, "w") as fp:
    
        write_fp = csv.writer(fp)
        write_fp.writerow(["round_id", "nof_cluster_discard", "nof_point_discard", "nof_cluster_compression", "nof_point_compression", "nof_point_retained"])
        write_fp.writerows(intermittent_content)
        
        
    end = time.time()
    print("Duration:", end-start)