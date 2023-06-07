import numpy as np
import re
import open3d as o3d
import open3d_vis_utils as V
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

class AntColonyAlgorithm:
    def __init__(self, points, num_ants, num_iterations, reduction_factor):
        self.points = points  # 原始点云数据
        self.num_points = len(points)
        self.num_ants = num_ants
        self.num_iterations = num_iterations

        self.pheromone_matrix = np.ones((self.num_points,))  # 初始化信息素矩阵
        self.best_solution = None  # 保留最佳解
        self.best_solution_energy = float(0)  # 最佳解的能量
        self.max_point = self.num_points*reduction_factor #采样后点数

    def calculate_distance(self):
        distance_matrix = pairwise_distances(self.points)  # 计算点之间的欧氏距离
        return distance_matrix

    def calculate_energy(self, selected_points):
        distance_matrix = self.calculate_distance()
        energy = np.sum(distance_matrix[selected_points, :][:, selected_points])  # 选择的点之间的距离之和
        return energy

    def update_pheromone(self, selected_points):
        self.pheromone_matrix[selected_points] += 1  # 更新信息素

    def run(self):
        distance_matrix = self.calculate_distance()

        for _ in range(self.num_iterations):
            for ant in range(self.num_ants):
                visited = []
                current_point = np.random.randint(self.num_points)  # 随机选择起始点
                visited.append(current_point)

                while len(visited) < self.max_point:
                    probabilities = np.zeros((self.num_points,))
                    for point in range(self.num_points):
                        if point not in visited:
                            probabilities[point] = self.pheromone_matrix[point] / distance_matrix[current_point, point]

                    probabilities = probabilities / np.sum(probabilities)
                    next_point = np.random.choice(range(self.num_points), p=probabilities)
                    visited.append(next_point)
                    current_point = next_point

                energy = self.calculate_energy(visited)
                if energy > self.best_solution_energy:  # 更新最佳解
                    self.best_solution = visited
                    self.best_solution_energy = energy


        reduced_points = self.points[self.best_solution,:]  # 选择最佳解对应的点作为降维结果
        return reduced_points

class KNNAlgorithm:
    def __init__(self, k, reduction_factor=1):
        self.k = k
        self.reduction_factor = reduction_factor

    def process_point_cloud(self, points):
        # 使用KNN算法进行点云处理
        num_points = points.shape[0]

        # 使用KNN查找每个点的k个最近邻
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='kd_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)

        # 计算每个点的聚合程度
        aggregation_scores = np.sum(distances, axis=1)

        # 根据聚合程度对点进行排序
        sorted_indices = np.argsort(aggregation_scores)

        # 计算要保留的点的数量
        num_points_to_keep = int(num_points * self.reduction_factor)

        # 选择前num_points_to_keep个最聚合的点
        selected_indices = sorted_indices[:num_points_to_keep]

        # 提取选定的点
        processed_points = points[selected_indices]

        return processed_points



def main():
    # 读取KITTI点云数据
    point_file = '000013_Car_0.bin'
    # 读取标定文件
    calib_file = '000007.txt'
    reduction_factor = 0.7

    def read_calib(calib_path):
        with open(calib_path) as f:
            for line in f.readlines():
                if line[:2] == "P2":
                    P2 = re.split(" ", line.strip())
                    P2 = np.array(P2[-12:], np.float32)
                    P2 = P2.reshape((3, 4))
                if line[:14] == "Tr_velo_to_cam" or line[:11] == "Tr_velo_cam":
                    vtc_mat = re.split(" ", line.strip())
                    vtc_mat = np.array(vtc_mat[-12:], np.float32)
                    vtc_mat = vtc_mat.reshape((3, 4))
                    vtc_mat = np.concatenate([vtc_mat, [[0, 0, 0, 1]]])
                if line[:7] == "R0_rect" or line[:6] == "R_rect":
                    R0 = re.split(" ", line.strip())
                    R0 = np.array(R0[-9:], np.float32)
                    R0 = R0.reshape((3, 3))
                    R0 = np.concatenate([R0, [[0], [0], [0]]], -1)
                    R0 = np.concatenate([R0, [[0, 0, 0, 1]]])
        vtc_mat = np.matmul(R0, vtc_mat)
        return (P2, vtc_mat)

    def read_velodyne(path, P, vtc_mat, IfReduce=True):
        max_row = 374  # y
        max_col = 1241  # x
        lidar = np.fromfile(path, dtype=np.float32).reshape((-1, 4))

        if not IfReduce:
            return lidar

        mask = lidar[:, 0] > 0
        # lidar = lidar[mask]
        lidar_copy = np.zeros(shape=lidar.shape)
        lidar_copy[:, :] = lidar[:, :]

        velo_tocam = vtc_mat
        lidar[:, 3] = 1
        lidar = np.matmul(lidar, velo_tocam.T)
        img_pts = np.matmul(lidar, P.T)
        velo_tocam = np.mat(velo_tocam).I
        velo_tocam = np.array(velo_tocam)
        normal = velo_tocam
        normal = normal[0:3, 0:4]
        lidar = np.matmul(lidar, normal.T)
        lidar_copy[:, 0:3] = lidar
        x, y = img_pts[:, 0] / img_pts[:, 2], img_pts[:, 1] / img_pts[:, 2]
        mask = np.logical_and(np.logical_and(x >= 0, x < max_col), np.logical_and(y >= 0, y < max_row))

        return lidar_copy[mask]
        # return lidar_copy

    def farthest_point_sampling(points, reduction_factor):
        """
        使用最远点采样（FPS）算法对点云进行处理
        :param points: 输入点云数据，形状为(N, 3)
        :param reduction_factor: 点的数量减少因子，介于(0, 1]之间
        :return: 处理后的点云数据，形状为(num_points, 3)
        """
        num_total_points = points.shape[0]
        num_points_to_keep = int(num_total_points * reduction_factor)
        processed_points = np.zeros((num_points_to_keep, 3))
        processed_points[0] = points[np.random.randint(num_total_points)]
        distances = np.ones(num_total_points) * 1e10

        for i in range(1, num_points_to_keep):
            last_point = processed_points[i - 1]
            squared_distances = np.sum((points - last_point) ** 2, axis=1)
            distances = np.minimum(distances, squared_distances)
            next_point_index = np.argmax(distances)
            processed_points[i] = points[next_point_index]

        return processed_points

    P2,V2C = read_calib(calib_file)

    # 将点云转换到相机坐标系
    points = read_velodyne(point_file,P2,V2C)
    points = points[:,:3]

    # 创建蚁群算法和KNN算法对象
    ant_algorithm = AntColonyAlgorithm(points, num_ants=25, num_iterations=10, reduction_factor=reduction_factor)
    knn_algorithm = KNNAlgorithm(k=20, reduction_factor=reduction_factor)

    # 蚁群算法处理点云数据
    processed_points_ant = ant_algorithm.run()
    # processed_points_ant = processed_points_ant.squeeze(0)

    # KNN算法处理点云数据
    processed_points_knn = knn_algorithm.process_point_cloud(points)
    #常用FPS方法
    processed_points_fps =  farthest_point_sampling(points, reduction_factor)

    # 将点云转换回车辆坐标系
    points_vehicle = points

    # 保存处理后的点云数据为KITTI格式
    # points_processed_ant = np.concatenate((points_vehicle, np.expand_dims(intensities, axis=1)), axis=1)
    # points_processed_knn = np.concatenate((points_vehicle, np.expand_dims(intensities, axis=1)), axis=1)
    # processed_points_ant.astype(np.float32).tofile('path_to_processed_kitti_data_ant.bin')
    # processed_points_knn.astype(np.float32).tofile('path_to_processed_kitti_data_knn.bin')


    return points, processed_points_ant, processed_points_knn, processed_points_fps

    # # 可视化处理后的点云数据
    # pcd_processed_ant = o3d.geometry.PointCloud()
    # pcd_processed_ant.points = o3d.utility.Vector3dVector(processed_points_ant[:, :3])
    # pcd_processed_ant.colors = o3d.utility.Vector3dVector(np.zeros_like(processed_points_ant[:, :3]))
    #
    # pcd_processed_knn = o3d.geometry.PointCloud()
    # pcd_processed_ant.points = o3d.utility.Vector3dVector(processed_points_ant[:, :3])
    # pcd_processed_ant.colors = o3d.utility.Vector3dVector(np.zeros_like(processed_points_ant[:, :3]))

if __name__ == '__main__':

    origin_point, ant, knn, fps = main()
    V.draw_scenes(
        points=origin_point, ref_boxes=None,
        ref_scores=None, ref_labels=None
    )
    V.draw_scenes(
        points=ant, ref_boxes=None,
        ref_scores=None, ref_labels=None
    )
    V.draw_scenes(
        points=knn, ref_boxes=None,
        ref_scores=None, ref_labels=None
    )
    V.draw_scenes(
        points=fps, ref_boxes=None,
        ref_scores=None, ref_labels=None
    )

