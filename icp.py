
import numpy as np
import copy
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import time


class ICPOptimizer():

    def __init__(self, max_distance=10, num_iterations=10, kdtree_leaf_size=40, kdtree_query_dual_tree=True, kdtree_query_breadth_first=True):
        self.num_iterations = num_iterations
        self.max_distance = max_distance
        self.kdtree_leaf_size = kdtree_leaf_size
        self.kdtree_query_dual_tree = kdtree_query_dual_tree
        self.kdtree_query_breadth_first = kdtree_query_breadth_first

    @staticmethod
    def transform_points(source_points, pose_estimation):
        return source_points.dot(pose_estimation.T)

    @staticmethod
    def transform_normals(source_normals, pose_estimation):
        rotataion_matrix = pose_estimation[0:3, 0:3]
        translation = pose_estimation[0:3, 3]
        rotation_inverse = np.linalg.inv(rotataion_matrix)

        return (source_normals.dot(rotation_inverse))+translation

    def prune_correspondences(self, source_normals, target_normals, distances):
        matches = distances <= self.max_distance
        v1_u = normalize(source_normals, norm="l2", axis=1)
        v2_u = normalize(target_normals, norm="l2", axis=1)
        angles = np.arccos(np.clip(np.sum(v1_u*v2_u, axis=1), -1.0, 1.0))
        matches = matches * (angles < 1.0472)

        return matches

    @staticmethod
    def point_to_plane_distance(X, SP, TP, TN):

        A = np.identity(4)
        rotation_matrix = R.from_mrp(
            [X[0], X[1], X[2]]).as_matrix()
        A[0:3, 0:3] = rotation_matrix
        A[0:3, 3] = X[3:6]

        return np.einsum('ij,ij->i', (SP.dot(A.T)-TP)[:, :3], TN)

    def randomSample(self, vertices, sample_rate=1.0):
        vertex_num = vertices.shape[0]
        if int(vertex_num * sample_rate) < 100:
            sample_num = np.min(100, vertex_num)
        else:
            sample_num = int(vertex_num * sample_rate)
        mask = np.random.randint(0, vertex_num, sample_num)
        vertex_samples = vertices[mask, :]
        return vertex_samples

    def estimate_pose(self, source_points, target_points, source_noramls, target_normals, initial_pose=np.eye(4), show_verbose=False):
        source_points_orig = source_points.reshape(-1,3)
        target_points_orig = source_points.reshape(-1,3)
        source_noramls = source_noramls.reshape(-1,3)
        target_normals = target_normals.reshape(-1,3)


        source_points_hom = np.c_[source_points_orig, np.ones(source_points_orig.shape[0])]
        target_points_hom = np.c_[target_points_orig, np.ones(target_points_orig.shape[0])]

        source_points = self.randomSample(source_points_hom, sample_rate=1)
        target_points = self.randomSample(target_points_hom, sample_rate=1)

        print(source_points_hom.shape,source_noramls.shape)
        print(target_points_hom.shape,target_normals.shape)

        tree = KDTree(target_points[:, :3], metric="euclidean")

        pose_estimation = copy.deepcopy(initial_pose)
        print("-------------------------------------------")
        print("ICP starting with : ")
        print("Number of Iterations : {}".format(self.num_iterations))
        print("Number of Source Points : {}".format(source_points.shape[0]))
        print("Number of Target Points : {}".format(target_points.shape[0]))
        print("Initial Pose : {}".format(initial_pose))
        # print("-------------------------------------------")

        icp_start_time = time.time()

        for i in range(self.num_iterations):
            if show_verbose:
                print("Iteration {}/{} Iterations Statring".format(i +
                      1, self.num_iterations))

            iteration_start_time = time.time()

            transformed_points = self.transform_points(
                source_points, pose_estimation)
            transformed_normals = self.transform_normals(
                source_noramls, pose_estimation)

            distances, indices = tree.query(
                transformed_points[:, :3], k=1, return_distance=True)
            indices = np.asarray(indices).flatten()
            distances = np.asarray(distances).flatten()

            arranged_targets = target_points[indices]
            arranged_target_normals = target_normals[indices]

            matches = self.prune_correspondences(
                transformed_normals[indices], arranged_target_normals, distances)

            source_points_used = transformed_points[matches]
            taregt_points_used = arranged_targets[matches]
            target_normals_used = arranged_target_normals[matches]

            print(matches, source_points_used, taregt_points_used, target_normals_used)

            res_lsq = least_squares(self.point_to_plane_distance, np.zeros(6),
                                    args=(source_points_used, taregt_points_used, target_normals_used), method='lm', verbose=0)

            result = np.asarray(res_lsq['x'])
            rotation_matrix = R.from_mrp(
                [result[0], result[1], result[2]]).as_matrix()
            pose_estimation_2 = np.identity(4)
            pose_estimation_2[0:3, 0:3] = rotation_matrix
            pose_estimation_2[0:3, 3] = result[3:6]
            pose_estimation = np.matmul(pose_estimation_2, pose_estimation)

            if show_verbose:
                print("Iteration {}/{} Ended in : {} seconds".format(i+1,
                                                                     self.num_iterations, time.time()-iteration_start_time))

        icp_total_time = time.time()-icp_start_time

        print("")
        print("ICP Completed with total Time : {} seconds".format(icp_total_time))
        print("-------------------------------------------")
        return pose_estimation


# if __name__ == '__main__':
#     source = o3d.io.read_triangle_mesh("../Data/bunny_trans.off")
#     target = o3d.io.read_triangle_mesh("../Data/bunny.off")

#     target.compute_vertex_normals(normalized=True)
#     source.compute_vertex_normals(normalized=True)

#     source_vertices = np.asarray(source.vertices)
#     source_vertices = np.c_[source_vertices, np.ones(source_vertices.shape[0])]

#     target_vertices = np.asarray(target.vertices)
#     target_vertices = np.c_[target_vertices, np.ones(target_vertices.shape[0])]

#     source_vertex_normals = np.asarray(source.vertex_normals)
#     target_vertex_normals = np.asarray(target.vertex_normals)

#     # pose_estimation = np.eye(4)

#     optimizer = ICPOptimizer(num_iterations=10)

#     pose_estimation = (optimizer.estimate_pose(source_points=source_vertices, target_points=target_vertices,
#                                                source_noramls=source_vertex_normals, target_normals=target_vertex_normals))
#     mesh_t = copy.deepcopy(source).transform(pose_estimation)
#     o3d.visualization.draw_geometries([mesh_t, target])
#     exit()
