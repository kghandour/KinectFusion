import os
from statistics import NormalDist
from cv2 import resize
import numpy as np
import math
import open3d as o3d
import time
import cv2
from PIL import ImageOps
from camera_sensors import CamDetails
from icp import ICPOptimizer
from raycast import Raycast
from layer import Layer
from transforms import Transforms
from tsdf import TSDFVolume
import copy
from config import config
import torch
from tqdm import tqdm
from torchvision import transforms

class KinectParser():
    def __init__(self,  sensor):
        self.sensor = sensor
        self.fileBaseOut = "mesh_"
        self.tsdfVolume=TSDFVolume(CamDetails.depthIntrinsics)
        self.pyramids_so_far = []
        self.Transformation_list = []
        self.T_matrix = np.eye(4)
        self.icp_optimizer = ICPOptimizer()

    def process(self):
        # vis_pcd = o3d.visualization.Visualizer()
        # vis_pcd.create_window()
        i = 0

        

        if(config.getVisualizeTSDFBool() and i==0):
            vis_volume = o3d.visualization.Visualizer()
            vis_volume.create_window()
        pbar = tqdm(total=len(self.sensor.rgb_images_path), mininterval=1)
        while( self.sensor.processNextFrame(pbar)):
            depthImage = self.sensor.dImage
            # depthImageRaw =  self.sensor.dImageRaw
            colorImageRaw =  self.sensor.rgbImage
            w,h,_ = depthImage.shape
            pyramid = {}
            # resize2 = transforms.functional.resize(size=[w/2,h/2])
            # resize3 = transforms.Resize((0.5))
            
            pyramid['l1'] = Layer(depthImage, colorImageRaw, self.sensor)
            # pyramid['l2'] = Layer(torch.nn.functional.interpolate(depthImage.permute(2,1,0),size=(w/2,h/2)).permute(2,1,0), colorImageRaw.resize((int(w/2), int(h/2))), self.sensor)
            # pyramid['l3'] = Layer(resize3(depthImage), colorImageRaw.resize((int(w/4), int(h/4))), self.sensor)
            if(i==0):
                ## Ground truth
                # self.tsdfVolume.integrate(pyramid["l1"].depthImage, pyramid["l1"].rgbImageRaw,np.asarray(self.sensor.currentTrajectory, dtype=np.double), weight=1)
                # self.T_matrix = np.asarray(self.sensor.currentTrajectory, dtype=np.double)
                ## ICP
                if(config.useGroundTruth()):
                    self.T_matrix = np.asarray(self.sensor.currentTrajectory, dtype=np.double)
                self.tsdfVolume.integrate(pyramid["l1"].depthImage, pyramid["l1"].rgbImageRaw, self.T_matrix)
            else:
                tsdfMesh, normals= self.tsdfVolume.visualize()
                self.tsdf_vertices = np.asarray(tsdfMesh.vertices)
                self.tsdf_normals = np.asarray(normals)



                
                ## Ground truth
                # self.tsdfVolume.integrate(pyramid["l1"].depthImage, pyramid["l1"].rgbImageRaw,np.asarray(self.sensor.currentTrajectory, dtype=np.double))    
                if(config.useGroundTruth()):
                    self.T_matrix = np.asarray(self.sensor.currentTrajectory, dtype=np.double)

                ## ICP
                else:
                    self.T_matrix = self.icp_optimizer.estimate_pose(pyramid["l1"].Vk,self.tsdf_vertices,pyramid["l1"].Nk,self.tsdf_normals, initial_pose=self.T_matrix)
                self.tsdfVolume.integrate(pyramid["l1"].depthImage, pyramid["l1"].rgbImageRaw,self.T_matrix)

            #     self.Transformation_list.append(self.T_matrix)
                # world_vert = Transforms.cam2world(pyramid['l1'].Vk, np.eye(4))
                # print(world_vert)

            self.pyramids_so_far.append(pyramid)
                ### VISUALIZE CURRENT AND PREVIOUS WITH ICP OUTPUT
                # pcd.points = o3d.utility.Vector3dVector(pyramid["l1"].Vk)
                # pcd.colors = o3d.utility.Vector3dVector(pyramid["l1"].rgbImage/255)
                # print("ICP",self.T_matrix)
                # print("Ground", self.sensor.currentTrajectory)

                # print("ICP Inv",np.linalg.inv(self.T_matrix))
                # print("Ground Inv", np.linalg.inv(self.sensor.currentTrajectory))
                # pcd2 = o3d.geometry.PointCloud()
                # pcd2.points = o3d.utility.Vector3dVector(self.pyramids_so_far[-1]['l1'].Vk)
                # pcd2.colors = o3d.utility.Vector3dVector(self.pyramids_so_far[-1]["l1"].rgbImage/255)
                # mesh_t = copy.deepcopy(pcd).transform(np.linalg.inv(self.sensor.currentTrajectory))
                # # print("Copied")

                # o3d.visualization.draw_geometries([pcd2, mesh_t])
                # print("Reached Vis")

                #### END VISUALIZATION

            # print("Reached appending")
            i += self.sensor.increment

            if(i%100==0):
                tsdf_volume_mesh,_ = self.tsdfVolume.visualize()
                o3d.io.write_triangle_mesh("mesh_out/output-"+str(i)+".ply", tsdf_volume_mesh)

            if(config.getVisualizeBool()):
                if(i >= 700):
                    tsdf_volume_mesh,_ = self.tsdfVolume.visualize()
                    o3d.io.write_triangle_mesh("mesh_out/output"+str(i)+".ply", tsdf_volume_mesh)
                    o3d.visualization.draw_geometries([tsdf_volume_mesh])
                    

                    print(self.T_matrix)
                    print(self.sensor.currentTrajectory)
                    while(True):
                        pass
            if(config.getVisualizeTSDFBool()):
                if(i==0):
                    vis_volume.add_geometry(tsdf_volume_mesh)
                else:
                    continue
                    vis_volume.update_geometry(tsdf_volume_mesh)
                    vis_volume.poll_events()
                    vis_volume.update_renderer()
                    vis_volume.destroy_window()


            # self.prev_volume = curr_volume

            # exit()
            # # st = time.time()
            # # vert_pos, vert_col = self.one_loop()
            # # et = time.time()
            # # print("Time taken to process a frame ", et - st)
            # pcd.points = o3d.utility.Vector3dVector(vert_pos)
            # pcd.colors = o3d.utility.Vector3dVector(vert_col/255)
            

            # vis_pcd.poll_events()
            # vis_pcd.update_renderer()
            
            # while(True):
            #     pass
        pbar.close()
        # vis_pcd.destroy_window()


        
