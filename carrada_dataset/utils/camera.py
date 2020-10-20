#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:06:00 2019

@author: julien
"""

import numpy as np
import cv2
import xmltodict

class Camera:
    def __init__(self,Intrinsic_Filename,Extrinsic_Filename):
        
        self.MAX_ITER_NEWTON = 20
        self.MIN_EPS = 0.001
        
        self.__LoadIntrinsics(Intrinsic_Filename)
        self.__LoadExtrinsics(Extrinsic_Filename)
        
        self.cam_x,self.cam_y,self.cam_z,self.cam_pitch,self.cam_yaw,self.cam_roll = self.GetCameraPosition()
        
        print('Camera position:')
        print('\t- X (m):',self.cam_x)
        print('\t- Y (m):',self.cam_y)
        print('\t- Z (m):',self.cam_z)
        
        print('\t- Pitch (deg):',self.cam_pitch*180/np.pi)
        print('\t- Yaw (deg):',self.cam_yaw*180/np.pi)
        print('\t- Roll (deg):',self.cam_roll*180/np.pi)

    def __LoadIntrinsics(self,filename):

        with open(filename) as fd:
            Intrinsic_Dict = xmltodict.parse(fd.read())
            
        self.ImageWidth = int(Intrinsic_Dict['opencv_storage']['width'])
        self.ImageHeight = int(Intrinsic_Dict['opencv_storage']['height'])
        
        # load camera_matrix
        list_coeff=Intrinsic_Dict['opencv_storage']['camera_matrix']['data'].split(sep=' ')
        mat = []
        for coeff in list_coeff:
            mat.append(float(coeff))
        self.camera_matrix = np.asarray(mat).reshape(3,3)
        
        # load distortion coeffs
        list_coeff=Intrinsic_Dict['opencv_storage']['distortion_coefficients']['data'].split(sep=' ')
        mat = []
        for coeff in list_coeff:
            mat.append(float(coeff))
        self.distortion_coefficients = np.asarray(mat)   
        
    def __LoadExtrinsics(self,filename):

        with open(filename) as fd:
            Extrinsic_Dict = xmltodict.parse(fd.read())
        
        # load camera_matrix
        list_coeff=Extrinsic_Dict['opencv_storage']['rotation_matrix']['data'].split(sep=' ')
        mat = []
        for coeff in list_coeff:
            mat.append(float(coeff))
        self.rotation_matrix = np.asarray(mat).reshape(3,3)
        
        # load distortion coeffs
        list_coeff=Extrinsic_Dict['opencv_storage']['translation_vector']['data'].split(sep=' ')
        mat = []
        for coeff in list_coeff:
            mat.append(float(coeff))
        self.translation_vector = np.asarray(mat)
        
    def __worldToCam(self,x_m,y_m,z_m):
        xc_m = self.rotation_matrix[0,0] * x_m + self.rotation_matrix[0,1] * y_m + self.rotation_matrix[0,2] * z_m + self.translation_vector[0]
        
        yc_m = self.rotation_matrix[1,0] * x_m + self.rotation_matrix[1,1] * y_m + self.rotation_matrix[1,2] * z_m +self.translation_vector[1]
        
        zc_m = self.rotation_matrix[2,0] * x_m + self.rotation_matrix[2,1] * y_m +self.rotation_matrix[2,2] * z_m + self.translation_vector[2]
        
        return xc_m,yc_m,zc_m


    def __UndistortedToDistorted(self, xp, yp):
        
        rp2 = xp*xp + yp*yp
        xs = xp * (1 + self.distortion_coefficients[0]*rp2 + self.distortion_coefficients[1]*rp2*rp2 + self.distortion_coefficients[2]*rp2*rp2*rp2) + 2*self.distortion_coefficients[3]*xp*yp + self.distortion_coefficients[4]*(rp2 + 2*xp*xp)
                
        ys = yp * (1 + self.distortion_coefficients[0]*rp2 + self.distortion_coefficients[1]*rp2*rp2 + self.distortion_coefficients[2]*rp2*rp2*rp2) + 2*self.distortion_coefficients[4]*xp*yp + self.distortion_coefficients[3]*(rp2 + 2*yp*yp)
        
        return xs,ys

        
    def worldToImage(self,x_m,y_m,z_m):
    
        xc,yc,zc = self.__worldToCam(x_m, y_m, z_m, )
        
        xp = xc/zc
        yp = yc/zc
        # Out of the image 
        #To avoid change in the sign of the coordinates,
        #due to the undistortion, we apply the formula, without distorsion correction
 
        if(xp < -self.camera_matrix[0,2]/self.camera_matrix[0,0] or
           xp > (self.ImageWidth - self.camera_matrix[0,2])/self.camera_matrix[0,0] or
            yp < -self.camera_matrix[1,2]/self.camera_matrix[1,1] or
           yp > (self.ImageHeight - self.camera_matrix[1,2])/self.camera_matrix[1,1]):
            u_px = self.camera_matrix[0,0] * xp + self.camera_matrix[0,2]
            v_px = self.camera_matrix[1,1] * yp + self.camera_matrix[1,2]
            return u_px,v_px

        xs, ys = self.__UndistortedToDistorted(xp, yp)
        u_px = self.camera_matrix[0,0] * xs + self.camera_matrix[0,2]
        v_px = self.camera_matrix[1,1] * ys + self.camera_matrix[1,2]
        
        return u_px,v_px
    
    def imageToWorld_Z(self, u_px, v_px, z_m):
        
        xs = (u_px - self.camera_matrix[0,2])/self.camera_matrix[0,0]
        ys = (v_px - self.camera_matrix[1,2])/self.camera_matrix[1,1]
        
        imgpts = np.array([[[u_px,v_px]]], dtype=np.float32)
        undistorted_pts = cv2.undistortPoints(imgpts, self.camera_matrix, self.distortion_coefficients)
        xp = undistorted_pts[0][0][0]
        yp = undistorted_pts[0][0][1]
        
        tx = self.translation_vector[0]
        ty = self.translation_vector[1]
        tz = self.translation_vector[2]
        r0 = self.rotation_matrix[0,0]
        r1 = self.rotation_matrix[0,1]
        r2 = self.rotation_matrix[0,2]
        r3 = self.rotation_matrix[1,0]
        r4 = self.rotation_matrix[1,1]
        r5 = self.rotation_matrix[1,2]
        r6 = self.rotation_matrix[2,0]
        r7 = self.rotation_matrix[2,1]
        r8 = self.rotation_matrix[2,2]

        cp = np.cos(-self.cam_pitch)
        sp = np.sin(-self.cam_pitch)
        
        # Vive Maple ;-)
        denom = cp*r2*xp+cp*r5*yp+cp*r8+r0*xp*sp+r3*yp*sp+r6*sp
        num = z_m*sp*r5*yp+z_m*sp*r8+z_m*sp*r2*xp+r6*tz*r5*yp+r6*tz*r2*xp-r0*xp*r5*ty-r0*xp*r8*tz-r0*xp*z_m*cp+r0*tx*r5*yp-r3*yp*r2*tx-r3*yp*r8*tz-r3*yp*z_m*cp+r3*ty*r2*xp+r0*tx*r8-r6*r5*ty+r3*ty*r8-r6*z_m*cp-r6*r2*tx
        x_m = -1*(num/denom)
        zc = (r2*tx+r5*ty+r8*tz-(x_m)*sp+z_m*cp)/(r2*xp+r5*yp+r8)
        y_m = r1*zc*xp-r1*tx+r4*zc*yp-r4*ty+r7*zc-r7*tz

        cx = - r0*tx - r3*ty - r6*tz
        cy = - r1*tx - r4*ty - r7*tz
        cz = - r2*tx - r5*ty - r8*tz

        dx = r0*xp + r3*yp + r6
        dy = r1*xp + r4*yp + r7
        dz = r2*xp + r5*yp + r8

        x_m = (-cz*dx/dz)+cx
        y_m = (-cz*dy/dz)+cy
    
    
        return x_m,y_m
    

    def GetCameraPosition(self):

        # Project point [0 0 0]
        x = 0 - self.translation_vector[0]
        y = 0 - self.translation_vector[1]
        z = 0 - self.translation_vector[2]
        x0 = self.rotation_matrix[0,0] * x + self.rotation_matrix[1,0] * y + self.rotation_matrix[2,0] * z
        y0 = self.rotation_matrix[0,1] * x + self.rotation_matrix[1,1] * y + self.rotation_matrix[2,1] * z
        z0 = self.rotation_matrix[0,2] * x + self.rotation_matrix[1,2] * y + self.rotation_matrix[2,2] * z

        ## Pitch, from z : Project point [0 0 1]
        x = 0 - self.translation_vector[0]
        y = 0 - self.translation_vector[1]
        z = 1 - self.translation_vector[2]
        x1 = self.rotation_matrix[0,0] * x + self.rotation_matrix[1,0] * y + self.rotation_matrix[2,0] * z
        y1 = self.rotation_matrix[0,1] * x + self.rotation_matrix[1,1] * y + self.rotation_matrix[2,1] * z
        z1 = self.rotation_matrix[0,2] * x + self.rotation_matrix[1,2] * y + self.rotation_matrix[2,2] * z
        pitch1 = np.arctan2((z1 - z0),(x1 - x0))
        if(pitch1>np.pi):
            pitch1 = pitch1 - 2*np.pi


        ##Yaw, from z
        yaw1 = np.arctan2((y1 - y0), (x1 - x0))
        if(yaw1>np.pi):
            yaw1 = yaw1 - 2*np.pi


        ## Yaw, from x : Project point [1 0 0]
        x = 1 - self.translation_vector[0]
        y = 0 - self.translation_vector[1]
        z = 0 - self.translation_vector[2]
        x1 = self.rotation_matrix[0,0] * x + self.rotation_matrix[1,0] * y + self.rotation_matrix[2,0] * z
        y1 = self.rotation_matrix[0,1] * x + self.rotation_matrix[1,1] * y + self.rotation_matrix[2,1] * z
        z1 = self.rotation_matrix[0,2] * x + self.rotation_matrix[1,2] * y + self.rotation_matrix[2,2] * z
        yaw2 = (np.arctan2((y1 - y0), (x1 - x0)) + np.pi/2)
        
        if(yaw2>np.pi):
            yaw2 = yaw2 - 2*np.pi
            

        ## Roll , from x 
        roll1 = (np.arctan2((z1 - z0), (y1 - y0)) + np.pi)
        if(roll1>np.pi):
            roll1 = roll1 - 2*np.pi


        ## Roll , from y : Project point [0 1 0]
        x = 0 - self.translation_vector[0]
        y = 1 - self.translation_vector[1]
        z = 0 - self.translation_vector[2]
        x1 = self.rotation_matrix[0,0] * x + self.rotation_matrix[1,0] * y + self.rotation_matrix[2,0] * z
        y1 = self.rotation_matrix[0,1] * x + self.rotation_matrix[1,1] * y + self.rotation_matrix[2,1] * z
        z1 = self.rotation_matrix[0,2] * x + self.rotation_matrix[1,2] * y + self.rotation_matrix[2,2] * z
        roll2 = (np.arctan2((z1 - z0), (y1 - y0)) + np.pi/2)
        if(roll2>np.pi):
            roll2 = roll2 - 2*np.pi

        ## Pitch, from y
        pitch2 = (np.arctan2((z1 - z0),(x1 - x0)) + np.pi/2)
        if(pitch2>np.pi):
            pitch2 = pitch2 - 2*np.pi

        pitch_rad = (pitch1 + pitch2)/2
        yaw_rad = (yaw1 + yaw2)/2
        roll_rad = (roll1 + roll2)/2

        return x0,y0,z0,pitch_rad,yaw_rad,roll_rad
    
    def worldToImage_opcv(self,x,y,z):
        world_points = np.array([[x,y,z]],dtype = 'float32')
        imgpts, _ = cv2.projectPoints(world_points, self.rotation_matrix, self.translation_vector, self.camera_matrix, self.distortion_coefficients)

        u = int(min(max(0,imgpts[0][0][0]),self.ImageWidth-1))
        v = int(min(max(0,imgpts[0][0][1]),self.ImageHeight-1))
        
        return u,v