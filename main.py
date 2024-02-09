import open3d as o3d
import numpy as np
import copy
import time
import math

# MODE = 1
# MODE = 2
# MODE = 3
# MODE = 4
# MODE = 5
# MODE = 11

# EDGE_PICK = False
# EDGE_PICK = True

# PICK_TYPE = "flat"
# PICK_TYPE = "edge"
PICK_TYPE = "corner"

if PICK_TYPE == "edge":
    coords = [0.69, -0.1, -0.4075] 
    # coords = [0.69, -0.15, -0.4075] 
elif PICK_TYPE == "corner":
    coords = [0.69, -0.145, -0.4075] 
elif PICK_TYPE == "flat":
    coords = [0.7736653033783994, -0.06371333498628079, -0.4075]

#================================================
# Display the results of ICP registration. source and target are open3d point clouds. 
# Transformation is an affine transformation matrix that positions one point cloud over the other
#================================================
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


#================================================
# Display origin on point cloud
#================================================
def displayOrigin(point_cloud):

    zcone = o3d.geometry.TriangleMesh.create_cone(radius = 0.01, height = 0.05)
    R = zcone.get_rotation_matrix_from_xyz((0, 0, -np.pi))
    zcone.rotate(R , [0,0,0])
    zcone.paint_uniform_color([0.0, 0.0, 1.0])

    ycone = o3d.geometry.TriangleMesh.create_cone(radius = 0.01, height = 0.05)
    R = ycone.get_rotation_matrix_from_xyz((-np.pi/2, 0, -np.pi/2))
    ycone.rotate(R , [0,0,0])
    ycone.paint_uniform_color([0.0, 1.0, 0.0])

    xcone = o3d.geometry.TriangleMesh.create_cone(radius = 0.01, height = 0.05)
    R = xcone.get_rotation_matrix_from_xyz((0, np.pi/2, -np.pi/2))
    xcone.rotate(R , [0,0,0])
    xcone.paint_uniform_color([1.0, 0.0, 0.0])
    

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(point_cloud)
    viewer.add_geometry(zcone)
    viewer.add_geometry(ycone)
    viewer.add_geometry(xcone)
    viewer.run()
    viewer.destroy_window()

# def calc_registration(gripper_pcd, gripper_pcd2, pcd, threshold, trans_init, trans):
#     pcd.estimate_normals()
#     draw_registration_result(gripper_pcd, pcd, trans_init)
#     print("Initial alignment")
#     evaluation = o3d.pipelines.registration.evaluate_registration(gripper_pcd, pcd,
#                                                         threshold, trans_init)
#     print(evaluation)

#     print("Apply point-to-point ICP")
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#         gripper_pcd, pcd, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
#     print(reg_p2p)
#     print("Transformation is:")
#     print(reg_p2p.transformation)
#     print("")
#     draw_registration_result(gripper_pcd, pcd, reg_p2p.transformation)
#     gripper_pcd2.transform(trans)
#     draw_registration_result(gripper_pcd2, pcd, reg_p2p.transformation)

#     print("Apply point-to-plane ICP")
#     reg_p2l = o3d.pipelines.registration.registration_icp(
#         gripper_pcd, pcd, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
#     print(reg_p2l)
#     print("Transformation is:")
#     print(reg_p2l.transformation)
#     print("")
#     draw_registration_result(gripper_pcd, pcd, reg_p2l.transformation)
#     # gripper_pcd2.transform(trans)
#     draw_registration_result(gripper_pcd2, pcd, reg_p2l.transformation)


#================================================
# Returns coordinates for the 8 corners of a bounding cube centered on the [X, Y, X] position of coord
# size/2 is length of a cube edge. Units are m
#================================================
def boundingBoxPoints(coord, size):
    return np.array([
        # top face of bounding box
        [coord[0] + size/2, coord[1] + size/2, coord[2] + size/2],
        [coord[0] + size/2, coord[1] - size/2, coord[2] + size/2],
        [coord[0] - size/2, coord[1] - size/2, coord[2] + size/2],
        [coord[0] - size/2, coord[1] + size/2, coord[2] + size/2],

        # bottom face of bounding box
        [coord[0] + size/2, coord[1] + size/2, coord[2] - size/2],
        [coord[0] + size/2, coord[1] - size/2, coord[2] - size/2],
        [coord[0] - size/2, coord[1] - size/2, coord[2] - size/2],
        [coord[0] - size/2, coord[1] + size/2, coord[2] - size/2],
    ])

#================================================
# Fits a disk point cloud to the point cloud at the pick coords to 
# create an initial guess at pose and returns the transformation 
# matrices for point-to-point ICP and point-to-plane ICP Registration
#================================================
def diskRegistration(point_cloud, disk_mesh, coords):
    USE_SOLID_DISK = True
    pcd = copy.deepcopy(point_cloud)
    mesh = copy.deepcopy(disk_mesh)
    # set threshold for registration algorithm
    threshold = 0.02

    
    # create a point cloud from the mesh
    mesh_pcd = mesh.sample_points_uniformly(number_of_points=50000)
    mesh_pcd.translate([0,0, 0])
    R = mesh_pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh_pcd.rotate(R, [0,0,0])

    # create a "wire" model of the gripper
    disk_pcd = o3d.geometry.PointCloud()
    disk_pcd .points = mesh.vertices
    disk_pcd .colors = mesh.vertex_colors
    disk_pcd.translate([0,0, 0])
    disk_pcd.rotate(R, [0,0,0])    
    disk_pcd.estimate_normals()

    # load scene point cloud
    # pcd = o3d.io.read_point_cloud("colorizedPCD.pcd")
    # shift pick point to origin for cropping 
    coord = [-coords[0], -coords[1], -coords[2]]
    pcd.translate(coord)

    if USE_SOLID_DISK == True:
        o3d.visualization.draw_geometries([mesh_pcd, pcd])
    else:
        o3d.visualization.draw_geometries([disk_pcd, pcd])
    pcd.estimate_normals()

    # initial guess for pose of mesh_pcd
    trans_init = np.asarray([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    
    # initial registration fit values of mesh_pcd to pcd
    print("Initial alignment")
    if USE_SOLID_DISK == True:
        evaluation = o3d.pipelines.registration.evaluate_registration(mesh_pcd, pcd,
                                                            threshold, trans_init)
    else:
        evaluation = o3d.pipelines.registration.evaluate_registration(disk_pcd, pcd,
                                                            threshold, trans_init)

    print(evaluation)

    print("\nApply point-to-point ICP")
    if USE_SOLID_DISK == True:
        reg_p2p = o3d.pipelines.registration.registration_icp(
            mesh_pcd, pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    else:
        reg_p2p = o3d.pipelines.registration.registration_icp(
            disk_pcd, pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    print("\nRegisrtation results:")
    print(reg_p2p)
    print("\nTransformation is:")
    print(reg_p2p.transformation)

    # visualize Registration fit of mesh_pcd to pcd
    if USE_SOLID_DISK == True:
        draw_registration_result(mesh_pcd, pcd, reg_p2p.transformation)
    else:
        draw_registration_result(disk_pcd, pcd, reg_p2p.transformation)

    print("\nApply point-to-plane ICP")
    if USE_SOLID_DISK == True:
        reg_p2l = o3d.pipelines.registration.registration_icp(
            mesh_pcd, pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    else:
        reg_p2l = o3d.pipelines.registration.registration_icp(
            disk_pcd, pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    print(reg_p2l)
    print("\nTransformation is:")
    print(reg_p2l.transformation)

    # visualize Registration fit of mesh_pcd to pcd
    # draw_registration_result(mesh_pcd, pcd, reg_p2l.transformation)

    # return transformation results
    return reg_p2p.transformation, reg_p2l.transformation

#================================================
# convert degrees to Radian
#================================================
def d2r(theta):
    return theta*math.pi/180

#================================================
# convert Radians to degrees
#================================================
def r2d(theta):
    return theta*180/math.pi

#================================================
# convert from rotation matrix to rotation vector
#================================================
def affinerotmat2pose(rotmat):
    print(rotmat)
    # array to matrix
    r11 = rotmat[0][0]
    r21 = rotmat[0][1]
    r31 = rotmat[0][2]
    r41 = rotmat[0][3]
    r12 = rotmat[1][0]
    r22 = rotmat[1][1]
    r32 = rotmat[1][2]
    r42 = rotmat[1][3]
    r13 = rotmat[2][0]
    r23 = rotmat[2][1]
    r33 = rotmat[2][2]
    r43 = rotmat[2][3]
    r14 = 0
    r24 = 0
    r34 = 0
    r44 = 0

    print([r11, r21, r31, r41])
    print([r12, r22, r32, r42])
    print([r13, r23, r33, r43])
    print([r14, r24, r34, r44])


    # print("\n rotmat:")
    # print(rotmat)

    # rotation matrix to rotation vector
    val = (r11+r22+r33-1)/2
    if val > 1:
        val = 1
    elif val <-1:
        val = -1
    theta = math.acos(val)
    sth = math.sin(theta)
    # print("Theta:", theta, "\ttheta = 179.99:", d2r(179.99), "\ttheta = -179.99:", d2r(-179.99),"\ttheta = 180:", d2r(180.0), "\tval:", val)
    if ( (theta > d2r(179.99)) or (theta < d2r(-179.99)) ):
        theta = d2r(180)
        # avoid math domain error when r11, r22 and r33 are less than 0
        if r11 < -1:
            r11 = -1
        if r22 < -1:
            r22 = -1
        if r33 < -1:
            r33 = -1
        if (r21 < 0):

            if (r31 < 0):
                ux = math.sqrt((r11+1)/2)
                uy = -math.sqrt((r22+1)/2)
                uz = -math.sqrt((r33+1)/2)
            else:
                ux = math.sqrt((r11+1)/2)
                uy = -math.sqrt((r22+1)/2)
                uz = math.sqrt((r33+1)/2)

        else:
            if (r31 < 0):
                ux = math.sqrt((r11+1)/2)
                uy = math.sqrt((r22+1)/2)
                uz = -math.sqrt((r33+1)/2)
            else:
                ux = math.sqrt((r11+1)/2)
                uy = math.sqrt((r22+1)/2)
                uz = math.sqrt((r33+1)/2)


    else:
        if theta == 0:
            ux = 0
            uy = 0
            uz = 0
        else:
            ux = (r32-r23)/(2*sth)
            uy = (r13-r31)/(2*sth)
            uz = (r21-r12)/(2*sth)


    rotvec = [r41, r42, r43, (theta*ux),(theta*uy),(theta*uz)]

    return rotvec

#================================================
# convert rotVect to affine rotation Matrix
#================================================
def rotVec_to_rotMat_affine(pose):
    xCoord, yCoord, zCoord, Rx, Ry, Rz = pose
    angle = math.sqrt(Rx*Rx + Ry*Ry + Rz*Rz)
    if angle == 0:
        angle = 0.0000000001
    axis_zero = Rx/angle
    axis_one = Ry/angle
    axis_two = Rz/angle
    matrix = axis_to_rotMat_affine(xCoord, yCoord, zCoord, angle, axis_zero, axis_one, axis_two)
    return matrix

#================================================
# convert axis-angles to affine rotation Matrix
#================================================
def axis_to_rotMat_affine(xCoord, yCoord, zCoord, angle, axis_zero, axis_one, axis_two):
    x = copy.copy(axis_zero)
    y = copy.copy(axis_one)
    z = copy.copy(axis_two)
    s = math.sin(angle)
    c = math.cos(angle)
    t = 1.0-c
    magnitude = math.sqrt(x*x + y*y + z*z)
    if magnitude == 0:
        # print("!Error! Magnitude = 0")
        magnitude = 0.0000000001
    else:
        x /= magnitude
        y /= magnitude
        z /= magnitude
    # calulate rotation matrix elements
    m00 = c + x*x*t
    m11 = c + y*y*t
    m22 = c + z*z*t
    tmp1 = x*y*t
    tmp2 = z*s
    m10 = tmp1 + tmp2
    m01 = tmp1 - tmp2
    tmp1 = x*z*t
    tmp2 = y*s
    m20 = tmp1 - tmp2
    m02 = tmp1 + tmp2    
    tmp1 = y*z*t
    tmp2 = x*s
    m21 = tmp1 + tmp2
    m12 = tmp1 - tmp2
    # matrix = [ [m00, m01, m02, xCoord], [m10, m11, m12, yCoord], [m20, m21, m22, zCoord],[0.0, 0.0, 0.0, 1.0] ]
    matrix = [ [m00, m10, m20, xCoord], [m01, m11, m21, yCoord], [m02, m12, m22, zCoord],[0.0, 0.0, 0.0, 1.0] ]
    # matrix = np.array(matrix)
    return matrix


#================================================
# main
#================================================
def main(args):
    gripper_offset = 0.54/2
    
    threshold = 0.02

    # # picking location to check
    # coords = [0.7736653033783994, -0.06371333498628079, -0.4075]
    # # coords = [0.7736653033783994, -0.06371333498628079, -0.4175]
    # # coords = [0.7736653033783994, -0.06371333498628079, -0.155] 
    # # coords = [0.7736653033783994, -0.06371333498628079, -0.165] 
    # coords = [0.69, -0.1, -0.4075] 

    point_cloud = o3d.io.read_point_cloud("colorizedPCD.pcd")
    # import suctionDisk.stl which is used to find best fit of suction cup to point cloud in region around pick point
    if PICK_TYPE == "edge":
        # disk_cloud = o3d.io.read_triangle_mesh("edgePickDisk.stl")
        disk_cloud = o3d.io.read_triangle_mesh("edgePickDisk-longEdge.stl")
        # disk_cloud = o3d.io.read_triangle_mesh("angledPickDisk.stl")
    elif PICK_TYPE == "corner":
        disk_cloud = o3d.io.read_triangle_mesh("cornerPickDisk.stl")
    elif PICK_TYPE == "flat":
        disk_cloud = o3d.io.read_triangle_mesh("suctionDisk.stl")

    # find transforms for the best fit location of disk to the point cloud 
    # at the picking location (x, y, z) given by coords
    trans2, trans3 = diskRegistration(point_cloud, disk_cloud,  coords)

    # shift Z-coord of pick point so coords is on point cloud
    coords[2] += trans2[2][3]
    print("Disk Z shifted by", trans2[2][3], "to", coords[2])

    # list of modes to check. Each mode has its own STL file.
    # modeList = [3, 4, 5, "edge", "flat", "corner"]
    modeList = ["edge", "flat", "corner"]
    for MODE in modeList:

        print("\n------------------------------------------------")
        print("\nChecking MODE", MODE)
        print("------------------------------------------------\n")
        # if EDGE_PICK == False:
        if MODE == 1:
            mesh = o3d.io.read_triangle_mesh("MODE1_setm310.stl")
            gripper_length = 0.513
        elif MODE == 2:
            mesh = o3d.io.read_triangle_mesh("MODE2_setm311.stl")
            gripper_length = 0.513
        elif MODE == 3:
            mesh = o3d.io.read_triangle_mesh("MODE3 - setm312.stl")
            gripper_length = 0.525
            gripper_shift = 0.525 - gripper_length
        elif MODE == 4:
            mesh = o3d.io.read_triangle_mesh("MODE4 - setm317.stl")
            gripper_length = 0.520
            gripper_shift = 0.525 - gripper_length
        elif MODE == 5:
            gripper_length = 0.513
            gripper_shift = 0.525 - gripper_length
            mesh = o3d.io.read_triangle_mesh("MODE5 - setm313.stl")
        elif MODE == "edge":
            mesh = o3d.io.read_triangle_mesh("edgePickGripper-contactArea.stl")
            gripper_length = 0.525
            gripper_shift = 0.525 - gripper_length
        elif MODE == "flat":
            mesh = o3d.io.read_triangle_mesh("flatGripper-contactArea.stl")
            gripper_length = 0.525
            gripper_shift = 0.525 - gripper_length                
        elif MODE == "corner":
            mesh = o3d.io.read_triangle_mesh("cornerPickGripper-contactArea.stl")
            gripper_length = 0.525
            gripper_shift = 0.525 - gripper_length 


        # else:
            
        #     if MODE == 3:
        #         mesh = o3d.io.read_triangle_mesh("edgePickGripper.stl")
        #         gripper_length = 0.525
        #         gripper_shift = 0.525 - gripper_length
        #     elif MODE == 4:
        #         mesh = o3d.io.read_triangle_mesh("edgePickGripper2.stl")
        #         gripper_length = 0.525
        #         gripper_shift = 0.525 - gripper_length                
        #     elif MODE == 5:
        #         mesh = o3d.io.read_triangle_mesh("edgePickGripper3.stl")
        #         gripper_length = 0.525
        #         gripper_shift = 0.525 - gripper_length            
        

        # create ring model of gripper from stl mesh
        mesh_pcd = o3d.geometry.PointCloud()
        mesh_pcd .points = mesh.vertices
        mesh_pcd .colors = mesh.vertex_colors
        mesh_pcd.translate([0,0, -(gripper_shift/2) - gripper_length])
        R = mesh_pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        mesh_pcd.rotate(R, [0,0,0])

        # create a point cloud from stl file mesh and rotate into picking location
        gripper_pcd = mesh.sample_points_uniformly(number_of_points=50000)
        gripper_pcd.translate([0,0, -(gripper_shift/2) - gripper_length])
        gripper_pcd.rotate(R, [0,0,0])

        pcd = copy.deepcopy(point_cloud)

        coord = [-coords[0], -coords[1], -coords[2] + gripper_shift/2]
        # shift point cloud so that the coord becomes the origin (for cropping)
        pcd.translate(coord)

        newTrans = copy.copy(trans2)
        # print("\ninit[0][3]:", newTrans[0][3])
        # print("init[1][3]:", newTrans[1][3])
        # print("init[2][3]:", newTrans[2][3])

        # remove translations
        newTrans[0][3] = 0.0
        newTrans[1][3] = 0.0
        newTrans[2][3] = 0.0

        # print(newTrans)
        # gripperRPY = affinerotmat2pose(newTrans)
        # cloudRPY = copy.copy(gripperRPY)
        # cloudRPY[3] = -cloudRPY[3]
        # cloudRPY[4] = -cloudRPY[4]
        # cloudRPY[4] = -cloudRPY[4]

        # newPCDTrans = rotVec_to_rotMat_affine(cloudRPY)

        # print("Gripper RPY:", gripperRPY)

        no_rotation = [[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]
        # pcd.transform(newPCDTrans)
        # gripper_pcd.transform(newTrans)
        gripper_pcd.transform(newTrans)

        
        draw_registration_result(gripper_pcd, pcd, no_rotation)

        # displayOrigin(pcd)
        
        # crop point cloud using bounding box at origin
        center = np.array([0,0,0])
        r = np.array([[1,0,0], [0,1,0], [0,0,1]]) # no rotation
        # r = np.array([[newTrans[0][0], newTrans[0][1], newTrans[0][2]],
        #      [newTrans[1][0], newTrans[1][1], newTrans[1][2]],
        #      [newTrans[2][0], newTrans[2][1], newTrans[2][2]]])

        size = np.array([0.12,0.12,0.12])
        oriented_bounding_box = o3d.geometry.OrientedBoundingBox(center, r, size)
        point_cloud_crop = pcd.crop(oriented_bounding_box)
        r = np.array([[math.pi/4, -math.pi/4, 0], [math.pi/4, math.pi/4, 0], [0, 0, 1]])
        oriented_bounding_box = o3d.geometry.OrientedBoundingBox(center, r, size)
        point_cloud_crop = pcd.crop(oriented_bounding_box)

        
        
        # draw_registration_result(gripper_pcd, point_cloud_crop, no_rotation)
        

        # compute the distance between each point in the cropped point cloud and the gripper
        dists = point_cloud_crop.compute_point_cloud_distance(gripper_pcd)
        dists = np.asarray(dists)

        print("Total number of distance results: ", len(dists))
        ind = np.asarray(dists > 0.005).nonzero()[0]
        print("# of points with dist > 0.005:", len(ind))
        print("Percent remaining: ", (len(ind)/len(dists))*100)
        print("Percent cropped: ", ((len(dists) - len(ind))/len(dists))*100)

        # crop points at indexes where the dist is > 0.005 for visualization
        pcd_without_gripper = point_cloud_crop.select_by_index(ind)
        o3d.visualization.draw_geometries([pcd_without_gripper])



   




if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
            