import cv2
import mediapipe as mp
import numpy as np
import os
from custom.objloader_simple import *
from custom.utils import *
from custom.videosource import WebcamSource
from custom.face_geometry import (
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)


# Create mediapipe variables
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_connections = mp.solutions.face_mesh_connections
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)

points_idx = [33, 263, 61, 291, 199]
points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

frame_height, frame_width, channels = (360, 720, 3)

# Pseudo camera internals
focal_length = frame_width
center = (frame_width / 2, frame_height / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype="double",
)

# Distortion coefficients
dist_coeff = np.zeros((4, 1))


def render_lower(img, faces, vertices, camera_parameters, texture, base_point, head_up, pass_factor, left_oriented):
    """Render the loaded obj model into the current video frame for lower parts

    Args:
        img (np.array): Input image
        faces (list): Faces of the object
        vertices (list): Vertices of the object
        camera_parameters (dict): Camera parameters
        texture (np.array): Texture image
        base_point (tuple): Reference point for render
        head_up (bool): Boolean for head position
        pass_factor (int): Constant for render
        left_oriented (bool): Boolean for head left position

    Returns:
        np.array: Output image
    """
    texture_height = texture.shape[0] - 6
    texture_width = texture.shape[1] - 1
    src_points = np.array([[0, 0], [0, texture_height], [texture_width, texture_height], [texture_width, 0]], dtype=np.float32)
    src_points_triangle = np.array([[0, 0], [texture_height, 0], [0, texture_width]], dtype=np.float32)
    
    # Using scale matrix to upscale the model
    scale_matrix = np.eye(3) * 1
    
    # Downscale the image
    down_width = 320 #1280
    down_height = 180 #720
    down_points = (down_width, down_height)
    img = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)
    
    for idx, face in enumerate(faces):
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # Render model in the landmarks. To do so, model points must be displaced
        points = np.array([[p[0], p[1], p[2] - 6] for p in points])
        
        # Pass rendering if the point is behind the reference point
        pass_render = False
        for point in points:
            if point[2] < base_point[2] - pass_factor:
                pass_render = True
                
        if pass_render:
            if head_up:
                continue
        
        # Get projected points mp_rotation_vector, mp_translation_vector, camera_matrix, dist_coeff
        (pointer2D, jacobian) = cv2.projectPoints(
            np.array([points]),
            camera_parameters['mp_rotation_vector'],
            camera_parameters['mp_translation_vector'],
            camera_parameters['camera_matrix'],
            camera_parameters['dist_coeff'],
        )
        point_2d = pointer2D.squeeze().astype(np.float32)
        
        if head_up:
            if left_oriented:
                point_2d[:,0] += 5
            else:
                point_2d[:,0] -= 5
                
        point_2d = point_2d / 2

        cv2.fillConvexPoly(img, point_2d.astype(int), (1.0, 1.0, 1.0), 16, 0)
        
        # Add texture with perspective transform and warp
        if len(point_2d) > 3:
            pers = cv2.getPerspectiveTransform(src_points.astype(np.float32), point_2d.astype(np.float32))
            im_temp = cv2.warpPerspective(texture, pers, (img.shape[1], img.shape[0]))
            img = img + im_temp
        elif len(point_2d) == 3:
            # Add texture with affine transform and warp
            affine = cv2.getAffineTransform(src_points_triangle.astype(np.float32), point_2d.astype(np.float32))
            im_temp = cv2.warpAffine(texture, affine, (img.shape[1], img.shape[0]))
            img = img + im_temp
            
    # Upscale the image using new  width and height
    up_width = 640
    up_height = 360
    up_points = (up_width, up_height)
    img = cv2.resize(img, up_points, interpolation= cv2.INTER_LINEAR)

    return img


def render_upper(img, faces, vertices, camera_parameters, texture, base_point, head_up, pass_factor):
    """Render the loaded obj model into the current video frame for upper parts

    Args:
        img (np.array): Input image
        faces (list): Faces of the object
        vertices (list): Vertices of the object
        camera_parameters (dict): Camera parameters
        texture (np.array): Texture image
        base_point (tuple): Reference point for render
        head_up (bool): Boolean for head position
        pass_factor (int): Constant for render
        left_oriented (bool): Boolean for head left position

    Returns:
        np.array: Output image
    """
    texture_height = texture.shape[0] - 6
    texture_width = texture.shape[1] - 1
    src_points = np.array([[0, 0], [0, texture_height], [texture_width, texture_height], [texture_width, 0]], dtype=np.float32)
    src_points_triangle = np.array([[0, 0], [texture_height, 0], [0, texture_width]], dtype=np.float32)
    
    # Using scale matrix to upscale the model
    scale_matrix = np.eye(3) * 1
    
    # Downscale the image
    down_width = 320 #1280
    down_height = 180 #720
    down_points = (down_width, down_height)
    img = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)
    
    for idx, face in enumerate(faces):
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # Render model in the landmarks. To do so, model points must be displaced
        points = np.array([[p[0], p[1], p[2] - 6] for p in points])
        
        # Pass rendering if the point is behind the reference point
        pass_render = False
        for point in points:
            if point[2] < base_point[2] - pass_factor:
                pass_render = True
                
        if pass_render:
            continue
        
        # Get projected points
        (pointer2D, jacobian) = cv2.projectPoints(
            np.array([points]),
            camera_parameters['mp_rotation_vector'],
            camera_parameters['mp_translation_vector'],
            camera_parameters['camera_matrix'],
            camera_parameters['dist_coeff'],
        )
        point_2d = pointer2D.squeeze().astype(np.float32)
        point_2d = point_2d / 2
        
        cv2.fillConvexPoly(img, point_2d.astype(int), (1.0, 1.0, 1.0), 16, 0)
        
        # Add texture with perspective transform and warp
        if len(point_2d) > 3:
            pers = cv2.getPerspectiveTransform(src_points.astype(np.float32), point_2d.astype(np.float32))
            im_temp = cv2.warpPerspective(texture, pers, (img.shape[1], img.shape[0]))
            img = img + im_temp
        elif len(point_2d) == 3:
            # Add texture with affine transform and warp
            affine = cv2.getAffineTransform(src_points_triangle.astype(np.float32), point_2d.astype(np.float32))
            im_temp = cv2.warpAffine(texture, affine, (img.shape[1], img.shape[0]))
            img = img + im_temp
    
    # Upscale the image using new  width and height
    up_width = 640
    up_height = 360
    up_points = (up_width, up_height)
    img = cv2.resize(img, up_points, interpolation= cv2.INTER_LINEAR)

    return img


def render(img, faces, vertices, camera_parameters, texture, base_point, head_up):
    """Render the loaded obj model into the current video frame

    Args:
        img (np.array): Input image
        faces (list): Faces of the object
        vertices (list): Vertices of the object
        camera_parameters (dict): Camera parameters
        texture (np.array): Texture image
        base_point (tuple): Reference point for render
        head_up (bool): Boolean for head position

    Returns:
        np.array: Output image
    """
    texture_height = texture.shape[0] - 6
    texture_width = texture.shape[1] - 1
    src_points = np.array([[0, 0], [0, texture_height], [texture_width, texture_height], [texture_width, 0]], dtype=np.float32)
    src_points_triangle = np.array([[0, 0], [texture_height, 0], [0, texture_width]], dtype=np.float32)
    
    # Using scale matrix to upscale the model
    scale_matrix = np.eye(3) * 1
    
    # Downscale the image
    down_width = 320 #1280
    down_height = 180 #720
    down_points = (down_width, down_height)
    img = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)
    
    for idx, face in enumerate(faces):
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # Render model in the landmarks. To do so, model points must be displaced
        points = np.array([[p[0], p[1], p[2] - 6] for p in points])
        
        # Pass rendering if the point is behind the reference point
        pass_render = False
        for point in points:
            if point[2] < base_point[2] - 2:
                pass_render = True
                
        if pass_render:
            if not (0 < idx < 64):
                continue
            else:
                if head_up:
                    continue
        
        # Get projected points
        (pointer2D, jacobian) = cv2.projectPoints(
            np.array([points]),
            camera_parameters['mp_rotation_vector'],
            camera_parameters['mp_translation_vector'],
            camera_parameters['camera_matrix'],
            camera_parameters['dist_coeff'],
        )
        point_2d = pointer2D.squeeze().astype(np.float32)
        point_2d = point_2d / 2
        
        cv2.fillConvexPoly(img, point_2d.astype(int), (1.0, 1.0, 1.0), 16, 0)
        
        # Add texture with perspective transform and warp
        if len(point_2d) > 3:
            pers = cv2.getPerspectiveTransform(src_points.astype(np.float32), point_2d.astype(np.float32))
            im_temp = cv2.warpPerspective(texture, pers, (img.shape[1], img.shape[0]))
            img = img + im_temp
        elif len(point_2d) == 3:
            # Add texture with affine transform and warp
            affine = cv2.getAffineTransform(src_points_triangle.astype(np.float32), point_2d.astype(np.float32))
            im_temp = cv2.warpAffine(texture, affine, (img.shape[1], img.shape[0]))
            img = img + im_temp
            
    # Upscale the image using new  width and height
    up_width = 640
    up_height = 360
    up_points = (up_width, up_height)
    img = cv2.resize(img, up_points, interpolation= cv2.INTER_LINEAR)

    return img


def main():
    source = WebcamSource(flip=True)

    refine_landmarks = True
    
    dir_name = os.getcwd()
    # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, 'models/anime_hat.obj'), swapyz=False)
    
    # Load texture image
    texture = cv2.imread(os.path.join(dir_name, 'models/texture_small.jpg'), cv2.IMREAD_UNCHANGED)
    
    # Reversing the faces because we need to change the render order
    faces = obj.faces
    faces.reverse()
    
    # Divide faces for different rendering scenarios
    hat_lower = obj.faces[:64]
    hat_upper = obj.faces[65:]
    vertices = obj.vertices

    # These are taken from mediapipe (PCF)
    pcf = PCF(
        near=1,
        far=10000,
        frame_height=frame_height,
        frame_width=frame_width,
        fy=camera_matrix[1, 1],
    )

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        for idx, (frame, frame_rgb) in enumerate(source):
            results = face_mesh.process(frame_rgb)
            multi_face_landmarks = results.multi_face_landmarks

            if multi_face_landmarks:
                face_landmarks = multi_face_landmarks[0]
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )
                chin = landmarks[152]
                forehead = landmarks[10]
                left_ear = landmarks[234]
                right_ear = landmarks[454]
                landmarks = landmarks.T

                if refine_landmarks:
                    landmarks = landmarks[:, :468]

                _, pose_transform_mat = get_metric_landmarks(
                    landmarks.copy(), pcf
                )

                # Calculate extrinsic values
                pose_transform_mat[1:3, :] = -pose_transform_mat[1:3, :]
                mp_rotation_vector, _ = cv2.Rodrigues(pose_transform_mat[:3, :3])
                mp_translation_vector = pose_transform_mat[:3, 3, None]
                
                camera_parameters = {
                    'mp_rotation_vector': mp_rotation_vector,
                    'mp_translation_vector': mp_translation_vector,
                    'camera_matrix': camera_matrix,
                    'dist_coeff': dist_coeff
                }
                
                # To render only required part, using this point as reference
                base_point = (forehead[0], forehead[1], forehead[2] - 8)
                
                # This condition checks the head position whether it is up or down
                if chin[2] < forehead[2]:
                    # This condition checks the head whether it is turned left o right
                    if left_ear[2] < right_ear[2]:
                        frame = render_upper(frame, hat_upper, vertices, camera_parameters, texture=texture, base_point=base_point, head_up=True, pass_factor=-2)
                        frame = render_lower(frame, hat_lower, vertices, camera_parameters, texture=texture, base_point=base_point, head_up=True, pass_factor=2, left_oriented=True)
                    else:
                        frame = render_upper(frame, hat_upper, vertices, camera_parameters, texture=texture, base_point=base_point, head_up=True, pass_factor=-2)
                        frame = render_lower(frame, hat_lower, vertices, camera_parameters, texture=texture, base_point=base_point, head_up=True, pass_factor=2, left_oriented=False)
                else:
                    frame = render(frame, faces, vertices, camera_parameters, texture=texture, base_point=base_point, head_up=False)
                        
            source.show(frame)


if __name__ == "__main__":
    main()
