class CameraWrapper:
    def __init__(self, scene):
        """
        Initialize the wrapper with a ThreeDScene instance and read current camera state
        """
        self.scene = scene
        
        # Read current camera state
        camera = self.scene.camera
        self.current_phi = camera.phi
        self.current_theta = camera.theta
        self.current_gamma = camera.gamma
        self.current_zoom = camera.get_zoom()
        self.current_focal_distance = camera.get_focal_distance()
        self.current_frame_center = camera.get_frame_center()
    
    def scale(self, factor, animate=False):
        """
        Scale the view by adjusting the zoom
        """
        new_zoom = self.current_zoom * (1/factor)  # Inverse because zoom works opposite to scale
        
        if animate:
            self.scene.move_camera(zoom=new_zoom)
        else:
            self.scene.set_camera_orientation(zoom=new_zoom)
            
        self.current_zoom = new_zoom
        return self
    
    def move_to(self, point, animate=False):
        """
        Move the camera center to a specific point
        """
        if animate:
            self.scene.move_camera(frame_center=point)
        else:
            self.scene.set_camera_orientation(frame_center=point)
            
        self.current_frame_center = point
        return self
    
    def shift(self, vector, animate=False):
        """
        Shift the camera by a vector
        """
        new_center = (self.current_frame_center or [0, 0, 0]) + vector
        return self.move_to(new_center, animate)
    
    def to_corner(self, corner=None, animate=False):
        """
        Move camera to a corner. corner can be UL, UR, DL, DR, etc.
        This is a simplified version - you might want to adjust the actual positions
        """
        corner_positions = {
            "UL": [-4, 4, 0],
            "UR": [4, 4, 0],
            "DL": [-4, -4, 0],
            "DR": [4, -4, 0],
            None: [0, 0, 0]  # Center
        }
        return self.move_to(corner_positions[corner], animate)
    
    @property
    def animate(self):
        """
        Return a version of this wrapper that will animate all transformations
        """
        return AnimatingCameraWrapper(self)

class AnimatingCameraWrapper:
    def __init__(self, camera_wrapper):
        self.camera_wrapper = camera_wrapper
    
    def scale(self, factor):
        return self.camera_wrapper.scale(factor, animate=True)
    
    def move_to(self, point):
        return self.camera_wrapper.move_to(point, animate=True)
    
    def shift(self, vector):
        return self.camera_wrapper.shift(vector, animate=True)
    
    def to_corner(self, corner=None):
        return self.camera_wrapper.to_corner(corner, animate=True)