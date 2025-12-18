import h5py
import numpy as np


class FeatureListSaveLoad(object):
    def __init__(self, data):
        self.set_data(data)
        
    def save(self, path):
        if self.data is None:
            raise ValueError("Data is None")
        self.data.tofile(path)
        # with open(path, 'wb') as f:
        #     np.save(f, self.data)
        return self
    
    def set_data(self, data):
        assert data.ndim == 2, f"Data should be 2D, but got {data.ndim}D"
        self.data = data
        self.dim = data.shape[1]
        return self
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            vector = np.load(f)
        return FeatureListSaveLoad(vector)
    
    @staticmethod
    def load_range(path, start, end, dim):
        """Load a partial range [start:end] from a raw float32 file."""
        try:
            with open(path, 'rb') as f:
                offset = start * dim * np.dtype(np.float32).itemsize  # 自动拿float32的字节数
                f.seek(offset, 0)  # 0 means from start of the file
                count = (end - start) * dim
                vector = np.fromfile(f, dtype=np.float32, count=count)
                vector = vector.reshape((end - start, dim))
        except Exception as e:
            raise ValueError(f"Error loading range from {path}: start={start}, end={end}, dim={dim}\nError: {e}")
        return FeatureListSaveLoad(vector)

class AgibotActionState(object):
    def __init__(self):
        # self.action_end_orientation = None
        # self.action_end_position = None
        self.action_head_position = None
        self.action_joint_position = None
        self.action_robot_velocity = None
        self.action_waist_position = None
        self.action_effector_position = None
        
        # (x, y, z, w) quaternion
        self.state_end_orientation = None
        self.state_end_position = None
        
        self.state_joint_position = None
        self.state_effector_position = None
        self.state_head_position = None
        self.state_waist_position = None
        
        # self.state_joint_current_value = None
        # self.state_robot_orientation = None
        # self.state_robot_position = None
        
        
    
    @property
    def effector_dim(self):
        return self.action_effector_position.shape[1] if self.action_effector_position is not None else 0
    
    @property
    def length(self):
        if self.action_effector_position is None:
            raise ValueError("action_effector_position is None")
        return self.action_effector_position.shape[0]
    
    def get_dim_list(self):
        l = []
        for prop in self.get_property_list():
            if getattr(self, prop) is None:
                raise ValueError(f"Property {prop} is None")
            l.append(getattr(self, prop).shape[1])
        return l
    
    @staticmethod
    def get_effector_dim(dim_list):
        return dim_list[4]
    
    def get_property_list(self):
        l = [
            # 'action_end_orientation',
            # 'action_end_position',
            'action_head_position',
            'action_joint_position',
            'action_robot_velocity',
            'action_waist_position',
            'action_effector_position',
            'state_end_orientation',
            'state_end_position',
            'state_head_position',  
            'state_joint_position',
            # 'state_joint_current_value',
            # 'state_robot_orientation',
            # 'state_robot_position',
            'state_waist_position',
            'state_effector_position',
        ]
        return l
    
    def _check_same_length(self):
        property_list = self.get_property_list()
        length = self.action_effector_position.shape[0]
        for prop in property_list:
            if getattr(self, prop).shape[0] != length:
                print(f"Property {prop} has different length: {getattr(self, prop).shape[0]} != {length}")
                return False
        return True
    
    @classmethod
    def load_from_h5(cls, h5_file):
        if isinstance(h5_file, str):
            h5_file = h5py.File(h5_file, 'r')
        obj = cls()
        obj.action_effector_position = h5_file['action']['effector']['position'][:]
        action_effector_index = h5_file['action']['effector']['index'][:]
        # assert (action_effector_index == np.arange(obj.action_effector_position.shape[0])).all()
        # obj.action_end_orientation = h5_file['action']['end']['orientation'][:]
        # obj.action_end_position = h5_file['action']['end']['position'][:]
        # obj.action_end_orientation = obj.action_end_orientation.reshape(obj.action_end_position.shape[0], -1)
        # obj.action_end_position = obj.action_end_position.reshape(obj.action_end_position.shape[0], -1)
        # action_end_index = h5_file['action']['end']['index'][:]
        # assert (action_end_index == np.arange(obj.action_end_position.shape[0])).all()
        obj.action_head_position = h5_file['action']['head']['position'][:]
        action_head_index = h5_file['action']['head']['index'][:]
        # assert (action_head_index == np.arange(obj.action_head_position.shape[0])).all()
        obj.action_joint_position = h5_file['action']['joint']['position'][:]
        action_joint_index = h5_file['action']['joint']['index'][:]
        # assert (action_joint_index == np.arange(obj.action_joint_position.shape[0])).all()
        obj.action_robot_velocity = h5_file['action']['robot']['velocity'][:]
        action_robot_index = h5_file['action']['robot']['index'][:]
        # assert (action_robot_index == np.arange(obj.action_robot_velocity.shape[0])).all()
        obj.action_waist_position = h5_file['action']['waist']['position'][:]
        action_waist_index = h5_file['action']['waist']['index'][:]
        # assert (action_waist_index == np.arange(obj.action_waist_position.shape[0])).all()
        
        obj.state_effector_position = h5_file['state']['effector']['position'][:]
        obj.state_end_orientation = h5_file['state']['end']['orientation'][:]
        obj.state_end_orientation = obj.state_end_orientation.reshape(obj.state_effector_position.shape[0], -1)
        obj.state_end_position = h5_file['state']['end']['position'][:]
        obj.state_end_position = obj.state_end_position.reshape(obj.state_effector_position.shape[0], -1)
        obj.state_head_position = h5_file['state']['head']['position'][:]
        obj.state_joint_position = h5_file['state']['joint']['position'][:]
        # obj.state_joint_current_value = h5_file['state']['joint']['current_value'][:]
        # obj.state_robot_orientation = h5_file['state']['robot']['orientation'][:]
        # obj.state_robot_position = h5_file['state']['robot']['position'][:]
        obj.state_waist_position = h5_file['state']['waist']['position'][:]
        
        if obj._check_same_length() == False:
            raise ValueError("Properties have different lengths")
        return obj
    
    def get_action_vector(self):
        property_list = self.get_property_list()
        action_vector = []
        for prop in property_list:
            if getattr(self, prop) is None:
                raise ValueError(f"Property {prop} is None")
            action_vector.append(getattr(self, prop))
        action_vector = np.concatenate(action_vector, axis=1)
        action_vector = action_vector.astype(np.float32)
        return action_vector
    
    def set_from_vector(self, vector, dim_list):
        property_list = self.get_property_list()
        start = 0
        for i, prop in enumerate(property_list):
            end = start + dim_list[i]
            setattr(self, prop, vector[:, start:end])
            start = end
        self._check_same_length()
        return self
    
    def save_to_path(self, path):
        vector = self.get_action_vector()
        feature = FeatureListSaveLoad(vector)
        feature.save(path)

    @staticmethod
    def load_from_path(path, dim_list):
        expected_dim = sum(dim_list)
        features = FeatureListSaveLoad.load(path)
        
        if features.dim != expected_dim:
            raise ValueError(f"Feature dimension {features.dim} does not match expected {expected_dim}. "
                            f"Expected dimensions: {dim_list}")
        
        vector = features.data
        agibot_action_state = AgibotActionState()
        agibot_action_state.set_from_vector(vector, dim_list)
        return agibot_action_state

    @staticmethod
    def load_range_from_path(path, dim_list, start, end):
        expected_dim = sum(dim_list)
        features = FeatureListSaveLoad.load_range(path, start, end, expected_dim)
        vector = features.data
        agibot_action_state = AgibotActionState()
        agibot_action_state.set_from_vector(vector, dim_list)
        return agibot_action_state

    
        
        
        
        
        
# action
#   effector
#     force
#       /action/effector/force (0,)
#     index
#       /action/effector/index (1278,)
#     position
#       /action/effector/position (1278, 2)
#   end
#     index
#       /action/end/index (1221,)
#     orientation
#       /action/end/orientation (1278, 2, 4)
#     position
#       /action/end/position (1278, 2, 3)
#   head
#     index
#       /action/head/index (1278,)
#     position
#       /action/head/position (1278, 2)
#   joint
#     effort
#       /action/joint/effort (0,)
#     index
#       /action/joint/index (1221,)
#     position
#       /action/joint/position (1278, 14)
#     velocity
#       /action/joint/velocity (0,)
#   robot
#     index
#       /action/robot/index (1278,)
#     orientation
#       /action/robot/orientation (0,)
#     position
#       /action/robot/position (0,)
#     velocity
#       /action/robot/velocity (1278, 2)
#   waist
#     index
#       /action/waist/index (1278,)
#     position
#       /action/waist/position (1278, 2)
# state
#   effector
#     force
#       /state/effector/force (0,)
#     position
#       /state/effector/position (1278, 2)
#   end
#     angular
#       /state/end/angular (0,)
#     orientation
#       /state/end/orientation (1278, 2, 4)
#     position
#       /state/end/position (1278, 2, 3)
#     velocity
#       /state/end/velocity (0,)
#     wrench
#       /state/end/wrench (0,)
#   head
#     effort
#       /state/head/effort (0,)
#     position
#       /state/head/position (1278, 2)
#     velocity
#       /state/head/velocity (0,)
#   joint
#     current_value
#       /state/joint/current_value (1278, 14)
#     effort
#       /state/joint/effort (0,)
#     position
#       /state/joint/position (1278, 14)
#     velocity
#       /state/joint/velocity (0,)
#   robot
#     orientation
#       /state/robot/orientation (1278, 4)
#     orientation_drift
#       /state/robot/orientation_drift (0,)
#     position
#       /state/robot/position (1278, 3)
#     position_drift
#       /state/robot/position_drift (0,)
#   waist
#     effort
#       /state/waist/effort (0,)
#     position
#       /state/waist/position (1278, 2)
#     velocity
#       /state/waist/velocity (0,)
# timestamp
#   /timestamp (1278,)
# force
#   /state/effector/force (0,)
# position
#   /state/effector/position (1278, 2)action
#   effector
#     force
#       /action/effector/force (0,)
#     index
#       /action/effector/index (1278,)
#     position
#       /action/effector/position (1278, 2)
#   end
#     index
#       /action/end/index (1221,)
#     orientation
#       /action/end/orientation (1278, 2, 4)
#     position
#       /action/end/position (1278, 2, 3)
#   head
#     index
#       /action/head/index (1278,)
#     position
#       /action/head/position (1278, 2)
#   joint
#     effort
#       /action/joint/effort (0,)
#     index
#       /action/joint/index (1221,)
#     position
#       /action/joint/position (1278, 14)
#     velocity
#       /action/joint/velocity (0,)
#   robot
#     index
#       /action/robot/index (1278,)
#     orientation
#       /action/robot/orientation (0,)
#     position
#       /action/robot/position (0,)
#     velocity
#       /action/robot/velocity (1278, 2)
#   waist
#     index
#       /action/waist/index (1278,)
#     position
#       /action/waist/position (1278, 2)
# state
#   effector
#     force
#       /state/effector/force (0,)
#     position
#       /state/effector/position (1278, 2)
#   end
#     angular
#       /state/end/angular (0,)
#     orientation
#       /state/end/orientation (1278, 2, 4)
#     position
#       /state/end/position (1278, 2, 3)
#     velocity
#       /state/end/velocity (0,)
#     wrench
#       /state/end/wrench (0,)
#   head
#     effort
#       /state/head/effort (0,)
#     position
#       /state/head/position (1278, 2)
#     velocity
#       /state/head/velocity (0,)
#   joint
#     current_value
#       /state/joint/current_value (1278, 14)
#     effort
#       /state/joint/effort (0,)
#     position
#       /state/joint/position (1278, 14)
#     velocity
#       /state/joint/velocity (0,)
#   robot
#     orientation
#       /state/robot/orientation (1278, 4)
#     orientation_drift
#       /state/robot/orientation_drift (0,)
#     position
#       /state/robot/position (1278, 3)
#     position_drift
#       /state/robot/position_drift (0,)
#   waist
#     effort
#       /state/waist/effort (0,)
#     position
#       /state/waist/position (1278, 2)
#     velocity
#       /state/waist/velocity (0,)
# timestamp
#   /timestamp (1278,)
# force
#   /state/effector/force (0,)
# position
#   /state/effector/position (1278, 2)