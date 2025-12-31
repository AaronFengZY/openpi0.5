import pinocchio as pin
import numpy as np
import json


class G1_FK:
    """
    Forward-kinematics helper for the A2D robot (G1 variant).

    • Builds a reduced Pinocchio model with the hand-closing joints locked.
    • Provides fk() → dict with 4×4 homogeneous transforms for:
        - left_end   (left_base_link)
        - right_end  (right_base_link)
        - head_end   (link_pitch_head)
        - left_grip  (gripper_center)
        - right_grip (right_gripper_center)
    """

    _LOCKED_JOINTS = [
        "left_Right_0_Joint", "left_Right_1_Joint", "left_Right_2_Joint",
        "left_Right_Support_Joint", "right_Right_1_Joint", "right_Right_2_Joint",
        "right_Left_0_Joint", "right_Left_2_Joint", "right_Left_Support_Joint",
        "right_Right_0_Joint", "right_Right_Support_Joint",
        "left_Left_0_Joint", "left_Left_2_Joint", "left_Left_Support_Joint",
        "left_hand_joint1", "right_hand_joint1",
    ]

    # frames that roughly circumscribe each gripper
    _LEFT_BBOX_FRAMES = [
        # palm & supports
        "left_base_link", "gripper_center",
        "left_Left_Pad_Link", "left_Right_Pad_Link",
        "left_Left_Support_Link", "left_Right_Support_Link",
        # NEW – claw tips
        "left_Left_2_Link", "left_Right_2_Link",
        "Link7_l",          # small stub behind the fingers
    ]
    _RIGHT_BBOX_FRAMES = [
        "right_base_link", "right_gripper_center",
        "right_Left_Pad_Link",  "right_Right_Pad_Link",
        "right_Left_Support_Link", "right_Right_Support_Link",
        "right_Left_2_Link", "right_Right_2_Link",
        "Link7_r",
    ]


    def __init__(self, urdf_path: str, package_dirs=None):
        # full model ---------------------------------------------------------
        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path, package_dirs=package_dirs
        )

        # reduced model with finger joints locked ----------------------------
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self._LOCKED_JOINTS,
            reference_configuration=np.zeros(self.robot.model.nq),
        )

        # cache frame IDs we care about --------------------------------------
        f = self._find_frame                     # alias
        self._fid_left_base   = self._find_frame("left_base_link")
        self._fid_right_base  = self._find_frame("right_base_link")
        self._fid_head_pitch  = self._find_frame("link_pitch_head")
        self._fid_left_grip   = self._find_frame("gripper_center")
        self._fid_right_grip  = self._find_frame("right_gripper_center")

        self._fid_left_bbox   = [f(n) for n in self._LEFT_BBOX_FRAMES]
        self._fid_right_bbox  = [f(n) for n in self._RIGHT_BBOX_FRAMES]

    # --------------------------------------------------------------------- #
    # public API                                                            #
    # --------------------------------------------------------------------- #
    def fk(
        self,
        head_position:       list | np.ndarray,
        waist_position:      list | np.ndarray,
        joint_position:      list | np.ndarray,
        z_offset:            float = 0.30,          # keep old 0.3-m palm offset
    ) -> dict:
        """
        Compute forward kinematics for one configuration.

        Parameters
        ----------
        head_position   : [pitch, yaw] (or length-2 np array)
        waist_position  : [pitch, lift] (or length-2 np array)
        joint_position  : 14-DoF arm joints  (7 left + 7 right)
        z_offset        : subtract this from left/right palms (keeps legacy)
        """

        # ── build full q vector expected by the reduced model ──────────────
        if isinstance(waist_position, list | tuple):
            q = np.array([waist_position[1], waist_position[0]]
                         + joint_position
                         + head_position, dtype=float)
        elif isinstance(waist_position, np.ndarray):
            q = np.concatenate(
                (waist_position[1:], waist_position[:1],
                 np.asarray(joint_position, dtype=float),
                 np.asarray(head_position,  dtype=float))
            )
        else:
            raise TypeError("waist_position must be list/tuple or np.ndarray")

        # ── Pinocchio FK ----------------------------------------------------
        model   = self.reduced_robot.model
        data    = model.createData()
        pin.framesForwardKinematics(model, data, q)

        # ── gather transforms ----------------------------------------------
        res = {
            "left_end"  : data.oMf[self._fid_left_base ].copy().np,
            "right_end" : data.oMf[self._fid_right_base].copy().np,
            "head_end"  : data.oMf[self._fid_head_pitch].copy().np,
            "left_grip" : data.oMf[self._fid_left_grip ].copy().np,
            "right_grip": data.oMf[self._fid_right_grip].copy().np,
        }

        # historical 0.3-m adjustment on palm frames (keep behaviour)
        res["left_end"][2,  3] -= z_offset
        res["right_end"][2, 3] -= z_offset
        res["left_grip"][2,  3] -= z_offset
        res["right_grip"][2, 3] -= z_offset

        # axis-aligned bounding boxes
        res["left_bbox"]  = self._bbox_world(self._fid_left_bbox,
                                             data, z_offset)
        res["right_bbox"] = self._bbox_world(self._fid_right_bbox,
                                             data, z_offset)
        
        res["left_points"]  = self._points_world(self._fid_left_bbox, data, z_offset)
        res["right_points"] = self._points_world(self._fid_right_bbox, data, z_offset)

        return res

    # --------------------------------------------------------------------- #
    # helpers                                                               #
    # --------------------------------------------------------------------- #
    def _find_frame(self, frame_name: str) -> int:
        for fid, frame in enumerate(self.reduced_robot.model.frames):
            if frame.name == frame_name:
                return fid
        raise ValueError(f"Frame “{frame_name}” not found in reduced model.")
    
    @staticmethod
    def _points_world(frame_ids: list[int],
                      data: pin.Data,
                      z_offset: float = 0.0) -> np.ndarray:
        """
        Returns raw 3D points for the given frames.
        Shape: (N, 3)
        """
        pts = np.array([data.oMf[fid].translation for fid in frame_ids])
        pts[:, 2] -= z_offset
        return pts

    @staticmethod
    def _bbox_world(frame_ids: list[int],
                    data: pin.Data,
                    z_offset: float = 0.0) -> np.ndarray:
        """
        Build an AABB that encloses the given frames, **after** the legacy
        z-offset has been applied.

        Returns
        -------
        np.ndarray, shape (6,)
            [xmin, ymin, zmin, xmax, ymax, zmax] in the world frame.
        """
        pts = np.array([data.oMf[fid].translation for fid in frame_ids])
        pts[:, 2] -= z_offset                       # legacy shift
        bb_min = pts.min(axis=0)
        bb_max = pts.max(axis=0)
        return np.hstack((bb_min, bb_max))
