from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class Push(Task):
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_xy_range=0.3,
        obj_xy_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.obj_xy_range = obj_xy_range
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        self.obj_range_high_min = np.array([obj_xy_range / 2 - 0.05, obj_xy_range / 2 - 0.05, 0])
        self.obj_range_low_min = np.array([-obj_xy_range / 2 + 0.05, -obj_xy_range / 2 + 0.05, 0])
        self.last_dist_obj = None
        self.last_d = None
        self.last_dist_obj_norm = None
        self.last_d_norm = None 

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=self.table_length, width=self.table_width, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = np.array(self.sim.get_base_position("object"))
        object_rotation = np.array(self.sim.get_base_rotation("object"))
        object_velocity = np.array(self.sim.get_base_velocity("object"))
        object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object"))
        observation = np.concatenate(
            [
                object_position,
                object_rotation,
                object_velocity,
                object_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_object(self) -> np.ndarray:
        """Randomize goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        return goal

    def _sample_goal(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        if self.np_random.random() < 0.5:
            # Forward End
            noise = [self.np_random.uniform(self.obj_xy_range / 2 - 0.05, self.obj_xy_range/2),\
                         self.np_random.uniform(-self.obj_xy_range/2, self.obj_xy_range/2), 0]
        else:
            # Sides
            noise = [self.np_random.uniform(-self.obj_xy_range/2, self.obj_xy_range/2),\
                    self.np_random.choice([1, -1]) * self.np_random.uniform(self.obj_xy_range / 2 - 0.05,\
                    self.obj_xy_range/2), 0]

        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, observation, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(observation["achieved_goal"], desired_goal) / self.last_d_norm
        obj = self.get_achieved_goal()
        # print(observation["observation"])
        d_obj = distance(observation["observation"][:3], obj) / self.last_dist_obj_norm
        reward = self.last_dist_obj - d_obj
        self.last_dist_obj = d_obj


        reward +=  (self.last_d - d)
        self.last_d = d
        # if self.reward_type == "sparse":
        #     return reward_dist-np.array(d > self.distance_threshold, dtype=np.float32)
        # else:
        #     return reward_dist-d.astype(np.float32)

        if d < self.distance_threshold:
            reward += 10


        return reward
