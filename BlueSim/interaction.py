import math
import numpy as np
import random


class Interaction():
    """Underwater interactions

    This class models interactions of the fish with their environment, e.g.,
    to perceive other fish or to change their position.
    """

    def __init__(self, environment, verbose=False):
        """Constructor

        Initializes the channel

        Arguments:
            nodes {list} -- List of fish instances
        """

        self.environment = environment
        self.verbose = verbose

    def perceive_object(self, source_id, pos):
        """Perceive the relative position to an object

        This simulates the fish's perception of external sources and targets.

        Arguments:
            source_id {int} -- Index of the fish that wants to know its
                location
            pos {np.array} -- X, Y, and Z position of the object
        """

        return pos - self.environment.node_pos[source_id]

    def perceive_pos(self, source_id, target_id):
        """Perceive the relative position to another fish

        This simulates the fish's perception of neighbors.

        Arguments:
            source_id {int} -- Index of the fish to be perceived
            target_id {int} -- Index of the fish to be perceived
        """

        if source_id == target_id:
            # You are very close to yourself!
            return np.zeros((3,))

        prob = self.environment.prob(source_id, target_id)

        success = random.random() <= prob

        if self.verbose:
            print('Interaction: {} perceived {}: {} (prob: {:0.2f})'.format(
                source_id, target_id, success, prob
            ))

        if success:
            return self.environment.get_rel_pos(source_id, target_id)
        else:
            return np.zeros((3,))

    def perceive_depth(self, source_id):
        """Perceive own depth. Simulation of pressure sensor.
        
        Args:
            source_id (int): ID of fish requesting depth.
        
        Returns:
            int: Absolute depth in global coordinate frame
        """
        pos = self.environment.node_pos[source_id]

        return pos[2]

    def rot_global_to_robot(self, source_id):
        """Rotate global coordinates to robot coordinates. Used before simulation of dynamics.
        
        Args:
            source_id (id): Fish ID
        
        Returns:
            np.array: 3x3 rotation matrix based on current orientation
        """
        phi = self.environment.node_phi[source_id]

        return np.array([[math.cos(phi), math.sin(phi), 0], [-math.sin(phi), math.cos(phi), 0], [0, 0, 1]])

    def blind_spot(self, source_id, neighbors, rel_pos):
        """Omits neighbors within the blind spot behind own body.
        
        Args:
            source_id (int): Fish ID
            neighbors (set): Set of visible neighbors
            rel_pos (dict): Relative positions of neighbors
        """
        r_blockage = 25 # 50mm blocking corridor behind itself
        vel = self.environment.node_vel[source_id]

        candidates = neighbors.copy()
        for neighbor in candidates:
            dot = np.dot(vel[:2], rel_pos[neighbor][:2])
            if dot < 0:
                mag_vel = max(0.001, np.linalg.norm(vel[:2]))
                dist_neighbor = max(0.001, np.linalg.norm(rel_pos[neighbor][:2]))

                angle = abs(math.acos(dot / (mag_vel * dist_neighbor))) - math.pi / 2 # cos(a-b) = ca*cb+sa*sb = sa

                if  math.cos(angle) * dist_neighbor < r_blockage:
                    neighbors.remove(neighbor)

    def occlude(self, source_id, neighbors, rel_pos):
        """Omits invisible neighbors occluded by others.
        
        Args:
            ource_id (int): Fish ID
            neighbors (set): Set of visible neighbors
            rel_pos (dict): Relative positions of neighbors
        """
        if not neighbors:
            return

        def sortSecond(val):
            return val[1]

        r_sphere = 50 # 50mm blocking sphere imposed by neighbors

        n_by_dist = []
        for key, value in rel_pos.items():
            if key in neighbors:
                n_by_dist.append((key, np.linalg.norm(value)))
        n_by_dist.sort(key = sortSecond)

        n_valid = [n_by_dist[0]]
        for neighbor in n_by_dist[1:]:
            occluded = False
            d_neighbor = max(0.001, neighbor[1])
            coord_neighbor = rel_pos[neighbor[0]]

            for verified in n_valid:
                d_verified = max(0.001, verified[1])
                coord_verified = rel_pos[verified[0]]

                theta_min = math.atan(r_sphere / d_verified)

                theta = abs(math.acos(np.dot(coord_neighbor, coord_verified) / (d_neighbor * d_verified)))

                if theta < theta_min:
                    occluded = True
                    neighbors.remove(neighbor[0])
                    if not neighbors:
                        return
                    break

            if not occluded:
                n_valid.append(neighbor)

    def move(self, source_id, target_direction):
        """Move a fish

        Moves the fish relatively into the given direction and adds
        target-based distortion to the fish position.

        Arguments:
            source_id {int} -- Fish identifier
            target_direction {np.array} -- Relative direction to move to
        """
        node_pos = self.environment.node_pos[source_id]
        target_pos = node_pos + target_direction
        # Restrict to tank
        target_pos[0] = np.clip(target_pos[0], 0, self.environment.arena_size[0])
        target_pos[1] = np.clip(target_pos[1], 0, self.environment.arena_size[1])
        target_pos[2] = np.clip(target_pos[2], 0, self.environment.arena_size[2])

        final_pos = self.environment.get_distorted_pos(source_id, target_pos)
        final_pos[0] = np.clip(final_pos[0], 0, self.environment.arena_size[0])
        final_pos[1] = np.clip(final_pos[1], 0, self.environment.arena_size[1])
        final_pos[2] = np.clip(final_pos[2], 0, self.environment.arena_size[2])

        #self.environment.set_vel(source_id, node_pos, final_pos) #xx
        #print(final_pos)
        self.environment.set_pos(source_id, final_pos)

        if self.verbose:
            print('Interaction: {} moved to {}'.format(
                source_id, final_pos
            ))
