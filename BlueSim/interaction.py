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

    def occlude(self, source_id, neighbors, rel_pos):
        def sortSecond(val):
            return val[1]

        r_sphere = 50 # 50mm blocking sphere imposed by neighbors
        r_blockage = 25 # 50mm blocking corridor behind itself
        vel = self.environment.node_vel[source_id]

        n_by_dist = []
        for key, value in rel_pos.items():
            n_by_dist.append((key, np.linalg.norm(value)))
        n_by_dist.sort(key = sortSecond)

        n_valid = [n_by_dist[0]]
        for candidate in n_by_dist[1:]:
            occluded = False
            d_candidate = max(0.001, candidate[1])
            coord_candidate = rel_pos[candidate[0]]

            # blind spot
            mag_vel = max(0.001, np.linalg.norm(vel[:2]))
            mag_pos_cand = max(0.001, np.linalg.norm(coord_candidate[:2]))
            print(mag_vel, '    ', mag_pos_cand)


            angle = abs(math.acos(np.dot(vel[:2], coord_candidate[:2])
                / (mag_vel * mag_pos_cand))) - math.pi / 2

            if  angle * mag_pos_cand < r_blockage:
                occluded = True
                neighbors.remove(candidate[0])
                break

            # occlusion
            for verified in n_valid:
                d_verified = max(0.001, verified[1])
                coord_verified = rel_pos[verified[0]]

                theta_min = math.atan(r_sphere / d_verified)

                theta = abs(math.acos(np.dot(coord_candidate, coord_verified)
                    / (d_candidate * d_verified)))

                if theta < theta_min:
                    occluded = True
                    neighbors.remove(candidate[0])
                    break

            if not occluded:
                n_valid.append(candidate)

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

        self.environment.set_vel(source_id, node_pos, final_pos)
        self.environment.set_pos(source_id, final_pos)

        if self.verbose:
            print('Interaction: {} moved to {}'.format(
                source_id, final_pos
            ))
