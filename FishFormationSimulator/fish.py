import math
import numpy as np
from queue import Queue
import time
import datetime
import pandas as pd 
from events import HopCount, Ping, InfoInternal, LeaderElection
from eventcodes import (
    PING, HOMING, HOP_COUNT, INFO_EXTERNAL, INFO_INTERNAL, START_HOP_COUNT,
    START_LEADER_ELECTION, LEADER_ELECTION, MOVE
)
from math import sqrt
from math import ceil
class Fish():
    """This class models each fish robot node in the network from the fish'
    perspective.

    Each fish has an ID, communicates over the channel, and perceives its
    neighbors and takes actions accordingly. In taking actions, the fish can
    weight information from neighbors based on their distance. The fish aims to
    stay between a lower and upper limit of neighbors to maintain a cohesive
    collective. It can moves at a maximal speed and updates its behavior on
    every clock tick.
    """
   """ Added by Magaly: 
   direction: [0,1,2,3]
   N,E,S,W
   formation : Different formation options 
   0: Triangle Base of 3 
   1: Rectangular base 
   """
    def __init__(
        self,
        id,
        channel,
        direction,
        formation, 
        interaction,
        lim_neighbors=[0, math.inf],
        fish_max_speed=1,
        clock_freq=1,
        neighbor_weight=1.0,
        name='Unnamed',
        verbose=False
    ):
        """Create a new fish

        Arguments:
            id {int} -- UUID.
            channel {class} -- Communication channel.
            interaction {class} -- Interactions which include perception of
                neighbors and own movement.

        Keyword Arguments:
            lim_neighbors {int, int} -- Lower and upper limit of neighbors each
                fish aims to be connected to.
                (default: {0, math.inf})
            fish_max_speed {number} -- Max speed of each fish. Defines by how
                much it can change its position in one simulation step.
                (default: {1})
            clock_freq {number} -- Behavior update rate in Hertz (default: {1})
            neighbor_weight {number} -- A weight based on distance that defines
                how much each of a fish's neighbor affects its next move.
                (default: {1.0})
            name {str} -- Unique name of the fish. (default: {'Unnamed'})
            verbose {bool} -- If `true` log out some stuff (default: {False})
        """

        self.id = id
        self.channel = channel
        self.interaction = interaction
        self.neighbor_weight = neighbor_weight
        self.lim_neighbors = lim_neighbors
        self.fish_max_speed = fish_max_speed
        self.clock_freq = clock_freq
        self.name = name
        self.verbose = verbose

        self.clock_speed = 1 / self.clock_freq
        self.clock = 0
        self.queue = Queue()
        self.target_pos = np.zeros((2,))
        self.is_started = False
        self.neighbors = set()

        self.status = None

        self.info = None  # Some information
        self.info_clock = 0  # Time stamp of the information, i.e., the clock
        self.info_hops = 0  # Number of hops until the information arrived
        self.last_hop_count_clock = -math.inf
        self.hop_count = 0
        self.hop_distance = 0
        self.hop_count_initiator = False
        self.initial_hop_count_clock = 0

        self.leader_election_max_id = -1
        self.last_leader_election_clock = -1

        now = datetime.datetime.now()

        # Stores messages to be send out at the end of the clock cycle
        self.messages = []

        # Logger instance
        # with open('{}_{}.log'.format(self.name, self.id), 'w') as f:
        #     f.truncate()
        #     f.write('TIME  ::  #NEIGHBORS  ::  INFO  ::  ({})\n'.format(
        #         datetime.datetime.now())
        #     )

    def start(self):
        """Start the process

        This sets `is_started` to true and invokes `run()`.
        """
        self.is_started = True
        self.run()

    def stop(self):
        """Stop the process

        This sets `is_started` to false.
        """
        self.is_started = False

    def log(self, neighbors=set()):
        """Log current state
        """

        with open('{}_{}.log'.format(self.name, self.id), 'a+') as f:
            f.write(
                '{:05}    {:04}    {}    {}\n'.format(
                    self.clock,
                    len(neighbors),
                    self.info,
                    self.info_hops
                )
            )

    def run(self):
        """Run the process recursively

        This method simulates the fish and calls `eval` on every clock tick as
        long as the fish `is_started`.
        """

        while self.is_started:
            start_time = time.time()
            self.eval()
            time_elapsed = time.time() - start_time

            sleep_time = (self.clock_speed / 2) - time_elapsed

            # print(time_elapsed, sleep_time, self.clock_speed / 2)
            time.sleep(max(0, sleep_time))
            if sleep_time < 0 and self.verbose:
                print('Warning frequency too high or computer too slow')

            start_time = time.time()
            self.communicate()
            time_elapsed = time.time() - start_time

            sleep_time = (self.clock_speed / 2) - time_elapsed
            time.sleep(max(0, sleep_time))
            if sleep_time < 0 and self.verbose:
                print('Warning frequency too high or computer too slow')

    def move_handler(self, event):
        """Handle move events, i.e., update the target position.

        Arguments:
            event {Move} -- Event holding an x and y target position
        """
        self.target_pos[0] = event.x
        self.target_pos[1] = event.y

    def ping_handler(self, neighbors, rel_pos, event):
        """Handle ping events

        Adds the

        Arguments:
            neighbors {set} -- Set of active neighbors, i.e., nodes from which
                this fish received a ping event.
            rel_pos {dict} -- Dictionary of relative positions from this fish
                to the source of the ping event.
            event {Ping} -- The ping event instance
        """
        neighbors.add(event.source_id)

        # When the other fish is not perceived its relative position is [0,0]
        rel_pos[event.source_id] = self.interaction.perceive_pos(
            self.id, event.source_id
        )

        if self.verbose:
            print('Fish #{}: saw friend #{} at {}'.format(
                self.id, event.source_id, rel_pos[event.source_id]
            ))

    def homing_handler(self, event, pos):
        """Homing handler, i.e., make fish aggregated extremely

        Arguments:
            event {Homing} -- Homing event
            pos {np.array} -- Position of the homing event initialtor
        """
        self.info = 'signal_aircraft'  # Very bad practice. Needs to be fixed!
        self.info_clock = self.clock

        self.messages.append(
            (self, InfoInternal(self.id, self.clock, self.info))
        )

        # update behavior based on external event
        self.status = 'wait'
        self.target_pos = self.interaction.perceive_object(self.id, pos)

        if self.verbose:
            print('Fish #{} got external info {}'.format(
                self.id, event.message
            ))

    def info_ext_handler(self, event):
        """External information handler

        Always accept the external information and spread the news.

        Arguments:
            event {InfoExternal} -- InfoExternal event
        """
        self.info = event.message
        self.info_clock = self.clock

        self.messages.append(
            (self, InfoInternal(self.id, self.clock, self.info))
        )

        if self.verbose:
            print('Fish #{} got external info {}'.format(
                self.id, event.message
            ))

    def info_int_handler(self, event):
        """Internal information event handler.

        Only accept the information of the clock is higher than from the last
        information

        Arguments:
            event {InfoInternal} -- Internal information event instance
        """
        if self.info_clock >= event.clock:
            return

        self.info = event.message
        self.info_clock = event.clock
        self.info_hops = event.hops + 1

        self.messages.append((
            self,
            InfoInternal(self.id, self.info_clock, self.info, self.info_hops)
        ))

        if self.verbose:
            print('Fish #{} got info: {} from #{}'.format(
                self.id, event.message, event.source_id
            ))

    def hop_count_handler(self, event):
        """Hop count handler

        Initialize only of the last hop count event is 4 clocks old. Otherwise
        update the hop count and resend the new value only if its larger than
        the previous hop count value.

        Arguments:
            event {HopCount} -- Hop count event instance
        """
        # initialize
        if (self.clock - self.last_hop_count_clock) > 4:
            self.hop_count_initiator = False
            self.hop_distance = event.hops + 1
            self.hop_count = event.hops + 1
            self.messages.append((
                self,
                HopCount(self.id, self.info_clock, self.hop_count)
            ))

        else:
            # propagate value
            if self.hop_count < event.hops:
                self.hop_count = event.hops

                if not self.hop_count_initiator:
                    self.messages.append((
                        self,
                        HopCount(self.id, self.info_clock, self.hop_count)
                    ))

        self.last_hop_count_clock = self.clock

        if self.verbose:
            print('Fish #{} counts hops {} from #{}'.format(
                self.id, event.hop_count, event.source_id
            ))

    def start_hop_count_handler(self, event):
        """Hop count start handler

        Always accept a new start event for a hop count

        Arguments:
            event {StartHopCount} -- Hop count start event
        """
        self.last_hop_count_clock = self.clock
        self.hop_distance = 0
        self.hop_count = 0
        self.hop_count_initiator = True
        self.initial_hop_count_clock = self.clock

        self.messages.append((
            self,
            HopCount(self.id, self.info_clock, self.hop_count)
        ))

        if self.verbose:
            print('Fish #{} counts hops {} from #{}'.format(
                self.id, event.hop_count, event.source_id
            ))

    def leader_election_handler(self, event):
        """Leader election handler

        Arguments:
            event {LeaderElection} -- Leader election event instance
        """
        # This need to be adjusted in the future
        if (self.clock - self.last_leader_election_clock) < math.inf:
            new_max_id = max(event.max_id, self.id)
            # propagate value
            if self.leader_election_max_id < new_max_id:
                self.leader_election_max_id = new_max_id

                self.messages.append((
                    self,
                    LeaderElection(self.id, new_max_id)
                ))

        self.last_leader_election_clock = self.clock

    def weight_neighbor(self, rel_pos_to_neighbor):
        """Weight neighbors by the relative position to them

        Currently only returns a static value but this could be tweaked in the
        future to calculate a weighted center point.

        Arguments:
            rel_pos_to_neighbor {np.array} -- Relative position to a neighbor

        Returns:
            float -- Weight for this neighbor
        """
        return self.neighbor_weight

    def start_leader_election_handler(self, event):
        """Leader election start handler

        Always accept a new start event for a leader election

        Arguments:
            event {StartLeaderElection} -- Leader election start event
        """
        self.last_leader_election_clock = self.clock
        self.leader_election_max_id = self.id

        self.messages.append((
            self,
            LeaderElection(self.id, self.id)
        ))

    def comp_center(self, rel_pos):
        """Compute the (potentially weighted) centroid of the fish neighbors

        Arguments:
            rel_pos {dict} -- Dictionary of relative positions to the
                neighboring fish.

        Returns:
            np.array -- 2D centroid
        """
        center = np.zeros((2,))
        n = max(1, len(rel_pos))

        for key, value in rel_pos.items():
            weight = self.weight_neighbor(value)
            center += value * weight

        center /= n

        if self.verbose:
            print('Fish #{}: swarm centroid {}'.format(self.id, center))

        return center
    
    #This function returns all of the neighbors that are visible for the fish 
    #This function needs to be updated if wanting to include other physical obstacles
    #A fish is ocluded if it is between the lines from the current fish and the front of a neighbor fish 
    #and the current fish to the tail of the neighbor fish. 
    def visible_neighbors(self, rel_pos, r, fish_size):
        rel_pos = rel_pos.sort_values(['dist'])
        not_visible_fish = []
        for index, neighbor in rel_pos.iterrows(): 
            if(index not in not_visible_fish):
                if(neighbor.dist> r):
                    rel_pos = rel_pos.drop(index, axis=0)
                    not_visible_fish.append(index)
                else:
                    m1 = (neighbor.Y+fish_size)/(neighbor.X)
                    m2 = (neighbor.Y-fish_size)/(neighbor.X)
                    for index2, neighbor2 in rel_pos.iterrows():
                        if(neighbor2.Y<(neighbor2.Y*m1)and neighbor2.Y>(neighbor2.Y*m2) and neighbor2.X>neighbor.X):
                            rel_pos = rel_pos.drop(index2, axis=0)
                            not_visible_fish.append(index2)
                            
        return rel_pos
    #Fibonacci Function
    def F(self,i):
        return ((1+sqrt(5))**i-(1-sqrt(5))**i)/(2**i*sqrt(5))

    #Returns parameters for fibonacci formation
    def get_row_n(self,front_neighbors):
        my_index = front_neighbors+1
        f_i =0
        i =0
        while(f_i<my_index):
            f_i = self.F(i)
            i+=1
        return my_index-1, i-3,np.round(self.F(i-2)), np.round(self.F(i-1))
    #returns dataframe with the target locations 
    def fibonacci_formation(self,nfish,linespace):
        columns = ['X', 'Y', 'dist']
        index= np.arange(nfish)
        df= pd.DataFrame(index=index, columns=columns)
        df= df.fillna(0.0)
        #print(df)
        for i,row in df.iterrows():
            my_index, my_row, rs, re = self.get_row_n(i)
            if(my_row<=0):
                my_index =0
                my_row =0
            if(my_index in [0,1,2]):
                df.iloc[i]['X'] = my_row*linespace
                df.iloc[i]['dist'] = my_index
                df.iloc[i]['Y'] = 0

            else:
                df.iloc[i]['X'] = my_row*linespace
                df.iloc[i]['dist'] = my_index
                if(re-rs ==2):
                    if(my_index==3):
                        df.iloc[i]['Y'] =-.5
                    else:
                        df.iloc[i]['Y'] =.5
                elif((re-rs)%2==0):
                    ys = np.linspace(-(re-rs)/2,int(re-rs)/2,int(re-rs))
                    df.iloc[i]['Y'] = ys[int(my_index -rs)]
                else:
                    df.iloc[i]['Y'] = ceil((re-rs)/2) - (re-my_index)
        df['X'] = -df['X']
        return df
   #Triangle_function
    def T(self,i):
        return i*(i+1)/2
    def get_row_n2(self,front_neighbors):
        if(front_neighbors==0):
            return 0,0,0,0
        else:
            my_index=front_neighbors 
            f_i =0
            i =1
            while(f_i<=my_index):
                f_i = self.T(i)
                i+=1        
            return my_index, i-2, self.T(i-1)-i+1,self.T(i)-(i+1)
    #Triangle formation map 
    def triangle_formation(self,nfish,linespace):
        columns = ['X', 'Y', 'dist']
        index= np.arange(nfish)
        df= pd.DataFrame(index=index, columns=columns)
        df= df.fillna(0.0)
        for i,row in df.iterrows():
            my_index, my_row, rs, re = self.get_row_n2(i)
            df.iloc[i]['X'] = my_row*linespace
            df.iloc[i]['dist'] = my_index
            if(re-rs>0):
                ys = np.linspace(-(re-rs)/2,(re-rs)/2,int(re-rs)+1)
                df.iloc[i]['Y'] = ys[int(my_index-rs)]
        df['X'] = -df['X']
        return df
    #returns the target formation
    def generate_map(self,n_fish,map_type,linespace):
        if(map_type==0):
            df = self.fibonacci_formation(n_fish,linespace)
                
        elif(map_type==1):
            df = self.triangle_formation(n_fish,linespace)
        return df
    def nearest(target_map, neighbor_pos):
        target_map['dist'] = np.sqrt( (target_map.X-neighbor_pos['X'])**2 + (target_map.Y-neighbor_pos['Y'])**2)
        target_map.sort_values(['dist'],ascending =True)
        return target_map.iloc[0]
    #function returns the target point for my fish 
#     def define_target(self,rel_pos,target_map):
#         #Define Leader
#         df = pd.DataFrame.from_dict(rel_pos,orient = 'index', columns=['X', 'Y'])
#         df['dist'] = np.sqrt( (df.X)**2 + (df.Y)**2)
#         rel_pos = df.sort_values(['X'],ascending =False)
#         my_target = [0,0]
#         #meaning I am the leader 
#         if(len(rel_pos)==0):
#             return [0,0]
#         elif(len(rel_pos)>0):
#             leader = rel_pos.iloc[0]
#             my_leader_x =leader['X']
#             my_leader_y = leader['Y']
#             #Specify my initial target 
#             target_map['X'] += my_leader_x
#             target_map['Y'] += my_leader_y
#             target_map['dist'] = np.sqrt( (target_map.X)**2 + (target_map.Y)**2)
#             target_map = target_map.sort_values(['dist'],ascending =True)
#             #my_target = [target_map.iloc[2]['X'],target_map.iloc[2]['Y']]

#            for(a in A)
#     b := nearest(B, a)
#     if a = nearest(A, b)
#         add (a, b) to results
#     end if
# end for



#             while(len(target_map)>0):
#                 my_target = [target_map.iloc[0]['X'],target_map.iloc[0]['Y']]
#                 my_dist = np.sqrt((my_target[0])**2 + (my_target[1])**2)
#                 for idx,target_point in target_map.iterrows():
#                     rel_pos['dist'] = np.sqrt( (rel_pos.X-target_point['X'])**2 + (rel_pos.Y-target_point['Y'])**2)
#                     rel_pos = df.sort_values(['dist'],ascending =True)
#                     if(my_dist<=rel_pos.iloc[0]['dist']):
#                         return my_targetxs
#                     else: 
#                         target_map.drop(target_map.index[0]) 
#                         rel_pos.drop(rel_pos.index[0]) 
                    
#         return my_target

    def align_school(self,rel_pos, linespace, r, fish_size):
        tolerance = 0.025
        df = pd.DataFrame.from_dict(rel_pos,orient = 'index', columns=['X', 'Y'])
        df['dist'] = np.sqrt( (df.X)**2 + (df.Y)**2)
        df['angles']= np.degrees(np.arcsin(df.Y/df.dist))
        #rel_pos = self.visible_neighbors(rel_pos, r, fish_size)
        rel_pos = df.sort_values(['X'],ascending =False)
        leader = rel_pos.iloc[0]
        avx =0
        avy =0
        #This means that I have the largest X value therefore I am the leader 
        if(leader['X']<=0):
            return [0,0]
        else: 
            front_neighbors = rel_pos[rel_pos['X'] >= 0]
            my_index, my_row, rs, re = self.get_row_n(len(front_neighbors))
            avx = leader['X']- my_row*linespace
            rs = int(np.round(rs))
            re = int(np.round(re))
            #Set target X
            avy = leader['Y'] - (my_index-re)
            if(my_index==rs):
                xline = rel_pos.iloc[rs+1:re]
                #assert(len(xline)== (re-rs)-1)
            else:
                xline =  rel_pos.iloc[rs:re-1]
                #assert(len(xline)== (re-rs)-1)
            row_size = re-rs
            y_idx = abs(my_index-rs)
            xline = xline.sort_values(['Y'],ascending =False)
            
            for _,l in xline.iterrows():
                if(avy-tolerance<=l['Y']<= avy-tolerance):
                    avy =0 
            if(row_size%2==0):
                if(y_idx<row_size/2):
                    avy = leader['Y']+(row_size/2-y_idx)
                else: 
                    avy = leader['Y'] - y_idx

            else:
                if(y_idx<row_size/2):
                    avy = leader['Y']+(row_size/2-y_idx)
                elif(y_idx>row_size/2): 
                    avy = leader['Y']-(y_idx)
                else: 
                    avy = leader['Y']
            if(avx==0):
                avy =0 
            #get the ones that have positive Y to get my row index 
            return[avx, avy]


    def move(self, neighbors, rel_pos):
        """Make a cohesion and target-driven move

        The move is determined by the relative position of the centroid and a
        target position and is limited by the maximum fish speed.

        Arguments:
            neighbors {set} -- Set of active neighbors, i.e., other fish that
                responded to the most recent ping event.
            rel_pos {dict} -- Relative positions to all neighbors

        Returns:
            np.array -- Move direction as a 2D vector
        # """
        # n = len(neighbors)
        # # Get the centroid of the swarm
        # centroid_pos = np.zeros((2,))

        # if n < self.lim_neighbors[0]:
        #     # Get the relative direction to the centroid of the swarm
        #     centroid_pos = self.comp_center(rel_pos)
        # elif n > self.lim_neighbors[1]:
        #     # Get the inverse relative direction to centroid of the swarm
        #     centroid_pos = -self.comp_center(rel_pos)
        #     # Adjust length
        #     magnitude = np.linalg.norm(centroid_pos)
        #     centroid_pos /= magnitude**2
        # print('target :',self.target_pos)
        # move = self.target_pos + centroid_pos
        
        # Cap the length of the move
        linespace = 1
        target_map = self.generate_map(len(neighbors),0,linespace)
        
        r =20
        fish_size =0.001
        move = self.align_school(rel_pos, linespace, r, fish_size)
        magnitude = 1
        # if magnitude == 0:
        #     magnitude = 1
        move = self.target_pos +  move/np.linalg.norm(move)
        #move = self.define_target(rel_pos,target_map)
        
        #magnitude = np.sqrt(move[0]**2 + move[1]**2)
        final_move = move* min(magnitude, self.fish_max_speed)

        #direction = move / magnitude
        #final_move = direction * min(magnitude, self.fish_max_speed)
        #if self.verbose:
         #   print('Fish #{}: move to {}'.format(self.id, final_move))
        #final_move = self.move_formation(neighbors,rel_pos)
        return final_move

   

    def update_behavior(self):
        """Update the fish behavior.

        This actively changes the cohesion strategy to either 'wait', i.e, do
        not care about any neighbors or 'signal_aircraft', i.e., aggregate with
        as many fish friends as possible.

        In robotics 'signal_aircraft' is a secret key word for robo-fish-nerds
        to gather in a secret lab until some robo fish finds a robo aircraft.
        """
        if self.status == 'wait':
            self.lim_neighbors = [0, math.inf]
        elif self.info == 'signal_aircraft':
            self.lim_neighbors = [math.inf, math.inf]

    def eval(self):
        """The fish evaluates its state

        Currently the fish checks all responses to previous pings and evaluates
        its relative position to all neighbors. Neighbors are other fish that
        received the ping element.
        """

        # Set of neighbors at this point. Will be reconstructed every time
        neighbors = set()
        rel_pos = {}

        self.saw_hop_count = False

        while not self.queue.empty():
            (event, pos) = self.queue.get()

            if event.opcode == PING:
                self.ping_handler(neighbors, rel_pos, event)

            if event.opcode == HOMING:
                self.homing_handler(event, pos)

            if event.opcode == START_HOP_COUNT:
                self.start_hop_count_handler(event)

            if event.opcode == HOP_COUNT:
                self.hop_count_handler(event)

            if event.opcode == INFO_EXTERNAL:
                self.info_ext_handler(event)

            if event.opcode == INFO_INTERNAL:
                self.info_int_handler(event)

            if event.opcode == START_LEADER_ELECTION:
                self.start_leader_election_handler(event)

            if event.opcode == LEADER_ELECTION:
                self.leader_election_handler(event)

            if event.opcode == MOVE:
                self.move_handler(event)

        if self.clock > 1:
            # Move around (or just stay where you are)
            self.interaction.move(self.id, self.move(neighbors, rel_pos))

        # Update behavior based on status and information - update behavior
        self.update_behavior()

        self.neighbors = neighbors

        # self.log(neighbors)
        self.clock += 1

    def communicate(self):
        """Broadcast all collected event messages.

        This method is called as part of the second clock cycle.
        """
        for message in self.messages:
            self.channel.transmit(*message)

        self.messages = []

        # Always send out a ping to other fish
        self.channel.transmit(self, Ping(self.id))
