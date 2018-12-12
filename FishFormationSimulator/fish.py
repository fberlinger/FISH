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
        orientation,
        formation_num, 
        tolerance, 
        interaction,
        fish_total_error,
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
        self.orientation = orientation
        self.formation_num = formation_num
        self.tolerance = tolerance
        self.interaction = interaction
        self.fish_total_error = fish_total_error
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

    # Start of Magaly's code 
    #Lists of target formations 
    def not_ocluded(self, rel_pos, r, fish_size):
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
                            
        return rel_pos, not_visible_fish
    '''
    0) Trigonal planar: Three neighbors f_i is the center of the neighbors each neighbor is 120 degrees from each other.
    1) Triangle: Two neighbors. 60 degrees between my two neighbors. 
    2) Global Fibonacci:  
    3) Global Triangle: 
    '''
    #Global algorithms 
    #Fibonacci function
    def F(self,i):
        return ((1+sqrt(5))**i-(1-sqrt(5))**i)/(2**i*sqrt(5))
    #Returns parameters for fibonacci formation
    def get_row_n(self,front_neighbors):
        linespace = 2
        my_index = front_neighbors+linespace
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

    #############################################################################################
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
    def mapped_formation_move(self,rel_pos,n_fish,map_type,linespace):
        if(map_type==0):
            df = self.fibonacci_formation(n_fish,linespace)     
        elif(map_type==1):
            df = self.triangle_formation(n_fish,linespace)
        return self.define_target(rel_pos, df)
    def nearest(self,target_map, neighbor_pos):
        target_map['dist'] = np.sqrt( (target_map.X-neighbor_pos['X'])**2 + (target_map.Y-neighbor_pos['Y'])**2)
        target_map.sort_values(['dist'],ascending =True)
        return target_map.iloc[0]
    #function returns the target point for my fish 
    def define_target(self,rel_pos,target_map):
        #Define Leader
        r = 1.5
        tolerance = self.tolerance
        df = pd.DataFrame.from_dict(rel_pos,orient = 'index', columns=['X', 'Y'])
        df['dist'] = np.sqrt( (df.X)**2 + (df.Y)**2)
        front_neighbors = df[df['X']>0]
        front_neighbors = front_neighbors.sort_values(['X'],ascending =False)
        rel_pos = df.sort_values(['dist'],ascending =True)
        my_target = [0,0]
        #meaning I am the leader 
        if(self.fish_total_error >= self.tolerance):
            if(len(front_neighbors)==0):
                if(len(df[df['dist']<=r])>0):
                    self.fish_total_error = 2*self.tolerance
                    return [0,0]
                return [0,0]
            elif(len(front_neighbors)>0):
                leader = front_neighbors.iloc[0]
                my_leader_x =leader['X']
                my_leader_y = leader['Y']
                target_map = target_map.sort_values(['X'],ascending =False)
                #No one should try to get the leader for now
                target_map.iloc[1:]
                #Transpose target map
                target_map['X'] += my_leader_x
                target_map['Y'] += my_leader_y
                target_map['dist'] = np.sqrt( (target_map.X)**2 + (target_map.Y)**2)
                target_map = target_map.sort_values(['dist'],ascending =True)


                if(len(target_map[target_map['dist']<0.5])>0 and len(rel_pos[rel_pos['dist']<=r])==0):
                    self.fish_total_error = target_map.iloc[0]['dist']
                    return [0,0]

                me =  pd.DataFrame([[0, 0, 0]])
                rel_pos.append(me)
                my_target = [target_map.iloc[0]['X'],target_map.iloc[0]['Y']]
                target_map = target_map.sort_values(['dist'],ascending =True)
                #In this case I assign positions to map targets so that I avoid collision
                for i, a in target_map.iterrows():
                    b = self.nearest(rel_pos, a)
                    if(b.X==0 and b.Y == 0):
                        if(len(rel_pos[rel_pos['dist']<=r])>0):
                            self.fish_total_error = a.dist + self.tolerance
                        return [a.X,a.Y]
                    c = self.nearest(target_map, b)
                    if(a.X == c.X and a.Y == c.Y):
                        rel_pos.drop(rel_pos.index[i]) 
                        target_map.drop(target_map.index[i])
                    elif(a.dist<= b.dist):
                        self.fish_total_error = abs(a.dist)
                        if(self.fish_total_error >= self.tolerance):
                            self.fish_total_error = abs(a.dist)
                            return [a.X, a.Y]
                        else: 
                            return [0,0]
            else: 
                self.fish_total_error = 0 
                return [0,0]
        return my_target
   
    #LOCAL FORMATIONS: 
    def visible_neighbors (self, my_neighbors):
        """Returns a dataframe of neighbors that are visible to the fish depending on target formation.
        Arguments:
            rel_pos {dict} -- relative position of all neighbors 
            orientation {} -- Orientation of the fish to detect visisble neighbors 
        Returns:
            pandas.dataframe -- a dataframe of the neighbors of the fish that are needed for a certain formation
        """
        # This part of the code  uses the orientation of the fish to detect which neighbors are seen in front. 
        if (self.formation_num == 0):
            my_neighbors['front'] = 0
            orientation= self.orientation 
            orientation= [orientation[1], -orientation[0]] 
            for i,row in my_neighbors.iterrows():
                front_n= ((row.X - orientation[0])*(orientation[0]) - (row.Y)*(orientation[1]))
                #front_n= np.cross([row.X-orientation[0],row.Y-orientation[1]], np.negative(orientation))
                my_neighbors.at[i, 'front'] = front_n
            neighbors_front =  my_neighbors[my_neighbors['front']>=0]
            neighbors_front = neighbors_front.sort_values(['dist','X'], ascending =True )
            if(len(neighbors_front)>=3):
                return neighbors_front.iloc[:3]
            else: 
                return neighbors_front 
        elif (self.formation_num == 1):
            orientation= self.orientation 
            orientation= [orientation[1], -orientation[0]] 
            my_neighbors['front'] = 0
            for i,row in my_neighbors.iterrows():
                front_n= ((row.X - orientation[0])*(orientation[0]) - (row.Y)*(orientation[1]))
                #front_n= np.cross([row.X-orientation[0],row.Y-orientation[1]], np.negative(orientation))
                my_neighbors.at[i, 'front'] = front_n
            neighbors_front =  my_neighbors[my_neighbors['front']>=0]
            neighbors_front = neighbors_front.sort_values(['dist','X'], ascending =True )
            if(len(neighbors_front)>=2):
                neighbors_front = neighbors_front.drop_duplicates()
                return neighbors_front.iloc[:2]
            else: 
                return neighbors_front
            
        return neighbors_front 
   
    def one_reference(self, n1,n2):
        if(np.array_equal(n1,n2)):
            return True
        if(np.count_nonzero(np.round(n2,1))==0):
            return True
        if(np.count_nonzero(np.round(n1,1))==0):
            return True
        return False

    def num_references(self, n1, n2, n3):
        num_references = 3
        if(np.count_nonzero(np.round(n2,1))==0):
            num_references -= 1
        if(np.count_nonzero(np.round(n1,1))==0):
            num_references -= 1
        if(np.count_nonzero(np.round(n3,1))==0):
            num_references -= 1
        if(np.array_equal(n1,n2)):
            num_references -= 1
        if(np.array_equal(n3,n2)):
            num_references -= 1
        if(np.array_equal(n1,n3)):
            num_references -= 1
        return num_references

    def possible_moves(self,linespace, formation_num, *neighbors):
        '''
        Returns the possible positions this fish has in order to arrive to a neighbor. 
        '''
        if(formation_num == 1):
            #either n1 or n2 equals zero 
            n1 = neighbors[0]
            n2 = neighbors[1]
            if(self.one_reference(n1,n2)):
                n = [0,0] 
                x_target = linespace*np.sin(np.deg2rad(60))
                y_target = linespace*np.cos(np.deg2rad(60))
                if(np.count_nonzero(np.round(n2,1))==0):
                    if(np.count_nonzero(np.round(n1,1))==0):
                        n=[0,0]
                    else:
                        n= n1
                elif(np.count_nonzero(np.round(n1,1))==0):
                    if(np.count_nonzero(np.round(n2,1))==0):
                        n=[0,0]
                    else:
                        n= n2
                elif(np.array_equal(n1,n2)):
                    n = n1
                m1 = [n[0] + x_target, n[1]+y_target]
                m2 = [n[0], n[1]+y_target]
                m3 = [n[0] + x_target, n[1]]
                m4 = [n[0] - x_target, n[1]+y_target]
                m5 = [n[0] + x_target, n[1]-y_target]
                m6 = [n[0] - x_target, n[1]-y_target]
                return[m1,m2,m3,m4,m5,m6]
            else:
                m = np.nan_to_num([(n1[0]+n2[0])/2,(n1[1]+n2[1])/2])
                ri = (n2[1]-n1[1])
                ru = (n2[0]-n1[0])
                if(ru==0):
                    slope = 0
                else: 
                    slope = -1/(ri/ru)
                b = m[1]- slope*m[0]
                p = [m[0]-1,(m[0]-1)*slope+b]
                v= np.subtract(m,p)
                m1 =np.add(m,v/np.linalg.norm(v)*linespace * np.sqrt(3))
                m2 = np.subtract(m,(v/np.linalg.norm(v)*linespace*np.sqrt(3)))
            return [m1 , m2]
        elif(formation_num==0):
            n1 = neighbors[0]
            n2 = neighbors[1]
            n3 = neighbors[2]
            x_target = linespace*np.sin(np.deg2rad(120))
            y_target = linespace*np.cos(np.deg2rad(120))
            if(self.num_references(n1,n2, n3)<=1):    
                if(np.count_nonzero(np.round(n1,1))==0):
                    if(np.count_nonzero(np.round(n2,1))==0):
                        if(np.count_nonzero(np.round(n3,1))):
                            n=[0,0]
                        else: 
                            n= n3
                    else:
                        n= n2
                else: 
                    n= n1
                
                if(np.array_equal(n1,n2) or np.array_equal(n1,n3)):
                    n = n1
                elif(np.array_equal(n2,n3)):
                    n = n2
                if(n[0]<0):
                    m1 = [n[0]+x_target, n[1]+ y_target]
                    m2 = [n[0]+ linespace, n[1]]
                    return [m1,m2]
                else: 
                    m1 = [n[0]-x_target, n[1]- y_target]
                    m2 = [n[0]+ linespace , n[1]]
                    return [m1,m2]
            elif(self.num_references(n1,n2, n3)==2):
                #Now discard the reference that is equal to zero 
                if(np.count_nonzero(np.round(n1,1))==0):
                    n1 = n2
                    n2 = n3
                    m = np.nan_to_num([(n1[0]+n2[0])/2,(n1[1]+n2[1])/2])
                    ri = (n2[1]-n1[1])
                    ru = (n2[0]-n1[0])
                    if(ru==0):
                        slope = 0
                    else: 
                        if(ri==0):
                            slope =0 
                        else:
                            slope = -1/(ri/ru)
                    b = m[1]- slope*m[0]
                    p = [m[0]-1,(m[0]-1)*slope+b]
                    v= np.subtract(m,p)
                    m1 =np.add(m,v/np.linalg.norm(v)*linespace * np.sqrt(3)/2)
                    m2 = np.subtract(m,(v/np.linalg.norm(v)*linespace*np.sqrt(3))/2)
                    return [m1 , m2]
                else: 
                    m = np.nan_to_num([(n1[0]+n2[0])/2,(n1[1]+n2[1])/2])
                    ri = (n2[1]-n1[1])
                    ru = (n2[0]-n1[0])
                    if(ru==0):
                        slope = 0
                    else: 
                        if(ri==0):
                            slope =0 
                        else:
                            slope = -1/(ri/ru) 
                    b= m[1]- slope*m[0]
                    p = [m[0]-1,(m[0]-1)*slope+b]
                    v= np.subtract(m,p)
                    m1 =np.add(m,v/np.linalg.norm(v)*linespace * np.sqrt(3)/2)
                    m2 = np.subtract(m,(v/np.linalg.norm(v)*linespace*np.sqrt(3))/2)
                    return [m1 , m2]
            else: 
                m = np.nan_to_num([(n1[0]+n2[0])/2,(n1[1]+n2[1])/2])
                ri = (n2[1]-n1[1])
                ru = (n2[0]-n1[0])
                if(ru==0):
                    slope = 0
                else:
                    if(ri==0):
                        slope =0 
                    else:
                        slope = -1/(ri/ru)
                b = m[1]- slope*m[0]
                p = [m[0]-1,(m[0]-1)*slope+b]
                v= np.subtract(m,p)
                m1 =np.add(m,v/np.linalg.norm(v)*linespace * np.sqrt(3)/2)
                m2 = np.subtract(m,(v/np.linalg.norm(v)*linespace*np.sqrt(3)/2))
                return [m1 , m2]
        return [[0,0]]

    #This function returns the fitness level of the fish f_i given the target formation
    def fitnes(self, my_neighbors,linespace, radius_neighbors):
        total_square_error = 0 
        local_tolerance = 0.25
        if (self.formation_num == 0):
            x_target = linespace*np.sin(np.deg2rad(120))
            y_target = linespace*np.cos(np.deg2rad(120))
            if(len(my_neighbors)==0):
                #If I have no reference neighbors (I am the first of the school) I still have to check whether neighbors are very close. 
                #If one or more neighbors are too close, then my fitness level will be increased so I move backwards and then can 
                #make another move. 
                for i,row in radius_neighbors.iterrows():
                    if((-local_tolerance <= round(row.X,1) <= local_tolerance) or  (-local_tolerance <= round(row.Y,1) <= local_tolerance)):
                        return 3*self.tolerance
                    else: 
                        return 0 
            elif(len(my_neighbors)==1):
                neighbor = [my_neighbors.iloc[0]['X'],my_neighbors.iloc[0]['Y']]
                moves = self.possible_moves(linespace, self.formation_num, neighbor, [0,0], [0,0])
                distances = []
                for move in moves: 
                    if(self.valid_move(radius_neighbors, move,local_tolerance)):
                        d = np.sqrt( (move[0])**2 + (move[1])**2)
                        distances.append(d)
                    else: 
                        distances.append(10)
                total_square_error += np.min(distances)
                for i,row in radius_neighbors.iterrows():
                    if((-local_tolerance <= round(row.X,1) <= local_tolerance) or  (-local_tolerance <= round(row.Y,1) <= local_tolerance)):
                        total_square_error += 3*self.tolerance
            elif(len(my_neighbors)==2):
                n1 = np.nan_to_num([my_neighbors.iloc[0]['X'],my_neighbors.iloc[0]['Y']])
                n2 = np.nan_to_num([my_neighbors.iloc[1]['X'],my_neighbors.iloc[1]['Y']])
                moves = self.possible_moves(linespace, self.formation_num, n1, n2, [0,0])
                distances = []
                for move in moves: 
                    if(self.valid_move(radius_neighbors, move, local_tolerance)):
                        d = np.sqrt( (move[0])**2 + (move[1])**2)
                        distances.append(d)
                    else: 
                        distances.append(10)
                total_square_error += np.min(distances)
                for i,row in radius_neighbors.iterrows():
                    if((-local_tolerance <= round(row.X,1) <= local_tolerance) or  (-local_tolerance <= round(row.Y,1) <= local_tolerance)):
                        total_square_error += self.tolerance
            elif(len(my_neighbors)>2):
                n1 = np.nan_to_num([my_neighbors.iloc[0]['X'],my_neighbors.iloc[0]['Y']])
                n2 = np.nan_to_num([my_neighbors.iloc[1]['X'],my_neighbors.iloc[1]['Y']])
                n3 = np.nan_to_num([my_neighbors.iloc[2]['X'],my_neighbors.iloc[2]['Y']])
                moves = self.possible_moves(linespace, self.formation_num, n1, n2, n3)
                distances = []
                for move in moves: 
                    if(self.valid_move(radius_neighbors, move, local_tolerance)):
                        d = np.sqrt( (move[0])**2 + (move[1])**2)
                        distances.append(d)
                    else: 
                        distances.append(10)
                total_square_error += np.min(distances)
                for i,row in radius_neighbors.iterrows():
                    if((-local_tolerance <= round(row.X,1) <= local_tolerance) or  (-local_tolerance <= round(row.Y,1) <= local_tolerance)):
                        total_square_error += self.tolerance
                    else: 
                        total_square_error += 0
        elif (self.formation_num == 1):
            x_target = linespace*np.sin(np.deg2rad(60))
            y_target = linespace*np.cos(np.deg2rad(60))
            if(len(my_neighbors)==0):
                #If I have no reference neighbors (I am the first of the school) I still have to check whether neighbors are very close. 
                #If one or more neighbors are too close, then my fitness level will be increased so I move backwards and then can 
                #make another move. 
                for i,row in radius_neighbors.iterrows():
                    if((-local_tolerance <= round(row.X,1) <= local_tolerance) or  (-local_tolerance <= round(row.Y,1) <= local_tolerance)):
                        return self.tolerance
                    else: 
                        return 0 
            elif(len(my_neighbors)==1):
                neighbor = [my_neighbors.iloc[0]['X'],my_neighbors.iloc[0]['Y']]
                moves = self.possible_moves(linespace, self.formation_num, neighbor, [0,0])
                distances = []
                for move in moves: 
                    if(self.valid_move(radius_neighbors, move,local_tolerance)):
                        d = np.sqrt( (move[0])**2 + (move[1])**2)
                        distances.append(d)
                    else: 
                        distances.append(10)
                total_square_error += np.min(distances)
                for i,row in radius_neighbors.iterrows():
                    if((-local_tolerance <= round(row.X,1) <= local_tolerance) or  (-local_tolerance <= round(row.Y,1) <= local_tolerance)):
                        total_square_error += 3*self.tolerance
                    else: 
                        total_square_error += 0
            elif(len(my_neighbors)>1):
                n1 = np.nan_to_num([my_neighbors.iloc[0]['X'],my_neighbors.iloc[0]['Y']])
                n2 = np.nan_to_num([my_neighbors.iloc[1]['X'],my_neighbors.iloc[1]['Y']])
                moves = self.possible_moves(linespace, self.formation_num, n1, n2)
                distances = []
                for move in moves: 
                    if(self.valid_move(radius_neighbors, move, local_tolerance)):
                        d = np.sqrt( (move[0])**2 + (move[1])**2)
                        distances.append(d)
                    else: 
                        distances.append(10)
                total_square_error += np.min(distances)
                for i,row in radius_neighbors.iterrows():
                    if((-local_tolerance <= round(row.X,1) <= local_tolerance) or  (-local_tolerance <= round(row.Y,1) <= local_tolerance)):
                        total_square_error += self.tolerance
        return total_square_error

    def valid_move(self, radius_neighbors, intended_move, local_tolerance):
        """Checks whether intended move of my fish is available, a.k.a. no other fish are already in that position
        Arguments:
            radius neighbors pandas df  -- df to all of the neighbors in the formation visible to the fish  
            intended_move [0,0] -- Array of intended location for this fish  
            local tolerance {float}  -- Ideal space between floats.
        Returns:
            True/False whether there is a spot available of nor 
        """
        intended_move= np.round(np.nan_to_num(intended_move),1)
        if(radius_neighbors.apply(lambda row: (((intended_move[0]-local_tolerance<=row.X <= intended_move[0]+ local_tolerance)
            &(( intended_move[1]- local_tolerance <=row.Y <= intended_move[1]+local_tolerance)))), axis=1).sum()>0):
            return False
        else: 
            if(np.count_nonzero(intended_move)>0):
                return True
            else:
                False
    

    def triangle_local(self,rel_pos,linespace,orientation):
        """Connect in a triangle formation with my nearest neighbors.
        Arguments:
            rel_pos {dict} -- dictionary to all of the neighbors in the formation
            linespace {float}  -- Ideal space between floats. 
            orientation {} -- Orientation of the fish to detect visisble neighbors 
        Returns:
            np.array -- Move direction as a 2D vector
        """
        #Target values 
        #Given the linespace (hypothemuse)  calculate the ideal x and y distances  
        #Trasnform into dataframe and filter through the ones that are only 3 linespaces away
        df = pd.DataFrame.from_dict(rel_pos,orient = 'index', columns=['X', 'Y'])
        df['dist'] = np.sqrt( (df.X)**2 + (df.Y)**2)
        radius_neighbors = df[df['dist']<= 10*linespace]
        radius_neighbors = radius_neighbors.sort_values(['dist'], ascending= True)
        
        local_tolerance = .25 
        x_target = linespace*np.sin(np.deg2rad(60))
        y_target = linespace*np.cos(np.deg2rad(60))

        #These are the neighbors I will take into consideration to choose my position
        my_neighbors = self.visible_neighbors(radius_neighbors)
        move = [0,0]
        #First we are going to check we indeed have access to the required number of neighbors
        self.fish_total_error = abs(self.fitnes(my_neighbors, linespace,radius_neighbors))
        if(self.fish_total_error <= self.tolerance):
             return [0,0]
        else:
            if(len(my_neighbors)==0):
                    return [-1,0]
            elif(len(my_neighbors)==1):
                    #drop your reference neighbor from the radious neighbors
                    radius_neighbors = radius_neighbors[1:]
                    n1 = [my_neighbors.iloc[0]['X'],my_neighbors.iloc[0]['Y']]
                    possible_moves = pd.DataFrame(self.possible_moves(linespace, self.formation_num,n1, [0,0]), columns=['X', 'Y'])
                    possible_moves['dist'] = np.sqrt( (possible_moves.X)**2 + (possible_moves.Y)**2)
                    possible_moves = possible_moves.sort_values(['dist'],ascending =True)
                    while(possible_moves.shape[0]>0 ):
                        move = possible_moves.iloc[0]
                        if(radius_neighbors.shape[0]>0):
                            b = self.nearest(radius_neighbors, move)
                            if(b.dist <= move.dist):
                                return [move['X'], move['Y']]
                            else: 
                                possible_moves.iloc[1:]
                                radius_neighbors.iloc[1:]
                        else: 
                            return [move['X'], move['Y']]
                    return [0,0]

                #1)get all of your possible moves given your reference neighbors
            else: 
                radius_neighbors = radius_neighbors[2:]
                n1 = [my_neighbors.iloc[0]['X'],my_neighbors.iloc[0]['Y']]
                n2 = [my_neighbors.iloc[1]['X'],my_neighbors.iloc[1]['Y']]
                possible_moves = pd.DataFrame(self.possible_moves(linespace, self.formation_num,n1, n1), columns=['X', 'Y'])
                possible_moves['dist'] = np.sqrt( (possible_moves.X)**2 + (possible_moves.Y)**2)
                possible_moves = possible_moves.sort_values(['dist'],ascending =True)
                while(possible_moves.shape[0]>0 ):
                    move = possible_moves.iloc[0]
                    if(radius_neighbors.shape[0]>0):
                        b = self.nearest(radius_neighbors, move)
                        if(b.dist <= move.dist):
                            return [move['X'], move['Y']]
                        else: 
                            possible_moves.iloc[1:]
                            radius_neighbors.iloc[1:]
                    else: 
                        return [move['X'], move['Y']]
                return [0,0]
            
    def trigonal_planar(self,rel_pos,linespace,orientation):
        df = pd.DataFrame.from_dict(rel_pos,orient = 'index', columns=['X', 'Y'])
        df['dist'] = np.sqrt( (df.X)**2 + (df.Y)**2)
        radius_neighbors = df[df['dist']<= 10*linespace]
        
        local_tolerance = .25 
        x_target = linespace*np.sin(np.deg2rad(120))
        y_target = linespace*np.cos(np.deg2rad(120))

        #These are the neighbors I will take into consideration to choose my position
        my_neighbors = self.visible_neighbors(radius_neighbors)
        self.fish_total_error = abs(self.fitnes(my_neighbors, linespace,radius_neighbors))
        if(self.fish_total_error<= self.tolerance):
            return [0,0]
        else: 
            if(len(my_neighbors)==0):
                    return [-1,0]
            elif(len(my_neighbors)==1):
                    #drop your reference neighbor from the radious neighbors
                    radius_neighbors = radius_neighbors[1:]
                    n1 = [my_neighbors.iloc[0]['X'],my_neighbors.iloc[0]['Y']]
                    possible_moves = pd.DataFrame(self.possible_moves(linespace, self.formation_num,n1, [0,0],[0,0]), columns=['X', 'Y'])
                    possible_moves['dist'] = np.sqrt( (possible_moves.X)**2 + (possible_moves.Y)**2)
                    possible_moves = possible_moves.sort_values(['dist'],ascending =True)
                    while(possible_moves.shape[0] >0 ):
                        move = possible_moves.iloc[0]
                        if(radius_neighbors.shape[0]>0):
                            b = self.nearest(radius_neighbors, move)
                            if(b.dist <= move.dist):
                                return [move['X'], move['Y']]
                            else: 
                                possible_moves.iloc[1:]
                                radius_neighbors.iloc[1:]
                        else: 
                            return [move['X'], move['Y']]
                    return [0,0]

            #1)get all of your possible moves given your reference neighbors
            elif(len(my_neighbors)==2): 
                radius_neighbors = radius_neighbors[2:]
                n1 = [my_neighbors.iloc[0]['X'],my_neighbors.iloc[0]['Y']]
                n2 = [my_neighbors.iloc[1]['X'],my_neighbors.iloc[1]['Y']]
                possible_moves = pd.DataFrame(self.possible_moves(linespace, self.formation_num,n1, n1, [0,0]), columns=['X', 'Y'])
                possible_moves['dist'] = np.sqrt( (possible_moves.X)**2 + (possible_moves.Y)**2)
                possible_moves = possible_moves.sort_values(['dist'],ascending =True)
                while(len(possible_moves) >0 ):
                    move = possible_moves.iloc[0]
                    if(radius_neighbors.shape[0]>0):
                        b = self.nearest(radius_neighbors, move)
                        if(b.dist <= move.dist):
                            return [move['X'], move['Y']]
                        else: 
                            possible_moves.iloc[1:]
                            radius_neighbors.iloc[1:]
                    else: 
                        return [move['X'], move['Y']]
                return [0,0]
            else: 
                radius_neighbors = radius_neighbors[2:]
                n1 = [my_neighbors.iloc[0]['X'],my_neighbors.iloc[0]['Y']]
                n2 = [my_neighbors.iloc[1]['X'],my_neighbors.iloc[1]['Y']]
                n3 = [my_neighbors.iloc[2]['X'],my_neighbors.iloc[2]['Y']]
                possible_moves = pd.DataFrame(self.possible_moves(linespace, self.formation_num,n1, n1,n3), columns=['X', 'Y'])
                possible_moves['dist'] = np.sqrt( (possible_moves.X)**2 + (possible_moves.Y)**2)
                possible_moves = possible_moves.sort_values(['dist'],ascending =True)
                while(len(possible_moves) >0 ):
                    move = possible_moves.iloc[0]
                    if(radius_neighbors.shape[0]>0):
                        b = self.nearest(radius_neighbors, move)
                        if(b.dist <= move.dist):
                            return [move['X'], move['Y']]
                        else: 
                            possible_moves.iloc[1:]
                            radius_neighbors.iloc[1:]
                    else: 
                        return [move['X'], move['Y']]
                return [0,0]


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
        """
        # Cap the length of the move
        linespace = 2
        if (self.formation_num == 0):
            move = self.trigonal_planar(rel_pos,linespace,self.orientation)
        elif (self.formation_num == 1):
            move = self.triangle_local(rel_pos,linespace,self.orientation)
        elif (self.formation_num == 2):
            move = self.mapped_formation_move(rel_pos,len(neighbors),0,linespace)
        elif (self.formation_num == 3):
            move = self.mapped_formation_move(rel_pos,len(neighbors),1, linespace)
       
        #self.orientation = np.linalg.norm(move)
        self.target_pos = np.nan_to_num(self.target_pos)
        move = np.nan_to_num(move)
        magnitude = np.linalg.norm(move)
        if magnitude == 0:
            magnitude = 1
        direction = move*(1/magnitude) 
        final_move = direction * min(magnitude, self.fish_max_speed)

        if self.verbose:
            print('Fish #{}: move to {}'.format(self.id, final_move))

        #move = self.define_target(rel_pos,target_map)
        #move = np.nan_to_num(move)
        #magnitude = np.sqrt(move[0]**2 + move[1]**2)
        #final_move = move* min(magnitude, self.fish_max_speed)

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
