from math import *
import numpy as np


class Dynamics():

    def __init__(self, environment, clock_freq=1):
        self.environment = environment

        # Simulation Step and Time
        self.deltat = 0.01 # [s]
        self.t_simu = 1 / clock_freq # [s]

        # Robot Specs
        self.rho = 998 # [kg/m^3], water density
        self.l_robot = 0.150 # [m], including fin
        self.w_robot = 0.050 # [m]
        self.h_robot = 0.080 # [m]
        self.A_x = pi/4 * self.h_robot * self.w_robot # [m**2]
        self.A_y = pi/4 * self.l_robot * self.h_robot # [m**2]
        self.A_z = pi/4 * self.l_robot * self.w_robot # [m**2]
        self.A_phi = self.A_y
        self.m_robot = 2*0.25 # [kg], including added mass
        self.I_robot = self.m_robot/5 * 1/4*(self.l_robot**2 + self.h_robot**2) # [kg*m**2]
        self.C_dx = 0.5
        self.C_dy = 5.0
        self.C_dz = 0.9
        self.C_dphi = 1.2
        self.pect_dist = 0.055 # [m]
        self.pect_angle = pi / 6 # [rad]
        self.F_buoy = 0.011 # [N]

        # Initialize Control
        self.F_caud = 0 # [N]
        self.F_PR = 0 # [N]
        self.F_PL = 0 # [N]
        self.F_dors = 0 # [N]

    def update_ctrl(self, dorsal, caudal, pect_r, pect_l):
        F_caud_max = 0.022 # [N]
        F_PR_max = 0.010 # [N]
        F_PL_max = 0.010 # [N]
        F_dors_max = 0.022 # [N]

        self.F_caud = caudal * F_caud_max
        self.F_PR = pect_r * F_PR_max
        self.F_PL = pect_l * F_PL_max
        self.F_dors = dorsal * F_dors_max #xx change dorsal

    def simulate_move(self, source_id):
        g_P_r = np.zeros((3,))
        g_Pdot_r = 1/1000 * self.environment.node_vel[source_id]
        phi = self.environment.node_phi[source_id]
        vphi = self.environment.node_vphi[source_id]

        r_T_g = np.array([[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0, 0, 1]])
        r_Pdot_r = r_T_g @ g_Pdot_r #xx
        vx = r_Pdot_r[0] #xx
        vy = r_Pdot_r[1]
        vz = r_Pdot_r[2]

        for t in range(int(self.t_simu*1/self.deltat)):
            # Equations of Motion
            x_dot = vx
            y_dot = vy
            z_dot = vz
            phi_dot = vphi

            vx_dot = 1/self.m_robot * (self.F_caud - sin(self.pect_angle)*self.F_PL - sin(self.pect_angle)*self.F_PR - 1/2*self.rho*self.C_dx*self.A_x*np.sign(x_dot)*x_dot**2)
            vy_dot = 1/self.m_robot * (cos(self.pect_angle)*self.F_PL - cos(self.pect_angle)*self.F_PR - 1/2*self.rho*self.C_dy*self.A_y*np.sign(y_dot)*y_dot**2)
            vz_dot = 1/self.m_robot * (self.F_dors - self.F_buoy - 1/2*self.rho*self.C_dz*self.A_z*np.sign(z_dot)*z_dot**2)
            vphi_dot = 1/self.I_robot * (self.pect_dist*cos(self.pect_angle)*self.F_PL - self.pect_dist*cos(self.pect_angle)*self.F_PR - 1/2*self.rho*self.C_dphi*self.A_phi*np.sign(phi_dot)*(self.l_robot/6*phi_dot)**2)

            # Euler Integration
            vx = x_dot + self.deltat*vx_dot
            vy = y_dot + self.deltat*vy_dot
            vz = z_dot + self.deltat*vz_dot

            phi = phi + self.deltat*phi_dot
            vphi = phi_dot + self.deltat*vphi_dot

            # Robot to Global Transformation
            g_T_r = np.array([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])
            g_Pdot_r = g_T_r @ np.array([vx, vy, vz])
            g_P_r = g_P_r + self.deltat*np.transpose(g_Pdot_r)

        self.environment.node_vel[source_id] = 1000 * g_Pdot_r #xx
        self.environment.node_phi[source_id] = phi
        self.environment.node_vphi[source_id] = vphi

        #print(self.environment.node_vel[source_id])

        return 1000* g_P_r #xx

