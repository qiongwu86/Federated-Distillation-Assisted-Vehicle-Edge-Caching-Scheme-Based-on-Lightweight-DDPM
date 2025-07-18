import numpy as np
import math
class V2Ichannels:


    def __init__(self):
        self.h_bs = 25
        self.h_ms = 1.5
        self.BS_position = [0, 1000, 2000]
        self.BS_position_rsu = [600,1200,1800]
        self.BS_position_mbs = 1500
        self.shadow_std = 4
        self.Decorrelation_distance = 50
        self.Decorrelation_distance_V2V = 10
        self.RSU_road = 100
        self.MBS_road = 500
        self.frequency_GHz_v2i = 2.4

    def get_path_loss(self, position):
        distance=0
        if self.BS_position[0]<position<self.BS_position[1]:
            distance = abs(position - self.BS_position_rsu[0])
        if self.BS_position[1]<position<self.BS_position[2]:
            distance = abs(position - self.BS_position_rsu[1])
        if position>self.BS_position[2]:
            distance = (position - self.BS_position_rsu[2])
        distance = math.sqrt(distance ** 2 +  self.RSU_road** 2)
        return 32.4 + 20 * np.log10(self.frequency_GHz_v2i) + 20 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)
    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)


class Environ:

    def __init__(self, n_veh):

        self.V2Ichannels = V2Ichannels()

        self.V2I_Shadowing = []
        self.delta_distance = []
        self.V2I_channels_abs = []
        self.decorrelation_distance = 50
        self.sig2_dB = -114
        self.bsAntGain = 8
        self.vehAntGain = 3
        self.bsNoiseFigure = 5
        self.vehNoiseFigure = 9
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.n_Veh = n_veh
        self.time_fast = 0.001
        self.time_slow = 0.1
        self.bandwidth = 1000000
        self.bandwidth_mbs=2000000
        self.bandwidth_V2V =10000000

    def Compute_Performance_Train_mobility(self):


        self.platoon_V2I_Signal = np.zeros(self.n_Veh)

        for i in range(self.n_Veh):
            self.platoon_V2I_Signal[i] = 10 ** ((30 - self.V2I_channels_with_fastfading[i][0] +
                                                 self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(self.platoon_V2I_Signal, self.sig2))
        self.interplatoon_rate = V2I_Rate * self.bandwidth
        return self.interplatoon_rate

    def renew_channel(self, number_vehicle,veh_dis,veh_speed):
        self.V2I_Shadowing = np.random.normal(0, 4, number_vehicle)
        self.V2I_pathloss = np.zeros((number_vehicle))
        self.V2I_channels_abs = np.zeros((number_vehicle))
        self.delta_distance = np.asarray([c * self.time_slow for c in veh_speed])
        self.V2I_Shadowing = self.V2Ichannels.get_shadowing(self.delta_distance, self.V2I_Shadowing)
        for i in range(number_vehicle):
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(veh_dis[i])
        self.V2I_channels_abs = self.V2I_pathloss + self.V2I_Shadowing
    def renew_channels_fastfading(self):

        V2I_channels_with_fastfading = self.V2I_channels_abs[:, np.newaxis]
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_channels_with_fastfading.shape) +
                   1j * np.random.normal(0, 1, V2I_channels_with_fastfading.shape)) / math.sqrt(2))
    def new_random_game(self, veh_dis,veh_speed):
        self.renew_channel(int(self.n_Veh),veh_dis,veh_speed)
        self.renew_channels_fastfading()
