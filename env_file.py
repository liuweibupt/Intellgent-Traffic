import time
import numpy as np
import pandas as pd
import win32api

exe_path = '..\\newProject.exe'
out_path = '..\\newProject\\output.csv'
in_path = '..\\newProject\\input.csv'

META_TIME = 0.05


class Crossroad(object):
    def __init__(self):
        self.crush = False
        self.env_data = pd.DataFrame
        self.temp_flag = True
        self.acc = pd.DataFrame

        # self.velocity=pd.read_csv(in_path,usecols=[1],dtype={'speed':np.float32})

        self.V_ini = pd.DataFrame(np.ones(50) * 10)
        self.velocity = self.V_ini

        self.counter_list = np.zeros(50, dtype=int)
        self.time_counter = 0
        self.more, self.less = [], []
        self.get_data()
        self.car_temp = self.env_data

        win32api.ShellExecute(0, 'open', exe_path, '', '', 1)

    def initial(self):

        self.V_ini.to_csv(in_path)
        ini = True
        while ini:
            try:
                f = open(out_path, 'w')
                f.truncate()
                f.close()
                ini = False
                print('reset complete')
            except:
                pass

    def get_data(self):
        try:
            self.env_data = pd.read_csv(out_path,
                                        header=None,
                                        names=['bool_straight', 'bool_left', 'x_position', 'y_position', 'angle',
                                               'CRUSH']) \
                .dropna()
        except:
            pass

    def detect_crush(self):

        self.crush = False
        flag = any(self.env_data.CRUSH.tolist())
        if flag is True and self.temp_flag is False:
            print("crush")
            self.crush = True
        self.temp_flag = flag
        return self.crush

    def generate_reward(self):
        car = self.env_data
        self.more = list(set(car.index.tolist()) - set(self.car_temp.index.tolist()))
        self.less = list(set(self.car_temp.index.tolist()) - set(car.index.tolist()))
        if any(self.env_data.CRUSH.tolist()) is True:
            reward = -100 * len(car.index.tolist())
            time.sleep(0.5)
        else:
            V_list = self.velocity.to_numpy().flatten()
            V_list = V_list[car.index.tolist()]
            V_reward = 0
            if len(V_list) != 0:
                for v in V_list:
                    if v > 50:
                        V_reward += 0
                    else:
                        V_reward += 10 * np.log2(v * 0.2)
                V_reward /= len(V_list)
            reward = len(self.less) * 10 + V_reward

        self.car_temp = car
        if reward != 0:
            print(reward)
        return reward

    def test_print(self):
        print('')

        # more file output

    def state_output(self):
        output_data = np.zeros((50, 2))
        _v = np.zeros(50)
        obs = self.env_data[['x_position', 'y_position']]
        index_list = obs.index.to_list()
        for i in index_list:
            _v[i] = self.velocity.loc[i,]
            output_data[i] = obs.loc[i].to_list()  # 范围等比例缩小利于计算
        output_data = np.column_stack((output_data, _v))
        return output_data.flatten()

    def step(self, action_acc):
        # action_acc is 50*1
        self.acc = pd.DataFrame(action_acc, columns=self.velocity.columns)
        for i in self.env_data.index.tolist():
            self.velocity.loc[i] += self.acc.loc[i] * META_TIME
            if self.velocity.loc[i].item() < 1:
                self.velocity.loc[i] = 1
        try:
            self.velocity.to_csv(in_path)
        except:
            pass
        time.sleep(META_TIME)
        self.get_data()
        return self.state_output(), self.detect_crush(), self.generate_reward()

    def detect_car(self):
        '''
        car = self.env_data
        more, less = [], []
        if self.detect_crush() is False:

            self.more = list(set(car.index.to_numpy()) - set(self.car_temp.index.to_numpy()))
            self.less = list(set(self.car_temp.index.to_numpy()) - set(car.index.to_numpy()))
        if self.more:
            more = self.more
        if self.less:
            less = self.less
        self.car_temp = car
        print(self.more,self.less)
        return more, less
        '''


# test
if __name__ == '__main__':
    cr = Crossroad()
    cr.initial()
    cr.get_data()
    cr.test_print()
    # acc_list=np.ones((50,1))

    while True:
        cr.generate_reward()
        cr.detect_crush()
        cr.get_data()
