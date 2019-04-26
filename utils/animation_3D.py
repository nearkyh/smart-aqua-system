import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation


class Animation3D:

    def __init__(self, data_path, start_time, end_time):
        self.data_path = data_path
        self.start_time = start_time
        self.end_time = end_time
        self.fig = plt.figure(figsize=(int((900 / 100 + 1) * 3 / 4), int((900 / 100 + 1) * 3 / 4)))
        self.ax = p3.Axes3D(self.fig)

    def read_data(self):
        prc = pd.read_csv(self.data_path)
        return prc['coordinates_x'].tolist(), \
               prc['coordinates_y'].tolist(), \
               prc['depth'].tolist(),\
               prc['timestamp'].tolist(),\
               prc['frontCam_w'].tolist(),\
               prc['frontCam_h'].tolist(), \
               prc['sideCam_w'].tolist(),\
               prc['sideCam_h'].tolist()

    def set_time_zone(self, timestamp, x, y, depth):
        start_day = self.start_time.split('.')[2]
        start_hour = self.start_time.split('.')[3]
        start_minute = self.start_time.split('.')[4]
        start_second = self.start_time.split('.')[5]

        end_day = self.end_time.split('.')[2]
        end_hour = self.end_time.split('.')[3]
        end_minute = self.end_time.split('.')[4]
        end_second = self.end_time.split('.')[5]

        list_x = []
        list_y = []
        list_depth = []
        for i in range(len(timestamp)):
            time = timestamp[i]
            day = time.split('.')[2]
            hour = time.split('.')[3]
            minute = time.split('.')[4]
            second = time.split('.')[5]
            if (start_day <= day) and (day <= end_day):
                if (start_hour <= hour) and (hour <= end_hour):
                    if (start_minute <= minute) and (minute <= end_minute):
                        if (start_second <= second) and (second <= end_second):
                            list_x.append(x[i])
                            list_y.append(y[i])
                            list_depth.append(depth[i])

        return list_x, list_y, list_depth

    def update(self, num):
        coordinates_x, coordinates_y, depth, timestamp, frontCam_w, frontCam_h, sideCam_w, sideCam_h = self.read_data()

        x, y, depth = self.set_time_zone(timestamp=timestamp,
                                         x=coordinates_x,
                                         y=coordinates_y,
                                         depth=depth)

        data_list = [x, y, depth]
        data = np.array(data_list)

        line, = self.ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])
        line.set_data(data[:2, :num])
        line.set_3d_properties(data[2, :num])

        # Setting the axes properties
        self.ax.set_xlim3d([0, frontCam_w[0]])
        self.ax.set_xlabel('X')

        self.ax.set_ylim3d([0, sideCam_w[0]])
        self.ax.set_ylabel('Depth')

        self.ax.set_zlim3d([sideCam_h[0], 0])
        self.ax.set_zlabel('Y')

    def run_animation(self):
        N = 100
        ani = animation.FuncAnimation(self.fig, self.update, N, interval=10000 / N, blit=False)
        # ani.save('3D_Mapping.gif', writer='imagemagick')
        plt.show()


if __name__ == '__main__':

    animation3D = Animation3D(data_path='save_behavior_pattern/save_0000.csv',
                              start_time='2019.4.25.11.24.33',
                              end_time='2019.4.25.11.24.34')
    animation3D.run_animation()
