import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class Visualization3D:

    def __init__(self):
        self.coordinates_x = []
        self.coordinates_y = []
        self.depth = []
        self.timestamp = []
        self.frontCam_w = []
        self.frontCam_h = []
        self.sideCam_w = []
        self.sideCam_h = []

    def read_data(self, data_path):
        prc = pd.read_csv(data_path)
        self.coordinates_x = prc['coordinates_x'].tolist()
        self.coordinates_y = prc['coordinates_y'].tolist()
        self.depth = prc['depth'].tolist()
        self.timestamp = prc['timestamp'].tolist()
        self.frontCam_w = prc['frontCam_w'].tolist()
        self.frontCam_h = prc['frontCam_h'].tolist()
        self.sideCam_w = prc['sideCam_w'].tolist()
        self.sideCam_h = prc['sideCam_h'].tolist()

    def set_time_zone(self, start_time, end_time, timestamp, x, y, depth):
        '''
            Input shape (start_time, end_time)
            start_time = 'Year.Month.Day.Hour.Minute.Second'
            end_time = 'Year.Month.Day.Hour.Minute.Second'
        '''
        start_day = int(start_time.split('.')[2])
        start_hour = int(start_time.split('.')[3])
        start_minute = int(start_time.split('.')[4])
        start_second = int(start_time.split('.')[5])

        end_day = int(end_time.split('.')[2])
        end_hour = int(end_time.split('.')[3])
        end_minute = int(end_time.split('.')[4])
        end_second = int(end_time.split('.')[5])

        list_x = []
        list_y = []
        list_depth = []
        for i in range(len(timestamp)):
            time = timestamp[i]
            day = int(time.split('.')[2])
            hour = int(time.split('.')[3])
            minute = int(time.split('.')[4])
            second = int(time.split('.')[5])
            if (start_day <= day) and (day <= end_day):
                if (start_hour <= hour) and (hour <= end_hour):
                    if (start_minute <= minute) and (minute <= end_minute):
                        if (start_second <= second) and (second <= end_second):
                            list_x.append(x[i])
                            list_y.append(y[i])
                            list_depth.append(depth[i])

        return list_x, list_y, list_depth

    def run(self):
        # for numFile in range(10):
        #     fileName = "save_" + str(numFile).rjust(4, '0') + ".csv"
        #     coordinates_x, coordinates_y, depth, history = read_data(save_dir='save_behavior_pattern',
        #                                                                   save_file=fileName)

        fig = plt.figure(figsize=(int((900 / 100 + 1) * 3 / 4), int((900 / 100 + 1) * 3 / 4)))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.coordinates_x, self.coordinates_y, self.depth, color='g', marker='.')

        ax.set_xlim([0, self.frontCam_w[0]])
        ax.set_ylim([0, self.sideCam_w[0]])
        ax.set_zlim([self.frontCam_h[0], 0])

        ax.set_xlabel('front')
        ax.set_ylabel('depth')
        ax.set_zlabel('side')

        plt.show()


if __name__ == '__main__':

    visualization3D = Visualization3D()
    visualization3D.read_data('save_behavior_pattern/save_0000.csv')
    visualization3D.run()
