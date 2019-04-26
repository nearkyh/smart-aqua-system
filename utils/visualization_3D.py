import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class Visualization3D:

    def __init__(self, data_path, start_time, end_time):
        self.data_path = data_path
        self.start_time = start_time
        self.end_time = end_time

    def read_data(self):
        prc = pd.read_csv(self.data_path)
        return prc['coordinates_x'].tolist(), \
               prc['coordinates_y'].tolist(), \
               prc['depth'].tolist(), \
               prc['timestamp'].tolist(), \
               prc['frontCam_w'].tolist(), \
               prc['frontCam_h'].tolist(), \
               prc['sideCam_w'].tolist(), \
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

    def run(self):
        # for numFile in range(10):
        #     fileName = "save_" + str(numFile).rjust(4, '0') + ".csv"
        #     coordinates_x, coordinates_y, depth, history = read_data(save_dir='save_behavior_pattern',
        #                                                                   save_file=fileName)

        coordinates_x, coordinates_y, depth, timestamp, frontCam_w, frontCam_h, sideCam_w, sideCam_h = self.read_data()
        list_x, list_y, list_depth = self.set_time_zone(timestamp=timestamp,
                                                        x=coordinates_x,
                                                        y=coordinates_y,
                                                        depth=depth)

        fig = plt.figure(figsize=(int((900 / 100 + 1) * 3 / 4), int((900 / 100 + 1) * 3 / 4)))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(list_x, list_y, list_depth, color='g', marker='.')

        ax.set_xlim([0, frontCam_w[0]])
        ax.set_ylim([0, sideCam_w[0]])
        ax.set_zlim([frontCam_h[0], 0])

        ax.set_xlabel('front')
        ax.set_ylabel('depth')
        ax.set_zlabel('side')

        plt.show()


if __name__ == '__main__':

    '''
        Input 'start time' and 'end time' shape
        'Year.Month.Day.Hour.Minute.Second'
    '''
    vis3D = Visualization3D(data_path='save_behavior_pattern/save_0000.csv',
                            start_time='2019.4.25.11.24.33',
                            end_time='2019.4.25.11.24.34')
    vis3D.run()
