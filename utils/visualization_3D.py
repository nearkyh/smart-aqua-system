import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class Visualization3D:

    def __init__(self):
        self.coordinates_x = []
        self.coordinates_y = []
        self.depth = []
        self.speed = []
        self.timestamp = []
        self.frontCam_w = []
        self.frontCam_h = []
        self.sideCam_w = []
        self.sideCam_h = []
        # Pre-processing data
        self.pre_x = []
        self.pre_y = []
        self.pre_depth = []

    def read_data(self, data_path):
        prc = pd.read_csv(data_path)
        self.coordinates_x = prc['coordinates_x'].tolist()
        self.coordinates_y = prc['coordinates_y'].tolist()
        self.depth = prc['depth'].tolist()
        self.speed = prc['speed'].tolist()
        self.timestamp = prc['timestamp'].tolist()
        self.frontCam_w = prc['frontCam_w'].tolist()
        self.frontCam_h = prc['frontCam_h'].tolist()
        self.sideCam_w = prc['sideCam_w'].tolist()
        self.sideCam_h = prc['sideCam_h'].tolist()

        self.pre_processing()

    def pre_processing(self):
        # rate = []
        # for i in range(len(self.depth)):
        #     rate.append(self.depth[i] / self.sideCam_w[i])

        # x_list = []
        # y_list = []
        # depth_list = []
        # for i in range(len(self.depth)):
        #     if self.depth[i] == min(self.depth):
        #         x_list.append(self.coordinates_x[i])
        #         y_list.append(self.coordinates_y[i])
        #         # depth_list.append(self.depth[i])
        # x_average = int(sum(x_list) / len(x_list))
        # y_average = int(sum(y_list) / len(y_list))
        # # depth_average = int(sum(depth_list) / len(depth_list))
        # print(x_average, y_average)

        x_rate = 1/5
        self.pre_x = []
        for i in range(len(self.depth)):
            if self.coordinates_x[i] > 325:
                self.pre_x.append(self.coordinates_x[i] + int(self.depth[i] * x_rate))
            elif self.coordinates_x[i] < 315:
                self.pre_x.append(self.coordinates_x[i] - int(self.depth[i] * x_rate))
            else:
                self.pre_x.append(self.coordinates_x[i])

        y_rate = 1/6
        self.pre_y = []
        for i in range(len(self.depth)):
            if self.coordinates_y[i] > 245:
                self.pre_y.append(self.coordinates_y[i] + int(self.depth[i] * y_rate))
            elif self.coordinates_y[i] < 235:
                self.pre_y.append(self.coordinates_y[i] - int(self.depth[i] * y_rate))
            else:
                self.pre_y.append(self.coordinates_y[i])

        depth_rate = 1/5
        self.pre_depth = []
        for i in range(len(self.depth)):
            x = abs(self.coordinates_x[i]-self.frontCam_w[-1])
            if self.depth[i] > 325:
                self.pre_depth.append(self.depth[i] + int(x * depth_rate))
            elif self.depth[i] < 315:
                self.pre_depth.append(self.depth[i] - int(x * depth_rate))
            else:
                self.pre_depth.append(self.depth[i])

        pass

    def set_time_zone(self,
                      start_time,
                      end_time,
                      x,
                      y,
                      depth,
                      speed,
                      timestamp,
                      frontW,
                      frontH,
                      sideW,
                      sideH):
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

        # Check the day
        check_day = []
        for i in range(len(timestamp)):
            time = timestamp[i]
            day = int(time.split('.')[2])
            if (start_day <= day) and (day <= end_day):
                check_day.append(timestamp[i])

        # Check the hour
        check_hour = []
        for i in range(len(check_day)):
            time = check_day[i]
            hour = int(time.split('.')[3])
            if (start_hour <= hour) and (hour <= end_hour):
                check_hour.append(check_day[i])

        # Find (start & end) minute index
        check_start_minute = []
        check_end_minute = []
        for i in range(len(check_hour)):
            time = check_hour[i]
            hour = int(time.split('.')[3])
            minute = int(time.split('.')[4])
            if (hour == start_hour) and (minute == start_minute):
                check_start_minute.append(time)
            if (hour == end_hour) and minute == end_minute:
                check_end_minute.append(time)
        start_index_minute = check_hour.index(check_start_minute[0])
        end_index_minute = check_hour.index(check_end_minute[-1])

        # Check the minute
        check_minute = []
        for i in range(len(check_hour)):
            check_minute = check_hour[start_index_minute : end_index_minute]

        # Find (start & end) second index
        check_start_second = []
        check_end_second = []
        for i in range(len(check_minute)):
            time = check_minute[i]
            hour = int(time.split('.')[3])
            minute = int(time.split('.')[4])
            if (hour == start_hour) and (minute == start_minute):
                check_start_second.append(time)
            if (hour == end_hour) and minute == end_minute:
                check_end_second.append(time)
        start_index_second = check_minute.index(check_start_second[0])
        end_index_second = check_minute.index(check_end_second[-1])

        # Check the minute
        check_second = []
        for i in range(len(check_minute)):
            check_second = check_minute[start_index_second : end_index_second]

        # Select data
        list_x = []
        list_y = []
        list_depth = []
        list_speed = []
        list_timestamp = []
        list_frontW = []
        list_frontH = []
        list_sideW = []
        list_sideH = []
        for i in range(len(check_second)):
            list_x.append(x[i])
            list_y.append(y[i])
            list_depth.append(depth[i])
            list_speed.append(speed[i])
            list_timestamp.append(timestamp[i])
            list_frontW.append(frontW[i])
            list_frontH.append(frontH[i])
            list_sideW.append(sideW[i])
            list_sideH.append(sideH[i])
                
        return list_x, list_y, list_depth, list_speed, list_timestamp,\
               list_frontW, list_frontH, list_sideW, list_sideH

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
    visualization3D.read_data('../save_pattern/20190606.csv')
    visualization3D.run()
