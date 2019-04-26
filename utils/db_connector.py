import pymysql
import os


class DBConnector:

    def __init__(self):
        self.host = os.getenv('mysql_host')
        self.port = int(os.getenv('mysql_port'))
        self.user = os.getenv('mysql_user')
        self.password = os.getenv('mysql_password')
        self.db = 'smart_aquarium'
        self.charset = 'utf8'

    def connect_mysql(self):
        return pymysql.connect(host=self.host,
                               port=self.port,
                               user=self.user,
                               password=self.password,
                               db=self.db,
                               charset=self.charset)

    def close_mysql(self):
        self.connect_mysql().close()

    def insert_data(self, curs, conn, coordinates_x, coordinates_y, depth, timestamp, frontCam_w, frontCam_h, sideCam_w, sideCam_h):
        sql = """insert into data(coordinates_x, coordinates_y, depth, timestamp, frontCam_w, frontCam_h, sideCam_w, sideCam_h)
                 values({0},{1},{2},{3},{4},{5},{6},{7})""".format(coordinates_x, coordinates_y, depth, timestamp, frontCam_w, frontCam_h, sideCam_w, sideCam_h)
        curs.execute(sql)
        conn.commit()

    def select_data(self):
        curs = self.connect_mysql().cursor()
        sql = "select * from data"
        curs.execute(sql)

        return curs.fetchall()
