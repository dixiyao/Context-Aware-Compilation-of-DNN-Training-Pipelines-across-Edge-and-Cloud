import time

import datetime

class Timer:
    def __init__(self):
        self.start_time = 0
        
    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.start_time = self.current_time

    def elapsed(self):
        self.current_time = time.time()
        print('time {} elapsed!'.format(self.current_time - self.start_time))

    @staticmethod
    def timeString():
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        return st

    @staticmethod
    def timeFilenameString():
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d[%H_%M_%S]')
        return st
