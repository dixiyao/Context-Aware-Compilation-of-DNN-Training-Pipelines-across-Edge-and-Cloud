import psutil
import time

def getMemCpu():
    data=psutil.virtual_memory()
    return data.percent

for i in range(10):
    print(getMemCpu())
    time.sleep(3)
