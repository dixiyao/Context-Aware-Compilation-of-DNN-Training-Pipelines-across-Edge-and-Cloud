import logging
import os
import pandas as pd

class Logger():
    def __init__(self,name):
        super(Logger,self).__init__()

        self.file=name+'.csv'
        
    def write(self,items,contents):
        frame={x:y for x,y in zip(items,contents)}
        dataframe=pd.DataFrame(frame)
        dataframe.to_csv(self.file,index=False,sep=',')

if __name__=='__main__':
    log=Logger('log')
    log.wirte(["1","2"],[[1,2],[2,3]])
