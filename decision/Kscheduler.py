class Ksch():
    def __init__(self,dataset,init_K,step_size=1):
        super(Ksch,self).__init__()

        self.epoch=0
        self.K=init_K
        if dataset=='imdb':
            self.sch=[8]
        if dataset=='tiny_imagenet':
            self.sch=[3]
        else:
            self.sch=[-1]
        if self.K==0:
            self.sch=[-1]

    def step(self):
        self.epoch+=1
        if self.epoch in self.sch:
            self.K+=1
        return self.K
        
