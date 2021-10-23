import socket
import pickle
import torch
import time
import psutil

import sys
import utils.utils as utils

#speedtest and model update use another client&server on another port

def get_network_data():
    recv = {}
    sent = {}
    data = psutil.net_io_counters(pernic=True)
    interfaces = data.keys()
    for interface in interfaces:
        recv.setdefault(interface, data.get(interface).bytes_recv)
        sent.setdefault(interface, data.get(interface).bytes_sent)
    return interfaces, recv, sent

class Server():
    def __init__(self,address,port):
        super(Server, self).__init__()

        self.delegate= socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.delegate.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.delegate.bind((address,port))
        self.rec_data=b''
        self.rec_data=b''
        self.delegate.listen(5)
        self.delegate, addr = self.delegate.accept()
        #for speedtest
        self.send_size=0
        self.send_time=0
        self.upload=0
        self.download=0
        self.store=0
        self.new_store=1

    def send_tensor(self,head,epoch,iters,index,feature, use_Q):
        sendtime=time.time()
        if not use_Q:
            feature=feature.numpy()
            feature=pickle.dumps(feature)
        words=b'head'+pickle.dumps(head)+b'epoch'+pickle.dumps(epoch)+\
        b'iters'+pickle.dumps(iters)+b'index'+pickle.dumps(index)+\
        b'tensor'+feature+b'quantization'+pickle.dumps(use_Q)
        words=words+b'clientsend'+pickle.dumps(sendtime)+b'sendsize'+pickle.dumps(len(words))+b'end of feature'
        #send and test speed
        self.delegate.sendall(words)

        return len(words), sendtime

    def recieve_tensor(self):
        #recv and test speed
        while b'end of gradient' not in self.rec_data:
            self.rec_data+=self.delegate.recv(10240000)
        #deserialization
        head_pos=self.rec_data.find(b'head')
        epoch_pos=self.rec_data.find(b'epoch')
        iters_pos=self.rec_data.find(b'iters')
        gradient_pos=self.rec_data.find(b'gradient')
        rec_pos=self.rec_data.find(b'serverrec')
        send_pos=self.rec_data.find(b'serversend')
        clientsend_pos=self.rec_data.find(b'clientsend')
        clientsendsize_pos=self.rec_data.find(b'sendsize')
        eof_pos=self.rec_data.find(b'end of gradient')
        head=pickle.loads(self.rec_data[head_pos+4:epoch_pos])
        epoch=pickle.loads(self.rec_data[epoch_pos+5:iters_pos])
        iters=pickle.loads(self.rec_data[iters_pos+5:gradient_pos])
        gradient=pickle.loads(self.rec_data[gradient_pos+8:rec_pos])
        server_rec=pickle.loads(self.rec_data[rec_pos+9:send_pos])
        server_send=pickle.loads(self.rec_data[send_pos+10:clientsend_pos])
        client_send=pickle.loads(self.rec_data[clientsend_pos+10:clientsendsize_pos])
        client_send_size=pickle.loads(self.rec_data[clientsendsize_pos+8:eof_pos])
        gradient=torch.from_numpy(gradient)
        rec_data_length=len(self.rec_data)
        self.rec_data=self.rec_data[eof_pos+15:]
        return head,epoch,iters,gradient,client_send,server_rec,client_send_size,server_send,rec_data_length

    def send_model(self, index,models, file_path="client_model.pt"):
        '''
            save client_model to file_path and send it to server
            Usage: client.send_model(client_model, "./xxxx/xxxx.pt")
        '''
        index_word=b'index start'+pickle.dumps(index)+b'index end'
        self.delegate.sendall(index_word)
        for model in models:
            torch.save(model.state_dict(), file_path,_use_new_zipfile_serialization=False)
            fin = open(file_path, 'rb')
            self.delegate.sendall(b'begin of model'+fin.read()+b'end of model')
            fin.close()

    def recieve_and_update_model(self,models,file_path="server_model.pt"):
        while b'index end' not in self.rec_data:
            self.rec_data+=self.delegate.recv(10240000)
        begin_pos=self.rec_data.find(b'index start')
        end_pos=self.rec_data.find(b'index end')
        index=pickle.loads(self.rec_data[begin_pos+11:end_pos])
        self.rec_data=self.rec_data[end_pos+9:]
        for i in range(index[0],index[1]+1):
            while b'end of model' not in self.rec_data:
                self.rec_data += self.delegate.recv(10240000)
            data = self.rec_data[self.rec_data.find(b'begin of model')+14:self.rec_data.find(b'end of model')]
            fout = open(file_path, 'wb')
            fout.write(data)
            fout.close()
            self.rec_data = self.rec_data[self.rec_data.find(b'end of model')+12:]
            models[i].load_state_dict(torch.load(file_path))

    def testspeed(self):
        pk1=torch.rand(1024,1024,1,1)
        pk2=torch.rand(1,1,1,1)
        s1=utils.count_tensorsize_in_B(pk1)/1024/1024
        s2=utils.count_tensorsize_in_B(pk2)/1024/1024

        start1=time.perf_counter()
        self.send_tensor('test speed',-1,-1,torch.tensor([-1.1]),pk1,False)
        _=self.recieve_tensor()
        end1=time.perf_counter()
        t1=end1-start1
        start2=time.perf_counter()
        self.send_tensor('test speed',-1,-1,torch.tensor([-1.1]),pk2,False)
        _=self.recieve_tensor()
        end2=time.perf_counter()
        t2=end2-start2
        u=s1/t1
        d=s1/t2
        return u,d


class Client():
    def __init__(self,address,port):
        super(Client, self).__init__()

        self.delegate= socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.delegate.connect((address,port))

    def send_tensor(self,head,epoch,iters,gradient,server_rec,client_send,send_size):
        gradient=gradient.numpy()
        server_send=time.time()
        words=b'head'+pickle.dumps(head)+b'epoch'+pickle.dumps(epoch)+b'iters'+\
               pickle.dumps(iters)+b'gradient'+\
               pickle.dumps(gradient)+\
               b'serverrec'+pickle.dumps(server_rec)+b'serversend'+pickle.dumps(server_send)+\
               b'clientsend'+pickle.dumps(client_send)+b'sendsize'+pickle.dumps(send_size)+b'end of gradient'
        self.delegate.sendall(words)

    def recieve_tensor(self):
        while b'end of feature' not in self.rec_data:
            self.rec_data+=self.delegate.recv(10240000)
        server_rec=time.time()
        head_pos=self.rec_data.find(b'head')
        epoch_pos=self.rec_data.find(b'epoch')
        iters_pos=self.rec_data.find(b'iters')
        index_pos=self.rec_data.find(b'index')
        tensor_pos=self.rec_data.find(b'tensor')
        Q_pos=self.rec_data.find(b'quantization')
        clientsendpos=self.rec_data.find(b'clientsend')
        sendsizepos=self.rec_data.find(b'sendsize')
        eof_pos=self.rec_data.find(b'end of feature')
        head=pickle.loads(self.rec_data[head_pos+4:epoch_pos])
        epoch=pickle.loads(self.rec_data[epoch_pos+5:iters_pos])
        iters=pickle.loads(self.rec_data[iters_pos+5:target_pos])
        index=pickle.loads(self.rec_data[index_pos+5:tensor_pos])
        feature=pickle.loads(self.rec_data[tensor_pos+6:Q_pos])
        use_Q=pickle.loads(self.rec_data[Q_pos+12:clientsendpos])
        client_send=pickle.loads(self.rec_data[clientsendpos+10:sendsizepos])
        send_size=pickle.loads(self.rec_data[sendsizepos+8:eof_pos])
        self.rec_data=self.rec_data[eof_pos+14:]
        target=torch.from_numpy(target)
        if not use_Q:
            feature=torch.from_numpy(feature)

        return head,epoch,iters,index,feature,use_Q,server_rec,client_send,send_size

    def send_model(self, index,models, file_path="client_model.pt"):
        '''
            save client_model to file_path and send it to server
            Usage: client.send_model(client_model, "./xxxx/xxxx.pt")
        '''
        index_word=b'index start'+pickle.dumps(index)+b'index end'
        self.delegate.sendall(index_word)
        for model in models:
            torch.save(model.state_dict(), file_path)
            fin = open(file_path, 'rb')
            self.delegate.sendall(b'begin of model'+fin.read()+b'end of model')
            fin.close()

    def recieve_and_update_model(self,models,file_path="server_model.pt"):
        while b'index end' not in self.rec_data:
            self.rec_data+=self.delegate.recv(10240000)
        begin_pos=self.rec_data.find(b'index start')
        end_pos=self.rec_data.find(b'index end')
        index=pickle.loads(self.rec_data[begin_pos+11:end_pos])
        self.rec_data=self.rec_data[end_pos+9:]
        for i in range(index[0],index[1]+1):
            while b'end of model' not in self.rec_data:
                self.rec_data += self.delegate.recv(10240000)
            data = self.rec_data[self.rec_data.find(b'begin of model')+14:self.rec_data.find(b'end of model')]
            fout = open(file_path, 'wb')
            fout.write(data)
            fout.close()
            self.rec_data = self.rec_data[self.rec_data.find(b'end of model')+12:]
            models[i].load_state_dict(torch.load(file_path))
        
    def testspeed(self):
        pk1=torch.rand(1024,1024,1,1)
        pk2=torch.rand(1,1,1,1)

        _=self.recieve_tensor()
        self.send_tensor('test speed',-1,-1,torch.tensor([-1.1]),pk2,torch.tensor([-1.1]))
        _=self.recieve_tensor()
        self.send_tensor('test speed',-1,-1,torch.tensor([-1.1]),pk1,torch.tensor([-1.1]))        
    
