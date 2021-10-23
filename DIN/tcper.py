import socket
import pickle
import torch
import time
import psutil

import sys
sys.path.append('../')
import Utils.utils as utils

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

class Client():
    def __init__(self,address,port):
        super(Client, self).__init__()

        self.delegate= socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.delegate.connect((address,port))
        self.rec_data=b''
        #for speedtest
        self.send_size=0
        self.send_time=0
        self.upload=0
        self.download=0
        self.store=0
        self.new_store=1

    def send_tensor(self,head,epoch,iters,target,m,a,use_Q):
        sendtime=time.time()
        target=target.numpy()
        if not use_Q==True:
            m=m.numpy()
            m=pickle.dumps(m)
            a=a.numpy()
            a=pickle.dumps(a)
        words=b'head'+pickle.dumps(head)+b'epoch'+pickle.dumps(epoch)+\
        b'iters'+pickle.dumps(iters)+b'target'+pickle.dumps(target)+\
        b'tensormovie'+m+b'tensorads'+a+b'quantization'+pickle.dumps(use_Q)
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
        result_pos=self.rec_data.find(b'result')
        gradient_pos=self.rec_data.find(b'gradient')
        rec_pos=self.rec_data.find(b'serverrec')
        send_pos=self.rec_data.find(b'serversend')
        clientsend_pos=self.rec_data.find(b'clientsend')
        clientsendsize_pos=self.rec_data.find(b'sendsize')
        eof_pos=self.rec_data.find(b'end of gradient')
        head=pickle.loads(self.rec_data[head_pos+4:epoch_pos])
        epoch=pickle.loads(self.rec_data[epoch_pos+5:iters_pos])
        iters=pickle.loads(self.rec_data[iters_pos+5:result_pos])
        result=pickle.loads(self.rec_data[result_pos+6:gradient_pos])
        gradient=pickle.loads(self.rec_data[gradient_pos+8:rec_pos])
        server_rec=pickle.loads(self.rec_data[rec_pos+9:send_pos])
        server_send=pickle.loads(self.rec_data[send_pos+10:clientsend_pos])
        client_send=pickle.loads(self.rec_data[clientsend_pos+10:clientsendsize_pos])
        client_send_size=pickle.loads(self.rec_data[clientsendsize_pos+8:eof_pos])
        result=torch.from_numpy(result)
        gradient=torch.from_numpy(gradient)
        rec_data_length=len(self.rec_data)
        self.rec_data=self.rec_data[eof_pos+15:]
        return head,epoch,iters,result,gradient,client_send,server_rec,client_send_size,server_send,rec_data_length

class Server():
    def __init__(self,address,port):
        super(Server, self).__init__()

        self.delegate= socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.delegate.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.delegate.bind((address,port))
        self.rec_data=b''
        self.delegate.listen(5)
        self.delegate, addr = self.delegate.accept()

    def send_tensor(self,head,epoch,iters,result,gradient,server_rec,client_send,send_size):
        result=result.numpy()
        gradient=gradient.numpy()
        server_send=time.time()
        words=b'head'+pickle.dumps(head)+b'epoch'+pickle.dumps(epoch)+b'iters'+\
               pickle.dumps(iters)+b'result'+pickle.dumps(result)+b'gradient'+\
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
        target_pos=self.rec_data.find(b'target')
        m_pos=self.rec_data.find(b'tensormovie')
        a_pos=self.rec_data.find(b'tensorads')
        Q_pos=self.rec_data.find(b'quantization')
        clientsendpos=self.rec_data.find(b'clientsend')
        sendsizepos=self.rec_data.find(b'sendsize')
        eof_pos=self.rec_data.find(b'end of feature')
        head=pickle.loads(self.rec_data[head_pos+4:epoch_pos])
        epoch=pickle.loads(self.rec_data[epoch_pos+5:iters_pos])
        iters=pickle.loads(self.rec_data[iters_pos+5:target_pos])
        target=pickle.loads(self.rec_data[target_pos+6:m_pos])
        m=pickle.loads(self.rec_data[m_pos+11:a_pos])
        a=pickle.loads(self.rec_data[a_pos+9:Q_pos])
        use_Q=pickle.loads(self.rec_data[Q_pos+12:clientsendpos])
        client_send=pickle.loads(self.rec_data[clientsendpos+10:sendsizepos])
        send_size=pickle.loads(self.rec_data[sendsizepos+8:eof_pos])
        self.rec_data=self.rec_data[eof_pos+14:]
        target=torch.from_numpy(target)
        if not use_Q==True:
            m=torch.from_numpy(m)
            a=torch.from_numpy(a)

        return head,epoch,iters,target,m,a,use_Q,server_rec,client_send,send_size
     
    
