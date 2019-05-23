import pandas as pd
import os
from PIL import Image
import time
import csv
import random
import pandas as pd
import torch.distributed as dist
import numpy as np
import sys
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch import multiprocessing


nSteps = 1
miniBatchSize = 250
epochCount = 5
VERBOSE = False
#dist.init_process_group(init_method='file:///home/vs2202/Lab4/kuchMPI',backend='mpi')
#init_processes(0, 0, run, backend='mpi')
class KaggleAmazonDataset(Dataset):
    def __init__(self, csv_path, img_path, img_ext, transform=None):
        tmp_df = pd.read_csv(csv_path)
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform
        self.X_train = tmp_df['image_name']
        self.y_train = tmp_df['tags']
        self.num_labels = 17

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label_ids = self.y_train[index].split()
        label_ids = [ int(s) for s in label_ids ]
        label=torch.zeros(self.num_labels)
        label[label_ids] = 1
        return img, label

    def __len__(self):
        return len(self.X_train.index)


class Inception_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception_Module, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(x)
        out = [conv1x1, conv3x3, conv5x5]
        out = torch.cat(out, 1)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module1 = Inception_Module(3, 10)
        self.module2 = Inception_Module(30, 10)
        self.fc1 = nn.Linear(1920, 256)
        self.fc2 = nn.Linear(256, 17)

    def forward(self, x):
        x=self.module1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x=self.module2(x)
        x = F.relu(F.max_pool2d(x , 2))
        x = x.view(x.size(0), -1) # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def getParametersFromServer(model,broadcast=False):
	global VERBOSE
	if VERBOSE:
		print("Worker %d trying to fetch model" % dist.get_rank())

	for param in model.parameters():
		if broadcast:
			dist.broadcast(param.data,src=0)
		else:
			dist.recv(param.data,src=0,tag=0)

	#print("Worker %d, recieved model" % dist.get_rank())

def printModelParameters(model):
	count = 0
	for i in model.parameters():
		print("Parameter",count," -> ",i)
		count+=1
		break

def train(epoch, train_loader, model, criterion, optimizer):
	global VERBOSE
	if VERBOSE:
		print("Training in worker",dist.get_rank(),"starting")
	loader_times = AverageMeter()
	batch_times = AverageMeter()
	losses = AverageMeter()
	precisions_1 = AverageMeter()
	precisions_k = AverageMeter()
	model.train()
	sampleCount = 0
	t_train = time.monotonic()
	t_batch = time.monotonic()
	cumLoss = None
	global nSteps
	nStepsCur = 0
		
	getParametersFromServer(model,broadcast=True)
	
	accruedGradients = [torch.zeros(i.data.size()) for i in model.parameters()]
	for batch_idx, (data, target) in enumerate(train_loader):
		sampleCount += len(target)
		loader_time = time.monotonic() - t_batch
		loader_times.update(loader_time)
		data = data.to(device=device)
		target = target.to(device=device)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		nStepsCur += 1
		#print("%d Step->%d , %d" % (dist.get_rank(),nStepsCur,nSteps))
		if VERBOSE:
			print("Worker %d, loss:%f" % (dist.get_rank(),loss.item()))
		if nStepsCur == nSteps:
			#print("Pushing Parameters",dist.get_rank())
			nStepsCur = 0
			for gradDataInd in range(len(accruedGradients)):
				#print("GRADIENT SIZE-------------->",param.grad.data.size(),param.data.size())
				dist.send(accruedGradients[gradDataInd],dst=0,tag=0)
				accruedGradients[gradDataInd] = torch.zeros(accruedGradients[gradDataInd].size())
			
			#printModelParameters(model)	
			getParametersFromServer(model)
			#print("@@@@@ AFTER FETCH")
			#printModelParameters(model)
			#sys.exit()	
		else:
			counter  = 0
			
			for param in model.parameters():
				accruedGradients[counter] += param.grad
				counter += 1

		batch_times.update(time.monotonic() - t_batch)
        
		topk=3
		_, predicted = output.topk(topk, 1, True, True)
		batch_size = target.size(0)
		prec_k=0
		prec_1=0
		count_k=0
		for i in range(batch_size):
			prec_k += target[i][predicted[i]].sum()
			prec_1 += target[i][predicted[i][0]]
			count_k+=topk #min(target[i].sum(), topk)
		prec_k/=count_k
		prec_1/=batch_size
 
		losses.update(loss.item(), 1)
		precisions_1.update(prec_1, 1)
		precisions_k.update(prec_k, 1)
        

		#if (batch_idx+1) % 500 == 0:
			#print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f} ({:.3f}),\tPrec@1: {:.3f} ({:.3f}),\tPrec@3: {:.3f} ({:.3f}),\tTimes: Batch: {:.4f} ({:.4f}),\tDataLoader: {:.4f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), losses.val, losses.avg, precisions_1.val, precisions_1.avg , precisions_k.val, precisions_k.avg , batch_times.val, batch_times.avg, loader_times.avg))

		t_batch = time.monotonic()
	
	train_time = time.monotonic() - t_train
	
	#dist.send(torch.tensor(1),dst=0,tag=1)
	print('Rank: %.2d, Epoch: %d, Loss: %.3f, Prec@1: %.3f, Prec@3: %.3f, Time: %.2f' % (dist.get_rank(), epoch, losses.avg, precisions_1.avg, precisions_k.avg, train_time))
	#print("Worker:",dist.get_rank(),'Training Epoch: {} done. \tLoss: {:.3f},\tPrec@1: {:.3f},\tPrec@3: {:.3f}\tTimes: Total: {:.3f}, Avg-Batch: {:.4f}, Avg-Loader: {:.4f}\n'.format(epoch, losses.avg, precisions_1.avg, precisions_k.avg, train_time, batch_times.avg, loader_times.avg))
	return  (train_time, batch_times.avg, loader_times.avg) , (losses.avg,precisions_1.avg,precisions_k.avg,sampleCount)


class Partition(object):
	def __init__(self, data, start , end):
		self.data = data
		self.start = start
		self.end = end
	def __len__(self):
		return self.end - self.start
	def __getitem__(self, index):
		return self.data[self.start+index]


class DataPartitioner(object):
	def __init__(self, data, partitionCount, seed=1234):
		self.data = data
		self.partitions = []
		self.partitionCount = partitionCount
		#rng = Random()
		#rng.seed(seed)
		data_len = len(data)
		elementPerPartition = data_len // partitionCount
		#rng.shuffle(indexes)
		for i in range(partitionCount):
			self.partitions.append((i*elementPerPartition,(i+1)*elementPerPartition))

	def use(self, partition):
		start,end = self.partitions[partition]
		print("Partition: %d, Total Length:%d, [%d,%d)" % (partition,len(self.data),start,end))
		return Partition(self.data,start,end)

def parameterServer(network,optimizer,workGroup,epochCount,totalDataSize,miniBatchSize,nSteps):
	
	def sendParameters(network,rank,broadcast = False,workGroup=workGroup):
		global VERBOSE
		if VERBOSE:
			print("Server -> Worker",rank)
		for param in network.parameters():
			if broadcast:
				dist.broadcast(param.data,src=dist.get_rank())
			else:
				dist.send(param.data,dst=rank,tag=0)
		
	global VERBOSE
	if VERBOSE:
		print("----------------- PARAMETER SERVER STARTING ----------------")
	totalWorkers = dist.get_world_size() - 1
	
	firstReceive = True
	workerSending = None
	
	for param in network.parameters():
		param.grad = torch.zeros(param.data.size())	

	transactionsPerEpoch = ( ( (totalDataSize// totalWorkers ) // miniBatchSize ) //nSteps) * totalWorkers  
	if transactionsPerEpoch==0:
		print("ERROR: With current minibatch size, total workers and nsteps, there will be no update to the parameter server")
		#sys.exit()

	for _ in range(epochCount):
		sendParameters(network,-1,broadcast=True)
		for z in range(transactionsPerEpoch):
			if VERBOSE:
				print("\nTransaction %d/%d" % (z+1,transactionsPerEpoch))

			for param in network.parameters():
				#print("HERE 333")
				if firstReceive:
					workerSending = dist.recv(param.grad,tag=0)
					firstReceive = False
					if VERBOSE:
						print("Worker",workerSending," -> Server")
				else:
					dist.recv(param.grad.data,src=workerSending,tag=0)
			
			
			firstReceive = True
			optimizer.step()
			sendParameters(network,workerSending)		
		
	if VERBOSE:
		print("-------------------- PARAMETER SERVER EXITING -----------------")




if __name__ == '__main__':
	dist.init_process_group(backend='mpi')
	rank = dist.get_rank()
	wsize = dist.get_world_size()
	numberOfNodeToRunTraining = wsize - 1
	workerGroup = dist.new_group(ranks=[i for i in range(1,wsize)])
	if VERBOSE:
        	print(" +++++ WORLD SIZE:",dist.get_world_size(),rank,"++++++")
		
	parser = argparse.ArgumentParser(description='PyTorch Example')
	parser.add_argument('nSteps',type=int)
	parser.add_argument('--disable_cuda', action='store_true',help='Disable CUDA')
	parser.add_argument('--workers', type=int, default=0,help='Number of dataloader workers')
	parser.add_argument('--data_path', type=str, default='/scratch/gd66/spring2019/lab4/kaggleamazon/')
	parser.add_argument('--opt', type=str, default='adam',help='Examples: adam, rmsprop, adadelta, ...)')
	args = parser.parse_args()
	nSteps = args.nSteps
        
	device = None
	if not args.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	
	model = Net().to(device=device)
	if args.opt=='adam':
		optimizer = optim.Adam(model.parameters())
	elif args.opt=='adagrad':
		optimizer = optim.Adagrad(model.parameters())
	elif args.opt=='adadelta':
		optimizer = optim.Adadelta(model.parameters())
	elif args.opt=='nesterov':
		optimizer = optim.SGD(model.parameters(),nesterov=True)
	else:
		optimizer = optim.SGD(model.parameters(),nesterov=False)
	
	

	DATA_PATH=args.data_path
	IMG_PATH = DATA_PATH+'train-jpg/'
	IMG_EXT = '.jpg'
	TRAIN_DATA = DATA_PATH+'train.csv'
		
	transformations = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
	dset_train = KaggleAmazonDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)
		
	if dist.get_rank()==0:
		print("NSteps: %d, Data Loaders: %d, Optimizer: %s, World Size: %d" % (nSteps,args.workers,args.opt,dist.get_world_size()))
		parameterServer(model,optimizer,workerGroup,epochCount,len(dset_train),miniBatchSize,nSteps)
	else:
		try:
			multiprocessing.set_start_method('spawn')
		except RuntimeError:
			pass

		totalStart = time.monotonic()
		partitionObj = DataPartitioner(dset_train,numberOfNodeToRunTraining)
		partitionedData = partitionObj.use(dist.get_rank()-1)
		train_loader = DataLoader(partitionedData,batch_size= miniBatchSize,shuffle=True,drop_last=True,num_workers=args.workers)
		
		
		
		criterion = nn.BCELoss().to(device=device)
		train_times=[]
		batch_times=[]
		loader_times=[]
		
		tnsr = None
		timeExcludingBarrier = 0
		for epoch in range(epochCount):
			timer1 = time.monotonic()
			timeInfo , performance = train(epoch, train_loader, model, criterion, optimizer)
			train_time, batch_time, loader_time = timeInfo
			train_times.append(train_time)
			batch_times.append(batch_time)
			loader_times.append(loader_time)
			timeExcludingBarrier += time.monotonic()-timer1
			dist.barrier(group=workerGroup)
	
			if epoch==epochCount-1:
				loss ,precision1,precisionk,count = performance
				tnsr = torch.tensor([loss*count,precision1*count,precisionk*count,count,0])
				if VERBOSE or dist.get_rank()==-1:
					if tnsr[3]==0:
						print(" ********************** ERROR COUNT 0 *******************")
					else:
						print("\n++++++++++++++++++  AGGREGATED INFO +++++++++++++++++++++") 
						print("Rank:%d, Epoch:%d, Loss: %f, Precision@1: %f, Precision@3: %f" % (dist.get_rank(),epoch,tnsr[0]/tnsr[3],tnsr[1]/tnsr[3],tnsr[2]/tnsr[3]))
						print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++") 
			if VERBOSE:
				print("#################   BARRIER:",dist.get_rank()," #################")
			
		#dist.send(torch.tensor(1),dst=0,tag=1)
		
		#totalEnd = time.monotonic()
		dist.all_reduce(tnsr,op=dist.ReduceOp.SUM,group=workerGroup)
		maxTime = torch.tensor(time.monotonic() - totalStart)
		#idist.all_reduce(maxTime,op=dist.ReduceOp.MAX,group=workerGroup)
		
		print("[AGGREGATED] Rank: %.2d, Loss: %.4f, Prec@1: %.3f, Prec@3: %.3f, Total Time: %.2f sec" % ( dist.get_rank(), tnsr[0]/tnsr[3] , tnsr[1]/tnsr[3] , tnsr[2]/tnsr[3], maxTime.item() ) )
		#print("Worker:",dist.get_rank(),'Final Average Times: Total: {:.3f}, Avg-Batch: {:.4f}, Avg-Loader: {:.4f}\n'.format(np.average(train_times), np.average(batch_times), np.average(loader_times)))
