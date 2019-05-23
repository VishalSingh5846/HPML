

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn import functional as F
import torchvision
import time
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda',action='store_true',help="Disable CUDA mode.")
parser.add_argument('--numWorkers',help="No of data loading workers.",type=int)
parser.add_argument('--optimizer',help='Specify the optimizer for the CNN')
parser.add_argument('--trainFile',help="Specify dataset path")
parser.add_argument('--dataRoot',help='Directory contianing all the images')
args = parser.parse_args()


numWorkers = args.numWorkers if args.numWorkers is not None else 1
optimizerType = args.optimizer if args.optimizer is not None else 'sgd'
curDeviceStr = "cpu" if args.disable_cuda else "cuda"
trainFile = args.trainFile if args.trainFile is not None else '/scratch/gd66/spring2019/lab2/kaggleamazon/train.csv'
dataRoot = args.dataRoot if args.dataRoot is not None else '/scratch/gd66/spring2019/lab2/kaggleamazon/train-jpg/'


print("-------------------------------------------------\nTrain File:",trainFile)
print("Data Root:",dataRoot)




miniBatchSize = 250

print("--------------------------\nDevice Specified:",curDeviceStr)
if curDeviceStr == "cuda":
	print("CUDA Available:",torch.cuda.is_available())
	if torch.cuda.is_available():
		print("Device name:",torch.cuda.get_device_name())
		torch.cuda.init()
	else:
		print("CUDA specified, but isn't available. Program will exit!")
		sys.exit()
curDevice = torch.device(curDeviceStr)

class KaggleAmazonDataset(Dataset):
	def __init__(self, trainingCSV, rootDir, transform=None):
		data = filter(lambda x: len(x)>0,open(trainingCSV).read().split('\n'))
		data = list(map(lambda x: x.split(','),data))[1:]
		if len(rootDir)>0 and rootDir[-1] != '/':
			rootDir+='/'
		newData = []
		for i in data:
			#print("data="+str(i)+",Len="+str(len(i)))
			filename = i[0]
			labels = i[1]
			temp = [0 for i in range(17)]
			for label in labels.split(" "):
				temp[int(label)] = 1
			newData += [(rootDir+filename+".jpg",temp)]
			self.trainData = newData 
			self.rootDir = rootDir
			self.transform = transform

	def __len__(self):
		return len(self.trainData)

	def __getitem__(self, idx):
		#sample = io.imread(self.trainData[idx][0])
		sample = Image.open(self.trainData[idx][0]).convert('RGB')
		if self.transform:
			sample = self.transform(sample)
		return sample,torch.tensor(np.array(self.trainData[idx][1])).float()



class InceptionModule(torch.nn.Module):   
	def __init__(self,inCh,outCh):
		super(InceptionModule, self).__init__()
		self.Layer1 = torch.nn.Conv2d(in_channels = inCh, out_channels = outCh,kernel_size=1,stride=1,padding=0)
		self.Layer2 = torch.nn.Conv2d(in_channels = inCh, out_channels = outCh,kernel_size=3,stride=1,padding=1)
		self.Layer3 = torch.nn.Conv2d(in_channels = inCh, out_channels = outCh,kernel_size=5,stride=1,padding=2)
	
	def forward(self, x):
		#print("X ->"+str(x.shape))
		x1 = self.Layer1(x)
		x2 = self.Layer2(x)
		x3 = self.Layer3(x)
		x = torch.cat((x1,x2,x3), 1)
		#print("X1 ->"+str(x1.shape))
		#print("X2 ->"+str(x2.shape))
		#print("X3 ->"+str(x3.shape))
		#print("X ->"+str(x.shape))
		return(x)
#

class CNN(torch.nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.Layer1 = InceptionModule(3,10)
		self.Layer2 = InceptionModule(30,10)
		self.Layer3 = torch.nn.Linear(8*8*30,256)
		self.Layer4 = torch.nn.Linear(256,17)
	def forward(self,x):
		#print("MAIN X->"+str(x.shape))
		x = F.relu(F.max_pool2d(kernel_size=2,input=self.Layer1(x)))
		
		#print("MAIN X1->"+str(x.shape))
		x = F.relu(F.max_pool2d(kernel_size=2,input=self.Layer2(x)))
		#print("MAIN X2->"+str(x.shape))
		#print("X.view:",x.view(30*8*8,-1).shape,x.size(0))
		x = F.relu(self.Layer3(x.view(x.size(0), -1)))
		#print("MAIN X3->"+str(x.shape))
		x = torch.sigmoid(self.Layer4(x))
		#print("MAIN X4->"+str(x.shape))
		return x






def getOptimizer(optimizerType,model,learningRate,moment):
	if optimizerType == 'sgd':
		return torch.optim.SGD(model.parameters(), lr=learningRate,momentum = moment)
	elif optimizerType == 'sgd_nesterov':
		return torch.optim.SGD(model.parameters(), lr=learningRate,momentum = moment,nesterov = True)
	elif optimizerType == 'adagrad':
		return torch.optim.Adagrad(model.parameters(), lr=learningRate)
	elif optimizerType == 'adadelta':
		return torch.optim.Adadelta(model.parameters(),lr=learningRate)
	elif optimizerType == 'adam':
		return torch.optim.Adam(model.parameters(),lr=learningRate)
	
	print("Unsupported optimizer:",optimizerType)
	print("Supported optimizers: sgd,sgd_nesterov,adagrad,adadelta,adam")
	sys.exit()
	



def trainNetwork(epochs,curDevice,optimizerType,loaderCount):
	print("Optimizer:",optimizerType)
	print("Loaders:",loaderCount)
	print("-- TRAINING STARTING --")
	step = 0
	global trainFile
	global dataRoot
	kaggleDataset = KaggleAmazonDataset(trainingCSV=trainFile, rootDir=dataRoot, transform=transforms.Compose([ transforms.Resize(32), transforms.ToTensor()]))
	
	dataLoader = torch.utils.data.DataLoader(kaggleDataset, batch_size=miniBatchSize, num_workers=loaderCount,pin_memory=True,drop_last=True)
	model = CNN().to(device=curDevice)
	moment = 0.9
	learningRate = 0.01
	lossFunc = torch.nn.BCELoss()
	

	tensorJust0 = torch.tensor([0]).to(device=curDevice)
	tensorJust1 = torch.tensor([1]).to(device=curDevice)
	tensorJust2 = torch.tensor([2]).to(device=curDevice)
	optimizer = getOptimizer(optimizerType,model,learningRate,moment)

	zeroTensor = torch.tensor([0 for i in range(miniBatchSize)]).to(device=curDevice)

	cumEpochTime = 0
	cumMiniBatchPrecisionTime = 0
	cumMiniBatchForwardBackwardTime = 0
	cumLoadingTime = 0
	def getPrecision(Label,top,topInd,ind):
		#print("topInd:",topInd[0][0],top[0].data,Label[topInd[0]])
		cor = 0
		for i in range(len(Label)):
			if Label[i][topInd[i][ind]]==top[i][ind] and int(top[i][ind])==1:
				cor += 1
		return cor


	LossList = []
	Precision1List = []
	Precision3List = []
	for epoch in range(epochs):
		cumLoss = 0
		precision1Store = torch.tensor(0,device=curDevice).float()
		precision3Store = torch.tensor(0,device=curDevice).float()
		print("\tEPOCH->",epoch)
		startEpoch = time.monotonic()
		miniBatchForwardBackwardTime = cumMiniBatchForwardBackwardTime
		miniBatchPrecisionTime = cumMiniBatchPrecisionTime
		for x,(data,label) in enumerate(dataLoader):			
		
			startMB = time.monotonic()
		
			label = label.to(curDevice)
			data = data.to(curDevice)
		
			predLabel = model(data)
			loss = lossFunc(predLabel,label)
			step += 1
			optimizer.zero_grad()
			loss.backward()
			with torch.no_grad():
				for param in model.parameters():
					param -= learningRate * param.grad
			endMB = time.monotonic()
			cumMiniBatchForwardBackwardTime += endMB - startMB
			
			precisionStart = time.monotonic()
			#predLabel = predLabel.round()
		
			#top1,top1Ind = predLabel.topk(1,dim=1)
			#top1 = torch.round(top1)
			top3,top3Ind = predLabel.topk(3,dim=1)
			#top3 = torch.round(top3)

			#top1Orig = torch.gather(label,dim=1,index=top1Ind)
			#eq1 = torch.eq(top1Orig,top1)
			#precision1_LAB = eq1.sum()

		
		
			#cor = getPrecision(label,top1,top1Ind,0)
			#print("\t\tPrecision@1 Naive:",cor/250)	
			#cor = 0
			#ttt =getPrecision(label,top3,top3Ind,0)
			#print("\t\tPrecision@1 ALag:",ttt/250)
			#ttt = getPrecision(label,top3,top3Ind,0) + getPrecision(label,top3,top3Ind,1) + getPrecision(label,top3,top3Ind,2)
			#print("\t\tPrecision@3 Naive:",ttt/750)
			#for i in range(len(label)):
			
			
			top3Orig0 = torch.gather(label,dim=1,index=torch.index_select(top3Ind,dim=1,index=tensorJust0)).squeeze()
			top3Orig1 = torch.gather(label,dim=1,index=torch.index_select(top3Ind,dim=1,index=tensorJust1)).squeeze()
			top3Orig2 = torch.gather(label,dim=1,index=torch.index_select(top3Ind,dim=1,index=tensorJust2)).squeeze()
			
			#print("Top3Orig0:",top3Orig0)
			#print("Top3Orig1:",top3Orig1)
			#print("Top3Orig2:",top3Orig2)	
	
			#print("Top3",top3)
			#print("label:",list(label))
			#print("top3:",top3)
			#print("top3Ind:",top3Ind)
		
			top3Orig = torch.stack([top3Orig0,top3Orig1,top3Orig2],1)
			#precision1_STANDARD = torch.dot(top1.squeeze(),top3Orig0.squeeze())*100.0/miniBatchSize
			precision1_STANDARD = top3Orig0.sum()/miniBatchSize
			precision1Store += precision1_STANDARD
			#precision3_STANDARD = torch.dot(top3.view(-1),top3Orig.view(-1).squeeze())*100.0/(3*miniBatchSize)	
			precision3_STANDARD = top3Orig.sum()/(3*miniBatchSize)
			precision3Store += precision3_STANDARD
			cumLoss += loss.item()
			precisionEnd = time.monotonic()

			cumMiniBatchPrecisionTime += precisionEnd-precisionStart		
			#print("\t\tStep:",step,", Loss:",loss.item())
			#print("\t\tPrecision@1:",precision1_STANDARD)
			#print("\t\tPrecision@3:",precision3_STANDARD)
			#print("\t\tStep time:",endMB-startMB)
			#print("\t\t--------")
	
		
		endEpoch = time.monotonic()
		LossList += [cumLoss/120]
		Precision1List += [precision1Store.item()/120]
		Precision3List += [precision3Store.item()/120]
		
		loadingTime = (endEpoch-startEpoch) - (cumMiniBatchPrecisionTime + cumMiniBatchForwardBackwardTime - miniBatchPrecisionTime - miniBatchForwardBackwardTime)
		cumLoadingTime += loadingTime 
		cumEpochTime += (endEpoch-startEpoch)
		print("\tLoading Time:",loadingTime)
		print("\tEpoch Time: "+str(endEpoch-startEpoch),"\n\t----------")
	print("-- TRAINING COMPLETE --")
	return [(cumEpochTime,cumLoadingTime,cumMiniBatchForwardBackwardTime,cumMiniBatchPrecisionTime),(Precision1List,Precision3List),LossList]


def exerciseC2(curDevice):
	global optimizerType	
	print("-------------- C2 ----------------")
	res = trainNetwork(5,curDevice,optimizerType,1)
	print("Aggregated Data Loading Time:",res[0][1])
	print("Aggregated Mini Batch (Loading+NN Forward/Backward)",res[0][1]+res[0][2])
	print("Aggregated Epoch Time:",res[0][0])
	print("\n")

def exerciseC3(curDevice):
	global optimizerType
	print("-------------- C3 ----------------")
	optimal = None
	optTime = None
	for i in range(1,10):
		workerC = i*4
		res = trainNetwork(5,curDevice,optimizerType,workerC)
		curTime = res[0][0]/5	
		if optTime is None or optTime>curTime:
			optTime = curTime
			optimal = workerC
		
		print("Workers:",workerC,", Average Epoch Time:",curTime,",Average Loading Time:",res[0][1]/5)
	print("Optimal Loader Count -> ",optimal)
	print("\n")
def exerciseC4(curDevice,workerCount):
	global optimizerType
	print("-------------- C4 ----------------")
	res = trainNetwork(5,curDevice,optimizerType,workerCount)
	
def exerciseC5(curDevice,workerC):
	
	print("-------------- C5 ----------------")
	optim = ['sgd','sgd_nesterov','adagrad','adadelta','adam']
	for o in optim:
		res = trainNetwork(5,curDevice,o,workerC)
		print("Optimizer:",o,"\nAvg. Epoch Time:",res[0][0]/5,",\nLoss Average:",res[2],"\nPrecision@1:",res[1][0],"\nPrecision@3:",res[1][1],"\n")


def fullSummary(curDevice,workerC):
	global optimizerType
	print("\n------------------------- FULL SUMMARY -------------------------")
	res = trainNetwork(5,curDevice,optimizerType,workerC)
	print("Total Time:",res[0][0])
	print("\tLoading Time:",res[0][1])
	print("\tComputation Time:",res[0][2]+res[0][3])
	print("\t\tNN Forward Backward Time:",res[0][2])
	print("\t\tPrecision Calculation:",res[0][3])
	
	print("\nAverage Loss:",res[2])
	print("Precision@1:",res[1][0])
	print("Precision@3:",res[1][1])
	print("------------------------------\n\n")



#fullSummary(curDevice,numWorkers)
exerciseC2(curDevice)
exerciseC3(curDevice)
#exerciseC4(curDevice,numWorkers)
#exerciseC5(curDevice,numWorkers)




