import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pdb
import argparse
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


# pdb.set_trace()
parser = argparse.ArgumentParser()
parser.add_argument('--Q', type=str, default='',
					help='Question number.')
parser.add_argument('--res_path', type=str, default='final_res',
					help='Directory of final trained models.')
args = parser.parse_args()
res_path = args.res_path

def filter_paths(keeps=[""]):
	files = os.listdir(res_path)
	paths=[]
	for name in files:
		if name==".DS_Store":continue
		arch = name.split("_")[0]
		optim = name.split("model")[0].split(arch)[1][1:-1]
		for keep in keeps:
			if keep==arch[:-1] or keep==optim:
				paths.append([name,arch,optim])
	return paths


if args.Q == '4.1':
	paths = filter_paths(['RNN', 'GRU', 'TRANSFORMER'])
	for path in paths:
		curves = np.load("{}/{}/learning_curves.npy".format(res_path,path[0])).item()
		train_ppls,val_ppls = curves['train_ppls'],curves['val_ppls']

		fig,ax = plt.subplots(1)
		plt.grid()
		ax.set_title("{} ({}) Learning Curve per Epochs".format(path[1],path[2]))
		ax.plot(train_ppls[1:],label="Train PPL") #Start from 1 as the first PPL is too high
		ax.plot(val_ppls[1:],label="Validation PPL") #Start from 1 as the first PPL is too high
		ax.legend()
		ax.set_xlabel("Epochs")
		ax.set_ylabel("PPL")	
		plt.savefig("plots/4.1/{}_{}.jpg".format(path[1],path[2]))
		plt.close()

		with open('{}/{}/log.txt'.format(res_path,path[0])) as f:
		    reader = csv.reader(f, delimiter="\t")
		    log = list(reader)
		x = np.cumsum(np.asarray([float(row[4][len('time (s) spent in epoch: '):]) for row in log]))[1:]
		fig,ax = plt.subplots(1)
		plt.grid()
		ax.set_title("{} ({}) Learning Curve per wall-clock-time".format(path[1],path[2]))
		ax.plot(x,train_ppls[1:],label="Train PPL") #Start from 1 as the first PPL is too high
		ax.plot(x,val_ppls[1:],label="Validation PPL") #Start from 1 as the first PPL is too high
		ax.legend()
		ax.set_xlabel("Time (seconds)")
		ax.set_ylabel("PPL")
		plt.savefig("plots/4.1/{}_{}_seconds.jpg".format(path[1],path[2]))
		plt.close()


elif args.Q == '4.4':
	optimizers = ['SGD', 'SGD_LR_SCHEDULE', 'ADAM']
	for optim in optimizers:
		paths = filter_paths([optim])
		fig,ax = plt.subplots(1)
		plt.grid()
		ax.set_title("Models trained with {}".format(optim))
		for path in paths:
			curves = np.load("{}/{}/learning_curves.npy".format(res_path,path[0])).item()
			train_ppls,val_ppls = curves['train_ppls'],curves['val_ppls']
			ax.plot(val_ppls[1:],label=path[1]) #Start from 1 as the first PPL is too high
		ax.legend()
		ax.set_xlabel("Epochs")
		ax.set_ylabel("Validation PPL")
		plt.savefig("plots/4.4/{}_comparison_model.jpg".format(path[2]))
		plt.close()	

		fig,ax = plt.subplots(1)
		plt.grid()
		ax.set_title("Models trained with {}".format(optim))
		for path in paths:
			curves = np.load("{}/{}/learning_curves.npy".format(res_path,path[0])).item()
			train_ppls,val_ppls = curves['train_ppls'],curves['val_ppls']
			with open('{}/{}/log.txt'.format(res_path,path[0])) as f:
			    reader = csv.reader(f, delimiter="\t")
			    log = list(reader)
			x = np.cumsum(np.asarray([float(row[4][len('time (s) spent in epoch: '):]) for row in log]))[1:]
			ax.plot(x,val_ppls[1:],label=path[1]) #Start from 1 as the first PPL is too high
		ax.legend()
		ax.set_xlabel("Time (seconds)")
		ax.set_ylabel("Validation PPL")	
		plt.savefig("plots/4.4/{}_comparison_model_seconds.jpg".format(path[2]))
		plt.close()


elif args.Q == '4.5':
	models = ['RNN', 'GRU', 'TRANSFORMER']
	for arch in models:
		paths = filter_paths([arch])
		fig,ax = plt.subplots(1)
		plt.grid()
		ax.set_title("Optimizers comparison for {}".format(arch))
		for path in paths:
			curves = np.load("{}/{}/learning_curves.npy".format(res_path,path[0])).item()
			train_ppls,val_ppls = curves['train_ppls'],curves['val_ppls']
			ax.plot(val_ppls[1:],label=path[2]+' ({})'.format(path[1])) #Start from 1 as the first PPL is too high
		ax.legend()
		ax.set_xlabel("Epochs")
		ax.set_ylabel("Validation PPL")
		plt.savefig("plots/4.5/{}_comparison_optim.jpg".format(arch))
		plt.close()	

		fig,ax = plt.subplots(1)
		plt.grid()
		ax.set_title("Optimizers comparison for {}".format(arch))
		for path in paths:
			curves = np.load("{}/{}/learning_curves.npy".format(res_path,path[0])).item()
			train_ppls,val_ppls = curves['train_ppls'],curves['val_ppls']
			with open('{}/{}/log.txt'.format(res_path,path[0])) as f:
			    reader = csv.reader(f, delimiter="\t")
			    log = list(reader)
			x = np.cumsum(np.asarray([float(row[4][len('time (s) spent in epoch: '):]) for row in log]))[1:]
			ax.plot(x,val_ppls[1:],label=path[2]+' ({})'.format(path[1])) #Start from 1 as the first PPL is too high
		ax.legend()
		ax.set_xlabel("Time (seconds)")
		ax.set_ylabel("Validation PPL")	
		plt.savefig("plots/4.5/{}_comparison_optim_seconds.jpg".format(arch))
		plt.close()


elif args.Q == '5.1':
	models = ['RNN', 'GRU', 'TRANSFORMER']
	for arch in models:
		path = "numpy_files/losst/{}_losst.npy".format(arch)
		losst = np.load(path)
		fig, ax = plt.subplots(1)
		plt.grid()
		ax.set_title("Comparison over architectures for loss_t over timesteps")
		model_num = 3 if arch is 'RNN' else 2
		ax.plot(losst,label=arch+str(model_num))
		ax.set_xlabel("Timestep")
		ax.set_ylabel("Loss_t")
		ax.legend()
		ax.plot()
		plt.savefig("plots/5.1/{}.jpg".format(arch))


elif args.Q == '5.2':
	rcParams.update({'figure.autolayout': True})
	models = ['RNN', 'GRU']
	fig, ax = plt.subplots(1)
	plt.grid()
	for arch in models:
		path = "numpy_files/hgrads/{}_hgrads.npy".format(arch)
		hgrads = np.load(path)
		lists = list(hgrads.item().items())
		x, y = zip(*lists)
		x = np.asarray([int(label[1:]) for label in x])
		y = np.asarray(y)
		y /= np.max(y)
		model_num = 3 if arch is 'RNN' else 2
		ax.plot(x, y ,label=arch+str(model_num))
	plt.tight_layout()
	ax.set_title("(5.2) Gradient of final loss wrt different hidden layer timesteps")
	ax.set_xlabel("Timestep of hidden layer")
	ax.set_ylabel("Gradient of final loss w.r.t. different hidden layer timesteps")
	ax.legend()
	ax.plot()
	plt.savefig("plots/5.2/RNN_GRU_hgrads.jpg")