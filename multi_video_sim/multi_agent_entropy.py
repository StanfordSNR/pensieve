# Script for running the multi video training model with automatically decreasing entropy weight
import os
import glob
import subprocess
from time import sleep
import psutil


def kill(proc_pid):
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
                proc.kill()
        process.kill()


def last_model():
	model_list = glob.glob('./models/nn_model_ep_*.ckpt*')
	highest_model = 0
	for model_file in model_list:
		iteration_num = int(model_file.replace("./models/nn_model_ep_", "").split(".ckpt")[0])
		if iteration_num > highest_model:
		        highest_model = iteration_num
	return highest_model
		

def run():

	os.environ['ENTROPY_WEIGHT']='5'
	err_log = open('err_log.txt', 'a')

	while(True):
		last_itr_num = last_model()
		print "last iteration before command: " + str(last_itr_num)
                if last_itr_num == 0:
	                os.environ['last_model']='None'
                else:
		        os.environ['last_model']='nn_model_ep_' + str(last_itr_num) + '.ckpt'
		command = 'exec /usr/bin/python ./multi_agent.py'
		proc = subprocess.Popen(command, shell=True, stderr=err_log)
                sleep(3600)
		kill(proc.pid)
		last_itr_num = last_model()
                print "last iteration after command:: " + str(last_itr_num)
		os.environ['last_model']='nn_model_ep_' + str(last_itr_num) + '.ckpt'
		if (last_itr_num < 20000):
		        pass
		elif (last_itr_num < 40000):
			print "Entropy now 1"
   		        os.environ['ENTROPY_WEIGHT']='1'
		elif (last_itr_num < 80000):
			print "Entropy now 0.5"
		        os.environ['ENTROPY_WEIGHT']='0.5'
		elif (last_itr_num < 100000):
			print "Entropy now 0.3"
		        os.environ['ENTROPY_WEIGHT']='0.3'
		elif (last_itr_num < 120000):
			print "Entropy now 0.1"
		        os.environ['ENTROPY_WEIGHT']='0.1'
		else:
		        break
        print "Done Training"
	return 0


if __name__ == "__main__":
	run()
