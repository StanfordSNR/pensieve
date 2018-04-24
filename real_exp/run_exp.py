import sys
import os
import subprocess
import numpy as np


RUN_SCRIPT = 'run_video.py'
RANDOM_SEED = 42
RUN_TIME = 60  # sec
#ABR_ALGO = ['fastMPC', 'robustMPC', 'BOLA', 'RL']
ABR_ALGO = ['RL']
REPEAT_TIME = 1


def main():

	np.random.seed(RANDOM_SEED)
        print "hi hudson"
	with open('./chrome_retry_log', 'wb') as log:
		log.write('chrome retry log\n')
		log.flush()

		for rt in xrange(REPEAT_TIME):
			np.random.shuffle(ABR_ALGO)
			for abr_algo in ABR_ALGO:
                                print "abr_algo is: " + abr_algo
				while True:
                                        print "running python script"
					script = 'python ' + RUN_SCRIPT + ' ' + \
							  abr_algo + ' ' + str(RUN_TIME) + ' ' + str(rt)
					proc = subprocess.Popen(script,
							  stdout=subprocess.PIPE, 
							  stderr=subprocess.PIPE, 
							  shell=True)

					print "ran run_video.py"
					(out, err) = proc.communicate()

					if out == 'done\n':
                                                print "should be finished"
						break
					else:
						log.write(abr_algo + '_' + str(rt) + '\n')
						log.write(out + '\n')
						log.flush()



if __name__ == '__main__':
	main()
