#!/usr/bin/env python

import os
from os import path
import sys
import argparse
import signal
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm',
                        choices=['RL', 'fastMPC', 'robustMPC', 'other'])
    args = parser.parse_args()

    abr_algo = args.algorithm
    rl_server_dir = path.abspath(path.join(path.dirname(__file__),
                                           os.pardir, 'rl_server'))

    if abr_algo == 'RL':
        cmd = ['python', path.join(rl_server_dir, 'rl_server_no_training.py')]
    elif abr_algo == 'fastMPC':
        cmd = ['python', path.join(rl_server_dir, 'mpc_server.py')]
    elif abr_algo == 'robustMPC':
        cmd = ['python', path.join(rl_server_dir, 'robust_mpc_server.py')]
    else:
        cmd = ['python', path.join(rl_server_dir, 'simple_server.py')]

    subprocess.call(cmd)


if __name__ == '__main__':
    main()
