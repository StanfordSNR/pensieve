#!/usr/bin/env python2

import os
import json
import sys
import socket
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import tensorflow as tf
import a3c
import struct

S_INFO = 7  # bit_rate, buffer_size, bandwidth_measurement, measurement_time, chunk_til_video_end
S_LEN = 10  # take how many frames in the past
A_DIM = 10
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [200,300,450,750,1200,1850,2850,4300,6000,8000]  # Kbps
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
M_IN_B = 1000000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
NN_MODEL = sys.argv[1]
SERVER_ADDRESS = sys.argv[2]
AVG_AUDIO_SIZE_BYTES = 8000; # Approx 64 Kb of audio sent every 2 seconds, add this to video chunks


def start_ipc_client():
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(SERVER_ADDRESS)
    return sock


def get_puffer_info(sock):
    json_len = sock.recv(2, socket.MSG_WAITALL)
    try:
        json_len_num = struct.unpack("!H", json_len)[0]
    except Exception:
        print "Failed to decode info from Puffer over IPC"
        sys.exit(1)
    json_data = sock.recv(json_len_num, socket.MSG_WAITALL)
    puffer_info = json.loads(json_data)
    delay = puffer_info['delay'];
    playback_buf = puffer_info['playback_buf'];
    rebuf_time = puffer_info['rebuf_time'];
    last_chunk_size = puffer_info['last_chunk_size'];
    next_chunk_size = puffer_info['next_chunk_sizes'];

    return delay, playback_buf, rebuf_time, last_chunk_size, next_chunk_size


def send_puffer_next_action(sock, bit_rate):
    bit_rate_dict = {}
    bit_rate_dict['bit_rate'] = bit_rate
    bit_rate_json = json.dumps(bit_rate_dict)
    json_len = struct.pack("!H", len(bit_rate_json))
    err = sock.sendall(json_len + bit_rate_json)
    return err


def bitrate_to_action(bitrate, mask, a_dim=A_DIM):
    assert len(mask) == a_dim
    assert bitrate >= 0
    assert bitrate < np.sum(mask)
    cumsum_mask = np.cumsum(mask) - 1
    action = np.where(cumsum_mask == bitrate)[0][0]
    return action


def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    # Originally defined in env.py
    mask = [1,1,1,1,1,1,1,1,1,1]
    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action = bitrate_to_action(bit_rate, mask)
        last_action = action

        s_batch = [np.zeros((S_INFO, S_LEN))]

        entropy_record = []

        video_chunks_sent = 0;
        video_num_chunks = 43200; # 24 hours of video. Is this an acceptable proxy for never ending video?

        puffer_sock = start_ipc_client()

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real

            video_chunk_remain = video_num_chunks - video_chunks_sent;

            delay, buffer_size, \
                rebuf, video_chunk_size, \
                next_video_chunk_size = \
                get_puffer_info(puffer_sock)

            reward = VIDEO_BIT_RATE[action] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[action] -
                                               VIDEO_BIT_RATE[last_action]) / M_IN_K

            last_bit_rate = bit_rate
            last_action = action

            # Add average audio size to each video chunk to improve throughput estimates
            # This is necessary because original Pensieve code does not consider audio, and
            # no simple solution exists given that our audio and video chunks are different
            # time scales.
            video_chunk_size += AVG_AUDIO_SIZE_BYTES
            for idx in xrange(len(next_video_chunk_size)):
                next_video_chunk_size[idx] = next_video_chunk_size[idx] + AVG_AUDIO_SIZE_BYTES

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            if delay == 0: #No division by zero
                delay = 1

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[action] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms # This is really just throughput
            state[3, -1] = float(delay) / M_IN_K
            state[4, -1] = video_chunk_remain / float(video_num_chunks)
            state[5, :] = -1
            nxt_chnk_cnt = 0
            for i in xrange(A_DIM):
                if mask[i] == 1:
                    state[5, i] = next_video_chunk_size[nxt_chnk_cnt] / M_IN_B
                    nxt_chnk_cnt += 1
            assert(nxt_chnk_cnt) == np.sum(mask)
            state[6, -A_DIM:] = mask

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))

            # the action probability should correspond to number of bit rates
            assert len(action_prob[0]) == np.sum(mask)

            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states
            action = bitrate_to_action(bit_rate, mask)

            # Now I have my action! Send this action back to the Puffer server over IPC
            send_puffer_next_action(puffer_sock, bit_rate)

            s_batch.append(state)

            entropy_record.append(a3c.compute_entropy(action_prob[0]))


if __name__ == '__main__':
    main()
