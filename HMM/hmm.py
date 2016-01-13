#!/usr/bin/env python
# coding:utf-8

import numpy as np
import sys

class HMM_SEG(object):
	"""
	The class is for segmenting Chinese text using hidden markov model(HMM). 

	train data: icwb2-data/training/msr_training.utf8
	"""

	def __init__(self, dataFile):
		'''
		:param dataFile: string
						The filename of the train data.

		lamda = (A, B, Pi)
		A: the transer matrix, 4x4
		B: the emission matrix, 4x65535 
		Pi: the initial probability of the hidden state.
		'''

		self.A = np.zeros((4, 4), dtype=np.double)
		self.B = np.zeros((4, 65535), dtype=np.double)
		self.Pi = np.zeros((4, 1), dtype=np.double)

		self.trainData(dataFile)


	def trainData(self, dataFile):
		'''
		Compute lamda=(A, B, Pi) according the data.
		:param dataFile: string
						The file name of the train data.
		'''

		I_sequences = []
		O_sequences = []
		T_sequences = []

		with open(dataFile, 'r') as f:
			line = f.readline().strip()
			while line:
				T_sequences.append(line.decode('utf-8').split())
				line = f.readline().strip()

		for sequence in T_sequences:
			t_iseq = []
			t_oseq = []
			for word in sequence:
				leng = len(word)
				if leng == 1:
					t_iseq.append(3)
				elif leng == 2:
					t_iseq.append(0)
					t_iseq.append(2)
				else:
					t_iseq.append(0)
					for i in range(leng-2):
						t_iseq.append(1)
					t_iseq.append(2)

				for i in range(leng):
					t_oseq.append(word[i])

			I_sequences.append(t_iseq)	
			O_sequences.append(t_oseq)

		self.calTransferMatrix(I_sequences)
		self.calEmissionMatirx(I_sequences, O_sequences)
		self.calPi(I_sequences)


	def calTransferMatrix(self, sequences):
		'''
		Compute the transfer matrix.
		:param sequences: list
						The sequence of hidden state.

		hidden state: 0, 1, 2, 3
		0: the head of the word.
		1: the middle word.
		2: the tail of the word.
		3: single wold.
		'''

		T = np.zeros((4, 4), dtype=np.double)
		for seq in sequences:
			leng = len(seq)
			for i in range(leng - 1):
				T[seq[i]][seq[i+1]] += 1
		N = np.sum(T, axis=1).reshape((4,1))
		self.A = T / N
		self.A = np.log(self.A)


	def calEmissionMatirx(self, I_sequences, O_sequences):
		'''
		Compute the emission matirx.
		:param I_sequences: list
							The sequence of hidden state.
		:param O_sequences: list
							The sequence of observation state.

		The number of the observation state: 65535 
		'''

		T = np.ones((4, 65535), dtype=np.double) 
		n = len(I_sequences)

		for i in range(n):
			for j in range(len(I_sequences[i])):
				T[I_sequences[i][j]][ord(O_sequences[i][j])] += 1
		N = np.sum(T, axis=1).reshape((4,1))
		self.B = T / N
		self.B = np.log(self.B)


	def calPi(self, sequences):
		'''
		Compute the initial probability of the hidden state.
		:param sequence: list
						The sequence of hidden state.
		'''

		B = 0
		S = 0
		n = len(sequences)
		for i in range(n):
			if sequences[i][0] == 0:
				B += 1
			elif sequences[i][0] == 3:
				S += 1
		self.Pi[0][0] = float(B) / n
		self.Pi[3][0] = float(S) / n
		self.Pi = np.log(self.Pi)


	def decode(self, text):
		'''
		Segmenting the chinese text.
		:param text: string
					The chinese text.
		'''

		uni_text = text.decode('utf-8').strip()
		n = len(uni_text)
		#the sequence of hidden state
		i_seq = []

		(path, s) = self.vertebi(uni_text)
	
		i_seq.append(s)
		for i in range(n-1):
			s = path[s][n-i-1]
			i_seq.insert(0, s)

		#list for storing word after segmented
		seg_list = []
		head = 0
		for i in range(n):
			if i_seq[i] == 2 or i_seq[i] == 3:
				tail = i + 1			
				seg_list.append(uni_text[head:tail].encode('utf-8'))
				head = tail
		if i_seq[n-1] == 0:
			seg_list.append(uni_text[n-1:n].encode('utf-8'))
		if i_seq[n-1] == 1:
			for i in range(n-2, 0, -1):
				if i_seq[i] == 0:
					break;
			seg_list.append(uni_text[i:n].encode('utf-8')) 

		
		seg_text = ' '.join(seg_list)
		return seg_text


	def vertebi(self, text):
		'''
		The vertebi algotithm.
		:param text: unicode string
					the chinese text.
		'''
		n = len(text)
		dist = np.zeros((4, n))
		path = np.zeros((4, n), dtype=np.int)

		for i in range(4):
			dist[i][0] = self.Pi[i][0] + self.B[i][ord(text[0])]

		for t in range(1,n):
			for i in range(4):
				maxScore = None
				for j in range(4): 
					score = dist[j][t-1] + self.A[j][i] + self.B[i][ord(text[t])]
					if  score > maxScore:
						maxScore = score
						path[i][t] = j
				dist[i][t] = maxScore
		s = np.argmax(dist[:, n-1])
		return (path, s)

