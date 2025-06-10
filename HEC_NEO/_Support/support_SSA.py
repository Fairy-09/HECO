import numpy as np
# import matplotlib.pyplot as plt

#data:(9608,1)
def SSA(series, level):
	# series = 0
	series = series - np.mean(series)  # 中心化(非必须)

	# step1 嵌入
	windowLen = level  # 嵌入窗口长度4
	seriesLen = len(series)  # 序列长度9608
	K = seriesLen - windowLen + 1   #K:(9605,)
	X = np.zeros((windowLen, K))    #X:(4,9605)
	for i in range(K):
		X[:, i] = series[i:i + windowLen]

	# step2: svd分解， U和sigma已经按升序排序
	U, sigma, VT = np.linalg.svd(X, full_matrices=False)   #U:(4,4)

	for i in range(VT.shape[0]):
		VT[i, :] *= sigma[i]
	A = VT    #A:(4,9605),VT:(4,9605)

	# 重组;重构
	rec = np.zeros((windowLen, seriesLen))
	for i in range(windowLen):
		for j in range(windowLen - 1):
			for m in range(j + 1):
				rec[i, j] += A[i, j - m] * U[m, i]
			rec[i, j] /= (j + 1)
		for j in range(windowLen - 1, seriesLen - windowLen + 1):
			for m in range(windowLen):
				rec[i, j] += A[i, j - m] * U[m, i]
			rec[i, j] /= windowLen
		for j in range(seriesLen - windowLen + 1, seriesLen):
			for m in range(j - seriesLen + windowLen, windowLen):
				rec[i, j] += A[i, j - m] * U[m, i]
			rec[i, j] /= (seriesLen - j)
	return rec

# rrr = np.sum(rec, axis=0)  # 选择重构的部分，这里选了全部
#
# plt.figure()
# for i in range(10):
# 	ax = plt.subplot(5, 2, i + 1)
# 	ax.plot(rec[i, :])
#
# plt.figure(2)
# plt.plot(series)
# plt.show()
