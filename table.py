import numpy as np

a = np.load('compare.npy')
b = a[1]/a[2]

avg = sum(a[1].T) / sum(a[2].T)

s = ''
for i in range(15):
	s = s + str(i+1) + '&'
	for j in range(14):
		if (i == j):
			s = s + '&'
			continue
		s = s + '%.2f' % (b[i][j]) + '&'
	if (i < 14):
		s = s + '%.2f' % (b[i][j])
	s = s + '&' + '%.2f' % (avg[i])
	s = s + '\\\\\\hline\n'
s = s + 'average'
avg = sum(a[1]) / sum(a[2])

for i in range(15):
	s = s + '&' + '%.2f' % (avg[i]) 
print(s)