f = open("test.txt","r").readlines()
f = f[0].split(",")

res = 0
for i in f:
	if i != '':
		print(i)
		res = res + int(i)


print(res)