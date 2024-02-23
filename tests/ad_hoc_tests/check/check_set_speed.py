import time


s = set()
d = {}
l = []

for i in range(1000000):
    s.add(str(i))
    d[i] = True
    l.append(i)

start_time = time.time()
if '500000' in s:
    print('found in set')
print("--- %s seconds in set ---" % (time.time() - start_time))

start_time = time.time()
if '500000' in d:
    print('found in dict')
print("--- %s seconds in dict ---" % (time.time() - start_time))

start_time = time.time()
if '500000' in s:
    print('found in list')
print("--- %s seconds in list ---" % (time.time() - start_time))