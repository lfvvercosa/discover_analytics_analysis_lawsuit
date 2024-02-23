


al = [1,2,4]
bl = [1,2,4]
cl = [1,2,4]
count = 0

for a in al:
    for b in bl:
        for c in cl:
            if (a % 2) == 0 and (b % 2) == 0 and (c % 2) == 0:
                continue
            print(a,b,c)
            count += 1

print('count: ' + str(count))