


a = {'0':'0','1':'1','2':'-1','3':'3','4':'-1'}

count = 0
update = list(a.values())
update.sort(key=lambda x: int(x))
map_vals = {}

for v in update:
    if v not in map_vals:
        map_vals[v] = count
        count += 1
        
new_a = {k:map_vals[v] for k,v in a.items()}


print()
