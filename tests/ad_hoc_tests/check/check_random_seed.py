import random


l = [1,4,3,2,8]

seed = 42
random.seed(seed)
random.shuffle(l)

print(l)