

test2 = [
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            [0,0,0,0,0,0,0,1,1,0,0,0,0,0]
        ]

def distance(list1, list2):
    """Distance between two vectors."""
    squares = [(p-q) ** 2 for p, q in zip(list1, list2)]
    return sum(squares) ** .5

d2 = distance(test2[0], test2[1]) 

print(d2)