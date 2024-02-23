


def func(a,b):
    c = 0

    def another(a):
        if c > 0:
           
            return 1
        else:
            c = c + 1
            return 2

    print(another(a))
    print(c)


a = 2
b = 3

func(a,b)
