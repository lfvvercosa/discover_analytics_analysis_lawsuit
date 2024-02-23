class Friend:
    name = ''

    def __init__(self, name):
        self.name = name

a = {Friend('Luiz') : True}
b = {Friend('Ali') : True}

print(dict(a, **b))