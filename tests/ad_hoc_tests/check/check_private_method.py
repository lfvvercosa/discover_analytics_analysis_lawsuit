class Carro:
    def andar(self):
        print('vrum!')
    
    def __hidden(self):
        print('forbiden!')


c = Carro()
c.andar()
c.__hidden()