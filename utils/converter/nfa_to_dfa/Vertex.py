from os import sep


class Vertex:
    name = ""
    states = {}
    my_sep = ','
    is_init_state = False
    is_final_state = False
    is_empty_state = False


    def __init__(self, states, final_states, init_state, my_sep=','):
        self.my_sep = my_sep
        self.states = states.copy()
        self.name = self.get_name(self.states)

        for s in states:
            if s in final_states:
                self.is_final_state = True
        
            if s.name == init_state.name:
                self.is_init_state = True

        if len(states) == 0:
            self.is_empty_state = True


    def get_name(self, states):
        name = '{'
        name_list = []

        if states:
            for s in states:
                name_list.append(s.name)
            
            name_list.sort()
            
            for n in name_list:
                name += n + self.my_sep

            return name[:-1] + '}'
        else:
            return '{}'
    

    def __str__(self):
        return self.name
    

    def __repr__(self):
        return self.name
    

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, str):
            return self.name == other
        
        return False

    
    def __hash__(self):
        return hash(self.name)

    


