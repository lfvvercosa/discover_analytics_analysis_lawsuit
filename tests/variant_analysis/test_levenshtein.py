from leven import levenshtein


a = 'ABCDE'
b = 'ABCDEF'

a = 'ABD'
b = 'AB'

print(levenshtein(a, b))