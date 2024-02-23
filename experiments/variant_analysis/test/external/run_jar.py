import subprocess


path = '/home/vercosa/Documentos/jar_generation/TestJar.jar'

subprocess.call(['java', '-jar', path, 'abc', '-33'])

