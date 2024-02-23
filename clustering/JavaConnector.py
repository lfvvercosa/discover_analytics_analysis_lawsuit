import subprocess


class JavaConnector():

    jar_path = ''

    def __init__(self, jar_path):
        self.jar_path = jar_path

    def run_java(self,
                 log_path):
        
        comp_proc = subprocess.run(['java', 
                                    '-jar', 
                                    self.jar_path,
                                    log_path,
                                   ],capture_output=True)
        
        return comp_proc.stdout
