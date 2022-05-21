import os

def move_to_top_directory():
    while '/analysis' in os.getcwd():
        os.chdir('..')