from pickle import load
import sys
sys.path.append('./')

from data.geom_dataloader import geom_dataloader

if __name__=='__main__':
    loader = geom_dataloader('chameleon')
    loader.load_data()
    print(loader)
    print(1)