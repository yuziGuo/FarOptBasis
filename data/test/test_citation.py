import sys
sys.path.append('./')

from data.citation_dataloader import  citation_loader

if __name__=='__main__':
    loader = citation_loader('Cora')
    print(1)