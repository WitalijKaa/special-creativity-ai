import sys
from rich.pretty import pprint
# from php_wk import *
# from src.helpers.php_wk import *

DEVICE_CUDA = 'cuda'
DEVICE_CPU = 'cpu'

def dd(*args):
    if len(args) <= 8:
        for dump in args:
            pprint(dump)
    else:
        pprint(args)
    sys.exit()

def ddd(*args):
    if len(args) <= 4:
        for dump in args:
            pprint(dump)
    else:
        pprint(args)

def ddp(size: int = 2): # pause
    for no in range(size):
        print('')