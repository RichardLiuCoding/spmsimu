import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
#For Python 3.8+, to avoid SyntaxWarning: "is" with a literal. Did you mean "=="?

__all__ = ['simulator']


from spmsimu.simulator import *
