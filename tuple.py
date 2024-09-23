import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

a = (5,6)
b = (2,3)
def cong(x,y):
	return (x[0]*y[1] + x[1]*y[0],x[1]*y[1])

def rutgon(x):
	gcd = np.gcd(x[0],x[1])
	return x/gcd

c = cong(a,b)
print(rutgon(c))