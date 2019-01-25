import numpy as np

#a= numpy.array([0,1,2])
tmp=np.array([25.10377891,  24.94944888, 10.27264423,  -9.86779197, -24.10733213,
-25.35074792])
tmp2=np.array([-661.1870631 , -628.49670789, -273.07940635,  280.03152613,
643.05788743,  660.00709711])
print "tmp=%s\n"%(tmp)
print "tmp2=%s\n"%(tmp2)

covMat=np.cov(tmp,tmp2)
print "covMat=%s"%(covMat)
covMatDeterminant=np.linalg.det(covMat)
print "covMatDeterminant=%s"%(covMatDeterminant)

#c=a+b
#print c




