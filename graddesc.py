# graddesc.py
#
# Standalone Python/Spark program to perform linear regression using Gradient Descent.
# 
# TODO: Write this.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
#
#

import sys
import numpy as np
#import pydoop.hdfs as hd
from pyspark import SparkContext


def XTY(d,beta,alpha):
    #"To find alpha*X^T(Y-X*Beta)
    y = float(d[0])
    #Add 1.0 to the array
    d[0] = 1.0
    temp = np.array(d).astype('float')
    x = np.asmatrix(temp)
    #Reshaping for transpose
    xt=x.reshape((-1,1))
    xt=alpha*xt
    #print np.shape(x)
    #print np.shape(beta)
    xb=np.dot(x,beta)
    yxb=y-xb
    xtyxb=np.dot(xt,yxb)
    b=xtyxb
    #print xty.shape
    return b



if __name__ == "__main__":
  if len(sys.argv) !=4:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])
  #Get Alpha
  alpha=float(sys.argv[2])
  #Get the # of iterations
  iterations=int(sys.argv[3])
  #yxoutputFile=sc.textFile(sys.argv[2])

  yxlines = yxinputFile.map(lambda line: line.split(','))
  
  yxfirstline = np.array(yxlines.first()).astype('float')

  #print yxfirstline
  #print yxlines

  #This length is needed to form beta matrix
  length=len(yxfirstline)
 
  #print length

  beta=np.ones((length,1), dtype=np.float)
  beta=np.asmatrix(beta)
  #print beta

  # Gradient Descent Approach

  # Calculate b on multiplying the alpha*X^T(Y-X*Beta) and sum up all the sub-values by using reduceByKey
  for i in range(iterations):
  	b = np.asmatrix(yxlines.map(lambda d: ("X^T.Y",XTY(d,beta,alpha))).reduceByKey(lambda x1,x2: np.add(x1,x2)).map(lambda d: d[1]).collect()[0])
	#Add beta to the result
	beta=beta+b

  # print the linear regression coefficients in desired output format

  print "beta: "
  #Convert the beta array to a list 
  newBeta = np.array(beta).tolist()
  #beta.saveAsTextFile(sys.argv[2])

  for coeff in newBeta:
       print coeff
  
  sc.stop()
