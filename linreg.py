# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
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

def XTX(d):
    #To find X^T*X
    #Add 1.0 to the array
    d[0]=1.0
    temp = np.array(d).astype('float')
    x = np.asmatrix(temp)
    #Reshaping for transpose
    xt=x.reshape((-1,1))
    xxt = np.dot(xt,x)
    #print xxt.shape
    return xxt

def XTY(d):
    #"To find X^T * y"
    y = float(d[0])
    #Add 1.0 to the array
    d[0] = 1.0
    temp = np.array(d).astype('float')
    x = np.asmatrix(temp)
    #Reshaping for transpose
    xt=x.reshape((-1,1))
    #y is scalar
    xty = np.multiply(xt,y)
    #print xty.shape
    return xty


if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])
  #yxoutputFile=sc.textFile(sys.argv[2])

  yxlines = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = yxlines.first()

  #print yxfirstline
  #print yxlines

  # Calculate A on multiplying the xi^T and xi and sum up all the xi and xi^T values by using reduceByKey
  A = np.asmatrix(yxlines.map(lambda d: ("X^T.X",XTX(d))).reduceByKey(lambda x1,x2: np.add(x1,x2)).map(lambda d: d[1]).collect()[0])

  # Calculate b on multiplying the xi^T and scalar Yi and sum up all the xi^T and Yi values by using reduceByKey
  b = np.asmatrix(yxlines.map(lambda d: ("X^T.Y",XTY(d))).reduceByKey(lambda x1,x2: np.add(x1,x2)).map(lambda d: d[1]).collect()[0])
  
  #Calculate A^-1 * b
  beta = np.dot(np.linalg.inv(A),b)

  # print the linear regression coefficients in desired output format

  print "beta: "
  #Convert the beta array to a list 
  newBeta = np.array(beta).tolist()
  #beta.saveAsTextFile(sys.argv[2])

  for coeff in newBeta:
       print coeff

  sc.stop()
