********************************README File for Linear Regression using OLS and Gradient Descent Approach***********************************************

OLS Method:

1. To implement linear regression, I made use of Numpy packages to find transpose and inverse of a matrix and lambda functions. 

2. I wrote two functions, one for calculating X*X^T and the other for calculating X*Y. In the map, it computes the value of Xi*Xi^T/Xi*Yi by calling the functions and the reducer computes the sum of all the sub-values.

3. To run the python file, run the command given below.

spark-submit linreg.py hdfs:///user/cloudera/LinReg/SparkInput/yxlin.csv

4. The output can be found in yxlin.out and yxlin2.out.



Gradient Descent Approach:

1. To calculate linear regression in this method, I made use of one function to compute alpha* X^T * (Y - X * beta).

2. The map function calls this method and the reducer adds the sub-values. I add the existing value of beta to the above result and assign the new value to beta.

3. Iterate the steps 1 to 2 for n iterations given in the command line arguments.

4. To run the python file, run the following command.

spark-submit graddesc.py hdfs:///user/cloudera/LinReg/SparkInput/yxnew.csv 0.01 50

where 0.01 is the value of alpha and 50 is the number of iterations.

P.S I used a matrix of 1's for the initial value of beta.
