Instructions on running MPI program and Spark MLlib program in terminal

—————————————
MPI Code:

Download the zip file for our MPI code. Import the zip file to your Eclipse project and correctly import “mpi.jar” and “opencsv.jar” as external libraries of the project. Then, unzip the file and navigate to the file directory. Run the following command in terminal to start the MPI code:

cd bin/
mpirun -np [# of threads] java -cp “.:../opencsv.jar" edu/davidson/csc357/mpi/clustering/KMeansCluster [# of iterations] [# of clusters]

The program will run K-Means Clustering accordingly and display the total execution time. If you want to see the result of the clustering, go to the bottom part of KMeansCluster.java and uncomment “printResult();”.

————————
Spark MLlib Code:

Download the zip file for our Spark Code. Import the zip file to your Eclipse project and correctly import all corresponding Spark dependency jar files as external libraries of the project. Then unzip the file and navigate to the file directory. Run the following command in terminal to start the Spark MLlib code:

cd src/
javac -cp ".:/usr/local/Cellar/apache-spark/2.0.1/libexec/jars/*" Spark_KMeans.java
java -cp ".:/usr/local/Cellar/apache-spark/2.0.1/libexec/jars/*" Spark_KMeans [# of iterations] [# of clusters]

*Note that here the classpath for Spark jar files are subject to change.

*Also note that the file pathes for the dataset are already configured in the codes based on its location in our local file system. They are subject to change in order for the codes to execute properly. The datasets are included in each zip file so that’s an easy fix.

The program will run K-Means Clustering accordingly and display the total execution time. If you want to see the result of the clustering, go to the bottom part of KMeansCluster.java and uncomment the block of codes that save the result to file.

(Please email us if you have trouble running the code in terminal. We think it’s easier to test with different parameters by running the java codes in terminal.)