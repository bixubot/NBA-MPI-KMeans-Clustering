package edu.davidson.csc357.mpi.clustering;

import java.io.*;

import java.util.*;

import mpi.Comm;
import mpi.MPI;
import mpi.MPIException;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

import com.opencsv.CSVReader;

import edu.davidson.csc357.mpi.clustering.Data;

//Class that implements KMeansCluster on MPI
public class KMeansCluster {
	private int ITERATION;//number of iterations to apply K-means Algorithm for clustering
	private int K_CLUSTERS;//number of clusters to group the data points
	private static int FEATURES = 56;//number of features each point contains
	
	private Random random;

	private Data[] data;//the entire set of data points
	private Data[] parsedData;//local parsed dataset stored in each machine
	private DoubleBuffer featureSums;//hold the sums of respective features for clusters from local machine
	private DoubleBuffer[] NSums;//hold an array of number of points in each cluster
	private IntBuffer sizesOfClusters;//placeholder for all calculated numbers of points for each cluster respectively 
	private DoubleBuffer globalSums;//a global DoubleBuffer to gather all featureSums from all machines
	
	private Comm communicator;
	private int rank; //rank of local machine
	private int size; //number of machines
	
	private int[] NDispls; //array of displacements for gatherv to collect sizes
	private int[] NRecvCount; //array of number of items to receive from buffer of each machine
	
	private int[] sumDispls; //array of displacements for featureSums to gatherv
	private int[] sumRecvCount; //array of counts of items in each featureSum to gatherv
	
	private int block_size; //size of parsed data in local machine
	
	private DoubleBuffer globalCenters; //k centers for k clusters globally present

	
	//Constructor
	public KMeansCluster (Comm communicator, int rank, int size, Data[] data, int iteration, int k_clusters){
		this.ITERATION = iteration;
		this.K_CLUSTERS = k_clusters;
		this.communicator = communicator;
	 	this.rank = rank;
	 	this.size = size;
	 	this.data = data;
	 	
	 	// Block size of data for each thread; Account for modular division 
	 	block_size = (rank != size -1) ? data.length / size : data.length % size; 
	 	
	 	this.parsedData = new Data[block_size]; 
	 	this.featureSums = MPI.newDoubleBuffer(FEATURES * K_CLUSTERS);
	 	this.NSums = new DoubleBuffer[K_CLUSTERS];
	 	this.sizesOfClusters = MPI.newIntBuffer(K_CLUSTERS * size);
	 	this.globalSums = MPI.newDoubleBuffer(FEATURES * K_CLUSTERS * size);
	 	
	 	globalCenters = MPI.newDoubleBuffer(K_CLUSTERS*FEATURES);
	 	
	 	NRecvCount = new int[size];
	 	NDispls = new int[size];
	 	sumRecvCount = new int[size];
	 	sumDispls = new int[size];
	 	for (int i = 0; i < size; i++){
	 		NRecvCount[i] = K_CLUSTERS;
	 		NDispls[i] = i * K_CLUSTERS;
	 		sumRecvCount[i] = FEATURES * K_CLUSTERS;
	 		sumDispls[i] = i * FEATURES * K_CLUSTERS;
	 	}
 		
 		DoubleBuffer tempN = MPI.newDoubleBuffer(1);
 		tempN.put(0, 0.0);
	 	for (int k = 0; k < K_CLUSTERS; k++){
	 		NSums[k] = tempN;
	 	}
	}
	
	
	//Helper method which takes in type Data and return DoubleBuffer of its features array
	private DoubleBuffer DataToDoubleBuffer(Data point){
		DoubleBuffer b = MPI.newDoubleBuffer(FEATURES);
		double[] array = point.get_data();
		
		for (int i = 0; i < FEATURES; i++){
			b.put(i, array[i]);	
		}
		return b;
	}
	
	
	//To begin clustering with k many randomly chosen centers
	public void randomlyChosen() throws MPIException {
		if (rank == 0) {
			random = new Random();

			for (int k = 0; k < K_CLUSTERS; k++){
				int index = random.nextInt(360);

				DoubleBuffer temp = DataToDoubleBuffer(data[index]);

				for (int j = 0; j < FEATURES; j++){
					globalCenters.put(k * FEATURES + j, temp.get(j));
				}
			}
		}
	 	
		//broadcast to all machines
		communicator.bcast(globalCenters, K_CLUSTERS*FEATURES, MPI.DOUBLE, 0);
		communicator.barrier();
	}
	
	
	//to parse features of center of cluster k from the globalCenters
	private DoubleBuffer getCenterFromRange(int start){
		DoubleBuffer result = MPI.newDoubleBuffer(FEATURES);
		for (int i = 0; i < FEATURES; i++){
			result.put(i, globalCenters.get(i+start));
		}
		return result;
	}
	
	
	//start KMeansClustering
	public void kMeansClustering() throws MPIException{
		int index = rank*360/size;//mark the starting index of local data in the entire dataset
		
		//divide the 1-dimension globalCenters into DoubleBuffer array of size k
		DoubleBuffer[] centers = new DoubleBuffer[K_CLUSTERS];
		for (int k = 0; k < K_CLUSTERS; k++){
			centers[k] = getCenterFromRange(k * FEATURES);
		}
		
		//start KMeansClustering with iterations designated
		for (int i = 0; i < ITERATION; i++){
			//calculate distance for each point
			for (int j = 0; j < parsedData.length; j++){
				Data currentData = parsedData[j];
				double[] current = currentData.get_data();
				double smallestDistance = 0.0;
				int rankCluster = 0;
				
				//compare distances between all centers to get the closest one
				for (int k = 0; k < K_CLUSTERS; k++){
					DoubleBuffer center = centers[k];
					
					//call helper to calculate the distance between two points
					double currentDistance = distanceMetrics(current, center);
					
					//update the smaller distance and rank of respective cluster
					if ((smallestDistance == 0.0) || (currentDistance < smallestDistance)){
						smallestDistance = currentDistance;
						rankCluster = k;
					}
				}
				
				//update the point to its closest center's cluster
				parsedData[currentData.get_id()-index].update_rank(rankCluster);
			}
			
			communicator.barrier();//wait for current iteration to finish in all machines
			
			computeSums(); //calculate sums of features with respect to clusters
			updateCenters();//MPI: calculate new centers
			
			communicator.barrier();//wait until done to start a new iteration
			
		}
	}
	
	
	//Helper method called in computeSum to sum of sizes of respective cluster across all machines
	private int sumSizeForCluster(int cluster_number, IntBuffer buffer){
		int sum = 0;
		for (int i = 0; i < size; i++){
			sum += buffer.get(i * K_CLUSTERS + cluster_number);
		}
		return sum;
	}
	
	
	//helper method to calculate the sums of each feature's values from 
	//all points in their own clusters respectively across all machines
	//as well as the number of points in each cluster across all machines
	public void computeSums() throws MPIException {
		//each time start with empty features sums
	 	for (int i = 0; i < FEATURES * K_CLUSTERS; i++){
	 		featureSums.put(i,0.0);
	 	}
	 	
	 	for (int k = 0; k < K_CLUSTERS; k++){
	 		int size_cluster= 0;
	 		for (int i = 0; i < parsedData.length; i++){
	 			if (parsedData[i].get_rank() == k){
	 				for (int j = 0; j < FEATURES; j++){
	 					//accumulate the value of current feature from all points in current cluster from all machines
	 					featureSums.put(k * FEATURES + j, (featureSums.get(k * FEATURES + j) + parsedData[i].get_data()[j]));
	 				}
	 			}
	 			size_cluster += 1;//count the points in current cluster from all machines
	 		}
	 		NSums[k].put(0, size_cluster);//record the sizes with respect to its cluster
	 	}

	 	communicator.barrier();

	}
	
	
	//use MPI gatherv to calculate new centers in root machine and broadcast to all
	public void updateCenters() throws MPIException{
		//collect all sizes of clusters from local machine
		IntBuffer sum_k = MPI.newIntBuffer(K_CLUSTERS);
		for (int k = 0; k < K_CLUSTERS; k++){
			sum_k.put(k, (int)NSums[k].get(0));
		}

		//gather all sizes of clusters from all machines
		communicator.gatherv(sum_k, K_CLUSTERS, MPI.INT, sizesOfClusters, NRecvCount, NDispls, MPI.INT, 0);
		
		int[] clusterSizes = new int[K_CLUSTERS];
		//calculate the entire size of clusters across all machines
		for (int k = 0; k < K_CLUSTERS; k++){
			int clusterKSize = sumSizeForCluster(k, sizesOfClusters);
			clusterSizes[k] = clusterKSize;
		}
		
		//gather all features sums for k clusters from all machines
		communicator.gatherv(featureSums, FEATURES * K_CLUSTERS, MPI.DOUBLE, globalSums, sumRecvCount, sumDispls, MPI.DOUBLE, 0);
		communicator.barrier();
		
		//placeholder for updated centers generated in root machine
		DoubleBuffer updatedCenters = MPI.newDoubleBuffer(K_CLUSTERS * FEATURES);

		//iterate over k clusters
		for (int k = 0; k < K_CLUSTERS; k++){
			//iterate over j features
			for (int j = 0; j < FEATURES; j++){
				double sum = 0.0;
				//sum up respective features of a specific cluster k over all m machines
				for (int m = 0; m < size; m++){
					sum += globalSums.get(m * FEATURES * K_CLUSTERS + k * FEATURES + j);
				}
				//calculate the new average for the respective feature of this cluster k
				double average = sum/clusterSizes[k];
				
				//put calculated new centers into the buffer
				updatedCenters.put(j+k*FEATURES, average);
			}
		}
		
		//update centers locally
		globalCenters = updatedCenters;
		//then broadcast to all machines
		communicator.bcast(globalCenters, K_CLUSTERS*FEATURES, MPI.DOUBLE, 0);
		communicator.barrier();
	}

	
	//calculate the euclidean distance between two points
	public double distanceMetrics(double[] x, DoubleBuffer y){

		double distance = 0.0;
		for (int i = 0; i < FEATURES; i++){
			distance += Math.pow((x[i] - y.get(i)),2.0);
		}
	 	return distance;
	}

	
	//load parsed data for local machine
	public void load() throws MPIException {
		int index = rank*360/size;
		for (int i = 0; i < block_size; i++){
			parsedData[i] = data[i+index];
		}
	 	communicator.barrier();
	}
	
	
	//helper method which allows users to see the resulted clusters
	public void printResult(){
		for (int i = 0; i < parsedData.length; i++){
			System.out.println(parsedData[i].get_team()+": "+parsedData[i].get_rank());
		}
	}
	
	
	public static void main(String[] args) throws MPIException, IOException{
	 	String filename = "/Users/bixu/Desktop/Team_Play_By_Play_Summary_Data_4-15.csv"; //file path subject to change
	 	CSVReader reader = new CSVReader(new FileReader(filename),',');

	 	String[] header = reader.readNext();//skip the header row in csv file

	 	Data[] data = new Data[360];
	 	String[] nextPoint;

	 	int indexTemp = 0;
	 	
	 	while ((nextPoint = reader.readNext()) != null){
		 	double[] temp = new double[56];
		 	//read each row and store features' values into double array
	 		for (int i = 1; i < nextPoint.length; i++) {
	 			temp[i-1] = Double.valueOf(nextPoint[i]);
	 		}
	 		//generate each point with type Data and add into the dataset
	 		Data point = new Data(temp, indexTemp, -1, nextPoint[0]);
	 		data[indexTemp] = point;
	 		
	 		indexTemp++;//keep id and data point match

	 	}
	 	reader.close();
	 	
	 	//initiate MPI
	 	MPI.Init(args);

	 	Comm comm = MPI.COMM_WORLD;
	 	
	 	int rank = comm.getRank();
	 	int size = comm.getSize();

	 	int it = Integer.valueOf(args[0]);//number of iteration for KMeansClustering
	 	int k_c = Integer.valueOf(args[1]);//number of clusters
	 	KMeansCluster cluster = new KMeansCluster(comm, rank, size, data, it, k_c);

	 	cluster.load();//local machine load parsed data
	 	
	 	// Start Time
	 	final long startTime = System.currentTimeMillis();
	 	
	 	cluster.randomlyChosen();//randomly choose centers to begin with

	 	cluster.kMeansClustering();//begin clustering

	 	//cluster.printResult();
	 	
	 	MPI.Finalize();
	 	// End Time
	 	final long endTime = System.currentTimeMillis();
	 	if (rank == 0){
	 		System.out.println("K="+k_c + ", Iterations=" + it + ", #Threads=" + size + ", Total Execution Time: " + (endTime- startTime) * 1.0 / 1000 + " seconds.");
	
	 	}
	}
}