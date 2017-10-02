
import java.io.*;
import java.util.*;
import java.util.stream.IntStream;

import mpi.Comm;
import mpi.MPI;
import mpi.MPIException;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

import com.opencsv.CSVReader;

//import com.opencsv.CSVReader;


public class KMeansCluster {
	private static int ITERATION = 100;//number of iterations to apply K-means Algorithm for clustering
	private static int K_CLUSTERS = 30;
	private static int FEATURES = 56;

	private Data[] data;
	private Data[] parsedData;
	private DoubleBuffer[] sums;
	private DoubleBuffer[][] clusteredData;
	
	private Comm communicator;
	private int rank;
	private int size;

	private DoubleBuffer[] globalCenters = new DoubleBuffer[K_CLUSTERS];

	public KMeansCluster (Comm communicator, int rank, int size, Data[] data){
		this.communicator = communicator;
	 	this.rank = rank;
	 	this.size = size;
	 	this.data = data;
	 	this.parsedData = new Data[360/size];
	 	this.sums = new DoubleBuffer[K_CLUSTERS];
	 	this.clusteredData = new DoubleBuffer[size][K_CLUSTERS];
	 	
	 	DoubleBuffer temp = MPI.newDoubleBuffer(FEATURES+1);
 		for (int i = 0; i < FEATURES+1; i++){ //** +1?
 			temp.put(0.0);
 		}
 		
	 	for (int k = 0; k < K_CLUSTERS; k++){
	 		sums[k]=temp;
	 	}
	 	
	 	//this.globalCenters = new DoubleBuffer[K_CLUSTERS];
	}
	
	private DoubleBuffer Data_to_DoubleBuffer(int size, Data data){
		DoubleBuffer b = MPI.newDoubleBuffer(size);
		double[] array = data.get_data();
		for (int i = 0; i < size; i++){
			b.put(i, array[i]);
		}
		return b;
	}
	
	private void print_DoubleBuffer(int size, DoubleBuffer b){
		System.out.print("\n[");
		for (int i = 0; i < size; i++){
			System.out.print(b.get(i)+",");
		}
		System.out.print("]\n");
	}
	
	public void randomlyChosen() throws MPIException {
		if (rank == 0) {
			Random random = new Random();

			//		Data[] centers = new Data[K_CLUSTERS];
			for (int i = 0; i < K_CLUSTERS; i++){
				int index = random.nextInt(360);
				//			centers[i] = data[index];
				//print_array(data[index].get_data());
				DoubleBuffer temp = Data_to_DoubleBuffer(FEATURES,data[index]);
				//temp.put(data[index].get_data());
				globalCenters[i] = temp;
				//print_DoubleBuffer(FEATURES, globalCenters[i]);
			}
			
		}
		communicator.bcast(globalCenters, K_CLUSTERS, MPI.DOUBLE, 0);
	}
	
	public void print_array(double[] data){
		System.out.print("\n[");
		for (int i = 0; i < data.length; i++){
			System.out.print(data[i]+",");
		}
		System.out.print("]\n");
	}
	
	public void kMeansClustring() throws MPIException{
		int index = rank*360/size;
		
		for (int i = 0; i < ITERATION; i++){
			for (int j = 0; j < parsedData.length; j++){
				Data currentData = parsedData[j];
				double[] current = currentData.get_data();
				double smallestDistance = 0.0;
				int rankCluster = 0;
				
				for (int k = 0; k < K_CLUSTERS; k++){
					DoubleBuffer center = globalCenters[k];
					//print_array(current);
					//print_DoubleBuffer(FEATURES,center);
					double currentDistance = distanceMetrics(current, center);
					//update the smaller distance and rank of respective cluster
					if ((smallestDistance == 0.0) || (currentDistance <= smallestDistance)){
						smallestDistance = currentDistance;
						rankCluster = k;
					}
				}
				
				parsedData[currentData.get_id()-index].update_rank(rankCluster);
			}
			
			communicator.barrier();
			
			computeSums(); //calculate sums of features with respect to clusters
			updateCenters();//MPI: calculate new centers
			
			communicator.barrier();
			
		}
	}
	
	public void computeSums() throws MPIException {
	 	for (int k = 0; k < K_CLUSTERS; k++){
	 		int size_cluster= 0;
	 		for (int i = 0; i < parsedData.length; i++){
	 			if (parsedData[i].get_rank() == k){
	 				for (int j = 0; j < FEATURES; j++){
	 					sums[k].put(j, (sums[k].get(j) + data[i].get_data()[j]));
	 				}
	 			}
	 			size_cluster += 1;
	 		}
	 		sums[k].put(FEATURES, size_cluster);
	 	}
	 	
	 	communicator.barrier();
	}
	
	//use MPI AllToAll to calculate new centers for each clusters in parallel
	public void updateCenters() throws MPIException{
		IntBuffer[] sizesOfClusters = new IntBuffer[K_CLUSTERS];
		for (int i = 0; i < K_CLUSTERS; i++){
			communicator.gather(sums[i].get(FEATURES), 1, MPI.INT, sizesOfClusters[i], size, MPI.INT, 0);
		}
		int[] clusterSizes = new int[K_CLUSTERS];
		
		for (int k = 0; k < K_CLUSTERS; k++){
			int clusterKSize = IntStream.of(sizesOfClusters[k].array()).sum();//calculate the size of combined cluster with rank k
			clusterSizes[k] = clusterKSize;
			communicator.gather(sums[k], 57, MPI.DOUBLE, clusteredData[rank][k], 57, MPI.DOUBLE, 0);
		}
		communicator.barrier();
		
		DoubleBuffer[] updatedCenters = new DoubleBuffer[K_CLUSTERS];
		DoubleBuffer temp = MPI.newDoubleBuffer(FEATURES);
		for (int k = 0; k < K_CLUSTERS; k++){
			for (int j = 0; j < FEATURES; j++){
				double sum = 0.0;
				for (int m = 0; m < size; m++){
					sum += clusteredData[m][k].get(j);
				}
				double average = sum/clusterSizes[k];
				temp.put(j, average);
			}
			updatedCenters[k] = temp;
		}
		
		globalCenters = updatedCenters;
		
		communicator.bcast(globalCenters, K_CLUSTERS, MPI.DOUBLE, 0);
		
		communicator.barrier();
	}

	//calculate the euclidean distance
	public double distanceMetrics(double[] x, DoubleBuffer y){
		//print_DoubleBuffer(FEATURES,y);
		double distance = 0.0;
		for (int i = 0; i < FEATURES; i++){
			//distance += Math.pow(x[i], 2.0) - Math.pow(y.get(i), 2.0);
			//System.out.println(x[i]+" "+y.get(i));
			distance += Math.pow((x[i] - y.get(i)),2.0);
		}
	 	return distance;
	}

	public void load() throws MPIException {
		int index = rank*360/size;
		for (int i = 0; i < 360/size; i++){
			parsedData[i] = data[i+index];
		}

	 	communicator.barrier();
	}
	
	public void print_result(){
		System.out.println("Rank is: "+rank);
		for (int i = 0; i < parsedData.length; i++){
			System.out.println(parsedData[i].get_team()+": "+parsedData[i].get_rank());
		}
	}
	
	public static void main(String args[]) throws MPIException, IOException{
	 	String filename = "/Users/brandonliang/Desktop/*5. NBA Stats Analytics Research/2016-2017 NBA Simulation/Team_Play_By_Play_Summary_Data_4-15.csv";
	 	CSVReader reader = new CSVReader(new FileReader(filename),',');

	 	String[] header = reader.readNext();

	 	Data[] data = new Data[360];
	 	String[] nextPoint;
	 	double[] temp = new double[56];
	 	int indexTemp = 0;
	 	
	 	while ((nextPoint = reader.readNext()) != null){
	 		for (int i = 1; i < nextPoint.length-1; i++) {
	 			temp[i-1] = Double.valueOf(nextPoint[i]);
	 		}
	 		data[indexTemp] = new Data(temp, indexTemp, 0, nextPoint[0]);
	 		indexTemp++;//keep id and data point match
	 	}
	 	
	 	reader.close();
	 	
	 	MPI.Init(args);

	 	Comm comm = MPI.COMM_WORLD;
	 	
	 	int rank = comm.getRank();
	 	int size = comm.getSize();

	 	KMeansCluster cluster = new KMeansCluster(comm, rank, size, data);

	 	cluster.load();
	 	
	 	cluster.randomlyChosen();
	 	
	 	cluster.kMeansClustring();

	 	cluster.print_result();
	 	
	 	MPI.Finalize();
	}
}