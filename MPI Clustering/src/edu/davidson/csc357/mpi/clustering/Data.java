package edu.davidson.csc357.mpi.clustering;

//Arthor: Brandon Liang, Binwei Xu
//
//Class for each data point which includes:
//    id: the index of this point in the original dataset
//    data: a double array that includes the features of this point
//    team: the string representation of this point
//    clusterRank: cluster to which it is clustered into
public class Data {
	private int id;
	private double[] data;
	private String team;
	private int clusterRank;
	
	public Data(double[] data, int id, int rank, String team){
		this.id = id;
		this.data = data;
		this.team = team;
		this.clusterRank = rank;
	}
	
	public String get_team(){
		return team;
	}
	
	public int get_id(){
		return id;
	}
	
	public double[] get_data(){
		return data;
	}
	
	public int get_rank(){
		return clusterRank;
	}
	
	//A method to update the cluster of this point
	public void update_rank(int newRank){
		this.clusterRank = newRank;
	}
}