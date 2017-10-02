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
	
	public void update_rank(int newRank){
		this.clusterRank = newRank;
	}
}