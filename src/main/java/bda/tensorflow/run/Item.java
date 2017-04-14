package bda.tensorflow.run;

import java.io.Serializable;

public class Item implements Serializable{
    public String variableName;
    public int[] shape;
    public int type;
    public InitValueConfig config;
  
    public Item(String variableName, int type, int[] shape, InitValueConfig config) {
        this.variableName = variableName;
        this.type = type;
        this.shape = shape;
        this.config = config;
    }
}
