package bda.tensorflow.run;

import java.io.Serializable;

public class InitValueConfig implements Serializable {
    public String name;
    public boolean isConstant;
    public float constant;
    public float mean;
    public float dev;
    public int dtype;
    public int[] shape;

    public InitValueConfig(String name, boolean isConstant, float constant, float mean, float dev, int dtype, int[] shape) {
        this.name = name;
        this.isConstant = isConstant;
        this.constant = constant;
        this.mean = mean;
        this.dev = dev;
        this.dtype = dtype;
        this.shape = shape;
    }
}
