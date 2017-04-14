package bda.tensorflow.nn.LayerInfo;

import bda.tensorflow.exception.LayerCreateException;
import bda.tensorflow.jni_11.Input;

/**
 * Created by yixuanhe on 9/20/16.
 */
public class PoolLayerInput extends LayerInput {
    public int[] ksize;
    public int[] strides;
    public String padding;

    public PoolLayerInput() {
    }

    public PoolLayerInput(int[] ksize, int[] strides, String padding)
            throws LayerCreateException {
        this.ksize = ksize;
        this.strides = strides;

        this.padding = padding;
    }

    public PoolLayerInput(Input[] input, int[] dimension, int[] ksize, int[] strides, String padding)
            throws LayerCreateException {
        super(dimension);
        if (dimension.length != 4)
            throw new LayerCreateException("Pool dimension must have 4 elements!");
        this.ksize = ksize;
        this.strides = strides;
        this.padding = padding;
    }
}
