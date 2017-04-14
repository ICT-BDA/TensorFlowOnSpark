package bda.tensorflow.nn.LayerInfo;

import bda.tensorflow.exception.LayerCreateException;
import bda.tensorflow.jni_11.Input;

/**
 * Created by yixuanhe on 01/11/2016.
 */
public class ConvLayerInput extends LayerInput {
    public int[] filter;
    public int[] strides;
    public String padding;

    public ConvLayerInput() {
    }

    public ConvLayerInput(int[] filter, int[] strides, String padding)
            throws LayerCreateException {
        this.filter = filter;
        this.strides = strides;

        this.padding = padding;
    }

    public ConvLayerInput(Input[] input, int[] dimension, int[] filter, int[] strides, String padding)
            throws LayerCreateException {
        super(dimension);
        if (dimension.length != 4)
            throw new LayerCreateException("Pool dimension must have 4 elements!");
        this.filter = filter;
        this.strides = strides;
        this.padding = padding;
    }
}
