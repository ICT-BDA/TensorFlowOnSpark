package bda.tensorflow.nn.LayerInfo;

import bda.tensorflow.jni_11.Input;

/**
 * Created by yixuanhe on 9/20/16.
 */
public class LossLayerOutput extends LayerOutput{
    public Input loss;
    public Input lossgrad;
    public Input[] predict;

    public LossLayerOutput(Input[] input, Input loss, Input lossgrad) {
        super(input);
        this.loss = loss;
        this.lossgrad = lossgrad;
    }

    public LossLayerOutput(Input[] output) {
        super(output);
        this.predict = predict;
    }
}
