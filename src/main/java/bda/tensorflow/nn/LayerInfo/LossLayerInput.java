package bda.tensorflow.nn.LayerInfo;

import bda.tensorflow.jni_11.Input;
import com.fasterxml.jackson.annotation.JsonIgnore;

/**
 * Created by yixuanhe on 9/19/16.
 */
public class LossLayerInput extends LayerInput {
    public LossLayerInput() {
    }

    @JsonIgnore
    public Input[] y;

    public LossLayerInput(Input[] y, int[] dimension) {
        super(dimension);
        this.y = y;
    }

    public LossLayerInput(int[] dimension) {
        super(dimension);
    }
}
