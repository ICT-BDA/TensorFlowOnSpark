package bda.tensorflow.nn.LayerInfo;

import bda.tensorflow.jni_11.Input;
import com.fasterxml.jackson.annotation.JsonIgnore;

/**
 * Created by yixuanhe on 9/19/16.
 */
public class RNNLayerInput extends LayerInput{
    @JsonIgnore
    public Input seq;

    public RNNLayerInput(int[] dimension) {
        super(dimension);
    }

    public RNNLayerInput() {
    }

    @Override
    public boolean needSeq(){
        return true;
    }

    @Override
    public void setSeq(Input seq){
        this.seq = seq;
    }

}
