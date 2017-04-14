package bda.tensorflow.nn.LayerInfo;

import bda.tensorflow.jni_11.Input;
import com.fasterxml.jackson.annotation.JsonIgnore;

/**
 * Created by yixuanhe on 07/10/2016.
 */
public class LayerInputWithSeq extends LayerInput{
    @JsonIgnore
    public Input seq;

    public LayerInputWithSeq(int[] dimension){
        super(dimension);
    }

    public LayerInputWithSeq() {
    }

    @Override
    public void setSeq(Input seq) {
        this.seq = seq;
    }

    @Override
    public boolean needSeq(){
        return true;
    }
}
