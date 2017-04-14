package bda.tensorflow.nn.LayerInfo;

import bda.tensorflow.jni_11.Input;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.lang.reflect.Array;

/** LayerInput : the input data for each layer **/
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY)
@JsonSubTypes({
        @JsonSubTypes.Type(value = LayerInputWithSeq.class, name = "LayerInputWithSeq"),
        @JsonSubTypes.Type(value = LossLayerInput.class, name = "LossLayerInput"),
        @JsonSubTypes.Type(value = PoolLayerInput.class, name = "PoolLayerInput"),
        @JsonSubTypes.Type(value = RNNLayerInput.class, name = "RNNLayerInput"),
})
public class LayerInput implements Serializable{
    @JsonIgnore
    public Input[] input;

    public int[] shape;

    public LayerInput(int[] dimension) {
        this.shape = dimension;
        this.input = null;
    }

    public void setInput(Input[] input){
        this.input = input;
    }

    public void setInput(LayerOutput output){
        this.input = output.output;
    }

    public void addInput(LayerOutput output){
        if  (input == null){
            input = output.output;
        } else {
            Input[] new_input = new Input[input.length + output.output.length];
            System.arraycopy(input, 0, new_input, 0, input.length);
            System.arraycopy(output.output, 0, new_input, input.length, output.output.length);
            input = new_input;
        }
    }

    public LayerInput() {
    }

    public boolean needSeq(){
        return false;
    }

    public void setSeq(Input seq){}

    public String toString(){
        String result = this.getClass().toString() + "\t";
        for (int i : shape){
            result += i + " ";
        }
        return result.trim();
    }

    public static LayerInput parse(String str){
        String[] s = str.split(" ");
        int[] shape = new int[s.length];
        for(int i = 0; i < s.length; i++){
            shape[i] = Integer.parseInt(s[i]);
        }

        return new LayerInput(shape);
    }
}
