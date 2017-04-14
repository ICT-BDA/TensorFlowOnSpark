package bda.tensorflow.nn.LayerInfo;

import bda.tensorflow.jni_11.Input;

/** LayerOutput: the output data for a layer **/
public class LayerOutput {
    public Input[] output;

    public LayerOutput(Input[] output) {
        this.output = output;
    }
}
