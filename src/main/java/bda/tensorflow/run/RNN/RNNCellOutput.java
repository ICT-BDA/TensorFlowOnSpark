package bda.tensorflow.run.RNN;

import bda.tensorflow.jni_11.Input;

public class RNNCellOutput {
    public Input state;
    public Input output;

    public RNNCellOutput(Input state, Input output) {
        this.state = state;
        this.output = output;
    }
}