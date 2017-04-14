package bda.tensorflow.run.RNN;

import bda.tensorflow.jni_11.Input;
import bda.tensorflow.run.Meta;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class RNNMeta extends Meta implements Serializable{
    public Map<String, Input> shareNodeMap = new HashMap();
}
