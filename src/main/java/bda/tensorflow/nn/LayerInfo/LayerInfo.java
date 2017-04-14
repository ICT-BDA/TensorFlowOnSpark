package bda.tensorflow.nn.LayerInfo;

import bda.tensorflow.exception.LayerCreateException;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.Serializable;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by yixuanhe on 9/20/16.
 */
public class LayerInfo implements Serializable{
    public String layerName;
    public String name;
    public LayerInput input;

    public LayerInfo(String layerName, String name, LayerInput input) {
        this.layerName = layerName;
        this.name = name;
        this.input = input;
    }

    public LayerInfo(String name, LayerInput input) throws LayerCreateException {
        this.name = name;
        this.input = input;
    }

    // This is invoked by jackson, not use it yourself!!!!!!!
    //TODO: To serialize, we add lots of non-parameter constructor. we need to prevent user invoked it
    public LayerInfo() {
    }
}
