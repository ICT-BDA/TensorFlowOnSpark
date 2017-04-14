package bda.tensorflow.nn.network;

import bda.tensorflow.exception.LayerCreateException;
import bda.tensorflow.nn.LayerFactory.BasicLayerFactory;
import bda.tensorflow.nn.LayerFactory.RNNLayerFactory;
import bda.tensorflow.nn.LayerInfo.*;
import bda.tensorflow.util.Type;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by yixuanhe on 9/20/16.
 */
public class NetworkConfig implements Serializable{
    public List<LayerInfo> layer;

    public Graph graph;

    public int batchSize;
    public int inputSize;
    public int outputSize;

    public int[] inputShape;

    public String appId;

    public int[] getInputShape() {
        return inputShape;
    }

    public void setInputShape(int[] inputShape) {
        this.inputShape = inputShape;
    }

    public int inputType;
    public int outputType;

    public boolean isRNN;
    public boolean hasLoss;

    public String[] master;

    public String[] ps;

    public boolean async = false;

    boolean normalization = true;

    public NetworkConfig() {
    }

    public NetworkConfig(int batchSize, int outputSize, int inputType, int outputType) {
        this.batchSize = batchSize;
        this.outputSize = outputSize;
        this.inputType = inputType;
        this.outputType = outputType;

        this.layer = new ArrayList<>();
        this.isRNN = false;

        this.hasLoss = false;
    }

    public void setInputSize(int inputSize) {
        this.inputSize = inputSize;
    }

    public void addLayer(String layerName, int[] shape) throws LayerCreateException {
        LayerInput input = new LayerInput(shape);
        addLayer(new LayerInfo(layerName, input));
    }

    public void addLossLayer(String layerName, int[] shape) throws LayerCreateException {
        LossLayerInput input = new LossLayerInput(shape);
        addLayer(new LayerInfo(layerName, input));
    }

    public void addRNNLayer(String layerName, int[] shape) throws LayerCreateException {
        RNNLayerInput input = new RNNLayerInput(shape);
        isRNN = true;
        addLayer(new LayerInfo(layerName, input));
    }

    public void addLayer(LayerInfo info) throws LayerCreateException {
        if (hasLoss)
            throw new LayerCreateException("Try to create Layer after create Loss Layer");
        isRNN = true;
        layer.add(info);
    }

    public String toString(){
        ObjectMapper mapper = new ObjectMapper();
        String result = null;
        try {
            result = mapper.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            e.printStackTrace();
        }
        return result;
    }

    public static NetworkConfig parse(String s) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(s, NetworkConfig.class);
    }

    public static void main(String[] args) throws LayerCreateException, IOException {
        int batchSize = 16;
        int outputSize = 9999;
        int inputType = Type.DT_FLOAT;
        int outputType = Type.DT_INT32;

        NetworkConfig config = new NetworkConfig(batchSize, outputSize, inputType, outputType);
        config.addRNNLayer(RNNLayerFactory.BasicRNNLayer, new int[]{20, 1});
        config.addLayer(new LayerInfo(RNNLayerFactory.BasicLayerWithSeq, new LayerInputWithSeq(new int[]{config.outputSize, 20})));
        config.addLossLayer(BasicLayerFactory.LossLayer, new int[]{config.outputSize});

        String json = config.toString();
        System.out.println(json);
    }
}
