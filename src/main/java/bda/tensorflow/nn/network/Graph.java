package bda.tensorflow.nn.network;

import bda.tensorflow.exception.TensorflowRuntimeException;
import bda.tensorflow.jni.GraphDef;
import bda.tensorflow.jni.Status;
import bda.tensorflow.jni_11.ClientSession;
import bda.tensorflow.jni_11.Input;
import bda.tensorflow.jni_11.Operation;
import bda.tensorflow.jni_11.Scope;
import bda.tensorflow.nn.Config;
import bda.tensorflow.nn.LayerFactory.LayerFactory;
import bda.tensorflow.nn.LayerInfo.*;
import bda.tensorflow.run.RNN.RNNMeta;
import bda.tensorflow.util.Type;

import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;

import static bda.tensorflow.nn.network.NetworkUtil.addAssignNode;

/**
 * Created by yixuanhe on 07/12/2016.
 */
public class Graph implements Serializable{
    Map<String, List<String>> edge;
    Map<String, LayerInfo> layers;
    Map<String, Integer> pending;

    String lossLayer;
    public Graph(){
        layers = new HashMap<>();
        edge = new HashMap<>();
        pending =  new HashMap<>();
    }

    public void addLayer(LayerInfo layer, List<String> inputs){
        layers.put(layer.layerName, layer);
        edge.put(layer.layerName, new ArrayList<>());
        pending.put(layer.layerName, 0);
        for (String s : inputs){
            addEdge(layer.layerName, s);
        }

        //TODO: only have one loss, may consider more?
        if (layer.name.contains("Loss"))
            lossLayer = layer.layerName;
    }

    public void addEdge(String dest, String source){
        edge.get(source).add(dest);
        pending.put(dest, pending.get(dest) + 1);
    }

//    public RNNMeta construct(int index, NetworkConfig network, RNNMeta meta, boolean async) throws InvocationTargetException, IllegalAccessException, TensorflowRuntimeException {
//        Scope scope = new Scope();
//
//        meta.shareNodeMap = new HashMap<>();
//        meta.variableMeta = new HashMap<>();
//        meta.varMap = new HashMap<>();
//
//        meta.items = new LinkedList<>();
//        meta.label = new LinkedList<>();
//        meta.variable = new LinkedList<>();
//        meta.data = new LinkedList<>();
//
//        Input[] x = new Input[network.inputSize];
//        for(int i = 0; i < network.inputSize; i++){
//            x[i] = Operation.Placeholder(network.inputType, scope, Config.INPUT + i);
//            meta.data.add(x[i]);
//        }
//        Input[] y = new Input[network.outputSize];
//        for(int i = 0; i < network.outputSize; i++){
//            y[i] = Operation.Placeholder(network.outputType, scope, Config.LABEL + i);
//            meta.label.add(y[i]);
//        }
//
//        meta.seq = Operation.Placeholder(Type.DT_FLOAT, scope, Config.SEQ);
//
//        Input[] res = create(x, y, meta.seq, scope, meta, network);
//
//        meta.loss = res[0];
//        meta.lossgrad = res[1];
//        meta.accu = res[2];
//
//        meta.scope = scope;
//
//        NetworkUtil.getGradient(meta);
//        addAssignNode(meta);
//
//        meta.lr = Operation.Placeholder(Type.DT_FLOAT, scope);
//
//        if (async) {
//            for (int m = 0; m < meta.variable.size(); m++) {
//                Input v = meta.variable.get(m);
//                Input g = meta.gradient[m];
//                Operation.ApplyGradientDescent(v, meta.lr, g, scope, true);
//            }
//        }
//
//        GraphDef def = new GraphDef();
//        Status status = new Status();
//        scope.toGraphDef(def, status);
//        assert (status.ok());
//        meta.session.create(def);
//
//        if(network.master == null) {
//            meta.client = new ClientSession(scope, "");
//        } else {
//            //TODO: might exist better way? worth to explore
//            meta.client = new ClientSession(scope, network.master[index%network.master.length]);
//        }
//        return meta;
//    }
//
//    // we need to return loss, lossgrad and accuracy node
//    public Input[] create(Input[] x, Input[] y, Input seq, Scope scope, RNNMeta meta, NetworkConfig network) throws InvocationTargetException, IllegalAccessException {
//        LayerOutput output = new LayerOutput(x);
//        GraphMetaInfo metaInfo = new GraphMetaInfo(scope, meta, network.batchSize, network.appId, network.ps);
//
//        Map<String, Integer> cur_pending = new HashMap<>(pending);
//        List<String> ready = new ArrayList<>();
//        Input[] ret = new Input[3];
//
//        for(Map.Entry<String, Integer> entry : cur_pending.entrySet()){
//            if (entry.getValue()== 0){
//                LayerInfo info = layers.get(entry.getKey());
//                info.input.setInput(output);
//                ready.add(entry.getKey());
//            }
//        }
//
//        while(!ready.isEmpty()){
//            String l = ready.get(0);
//            ready.remove(0);
//            LayerInfo info = layers.get(l);
//            LayerInput layerInput = info.input;
//            layerInput.setInput(output);
//            if (layerInput.needSeq()) {
//                layerInput.setSeq(seq);
//            }
//            Method create = LayerFactory.getMethod(info.name);
//
//            System.out.println("invoke " + info.layerName);
//
//            if(l == lossLayer){
//                LossLayerInput input = (LossLayerInput)info.input;
//                input.setInput(output);
//                input.y = y;
//                //construct train graph
//                output = (LayerOutput) create.invoke(LayerFactory.class, input, metaInfo, false);
//                LossLayerOutput lossOutput = (LossLayerOutput)output;
//                ret[0] = lossOutput.loss;
//                ret[1] = lossOutput.lossgrad;
//                output = (LayerOutput) create.invoke(LayerFactory.class, input, metaInfo, true);
//                ret[2] = output.output[1];
//                return ret;
//            }
//            output = (LayerOutput) create.invoke(LayerFactory.class, layerInput, metaInfo);
//            for(String e : edge.get(l)){
//                int p = cur_pending.get(e)-1;
//                cur_pending.put(e, p);
//                System.out.println(e + ": " + p);
//
//                layers.get(e).input.addInput(output);
//                if (p == 0){
//                    ready.add(e);
//                }
//            }
//        }
//        return ret;
//    }

    //TODO: can't extract API of graph build because some bug of gradient compute provided by TensorFlow. Try add when TensorFlow c++ code improve
    public RNNMeta construct(int index, NetworkConfig network, RNNMeta meta, boolean async) throws InvocationTargetException, IllegalAccessException, TensorflowRuntimeException {
        //TODO: we current suppose that there is only one input source, extend it to multi-input when have time
        Scope scope = new Scope();

        meta.shareNodeMap = new HashMap<>();
        meta.variableMeta = new HashMap<>();
        meta.varMap = new HashMap<>();

        meta.items = new LinkedList<>();
        meta.label = new LinkedList<>();
        meta.variable = new LinkedList<>();
        meta.data = new LinkedList<>();

        Input[] x = new Input[network.inputSize];
        for(int i = 0; i < network.inputSize; i++){
            x[i] = Operation.Placeholder(network.inputType, scope, Config.INPUT + i);
            meta.data.add(x[i]);
        }
        Input[] y = new Input[network.outputSize];
        for(int i = 0; i < network.outputSize; i++){
            y[i] = Operation.Placeholder(network.outputType, scope, Config.LABEL + i);
            meta.label.add(y[i]);
        }

        Input seq = null;
        if (network.isRNN) {
            seq = Operation.Placeholder(Type.DT_FLOAT, scope, Config.SEQ);
            meta.seq = seq;
        }

        LayerOutput output = new LayerOutput(x);
        GraphMetaInfo metaInfo = new GraphMetaInfo(scope, meta, network.batchSize, network.appId, network.ps);

        Map<String, Integer> cur_pending = new HashMap<>(pending);
        List<String> ready = new ArrayList<>();

        for(Map.Entry<String, Integer> entry : cur_pending.entrySet()){
            if (entry.getValue()== 0){
                LayerInfo info = layers.get(entry.getKey());
                info.input.setInput(output);
                ready.add(entry.getKey());
            }
        }
//
//        for (Map.Entry<String, List<String>> entry : edge.entrySet()){
//            if (entry.getValue().size() == 0){
//                LayerInfo info = layers.get(entry.getKey());
//                info.input.setInput(output);
//                ready.add(entry.getKey());
//            } else {
//                cur_pending.put(entry.getKey(), entry.getValue().size());
//            }
//        }

        while(!ready.isEmpty()){
            String l = ready.get(0);
            ready.remove(0);
            LayerInfo info = layers.get(l);
            LayerInput layerInput = info.input;
            layerInput.setInput(output);
            if (layerInput.needSeq()) {
                layerInput.setSeq(seq);
            }
            Method create = LayerFactory.getMethod(info.name);

            System.out.println("invoke " + info.layerName);

            if(l.equals(lossLayer)){
                LossLayerInput input = (LossLayerInput)info.input;
                input.setInput(output);
                input.y = y;

                //construct train graph
                output = (LayerOutput) create.invoke(LayerFactory.class, input, metaInfo, false);
                LossLayerOutput lossOutput = (LossLayerOutput)output;
                meta.loss = lossOutput.loss;

                if (network.normalization) {
                    for (Input i : meta.shareNodeMap.values()) {
                        meta.loss = Operation.Add(meta.loss, Operation.L2Loss(i, scope), scope);
                    }
                }

                meta.lossgrad = lossOutput.lossgrad;
                meta.scope = scope;

                NetworkUtil.getGradient(meta);
                addAssignNode(meta);

                meta.lr = Operation.Placeholder(Type.DT_FLOAT, scope);

                if (async) {
                    for (int m = 0; m < meta.variable.size(); m++) {
                        Input g = meta.gradient[m];
                        Input v = meta.variable.get(m);
                        Operation.ApplyGradientDescent(v, meta.lr, g, scope, true);
                    }
                }

                output = (LayerOutput) create.invoke(LayerFactory.class, input, metaInfo, true);
                GraphDef def = new GraphDef();
                Status status = new Status();
                scope.toGraphDef(def, status);
                assert (status.ok());
                meta.session.create(def);

                if(network.master == null) {
                    meta.client = new ClientSession(scope, "");
                } else {
                    //TODO: might exist better way? worth to explore
                    meta.client = new ClientSession(scope, network.master[index%network.master.length]);
                }
                return meta;
            }
            output = (LayerOutput) create.invoke(LayerFactory.class, layerInput, metaInfo);
            for(String e : edge.get(l)){
                int p = cur_pending.get(e)-1;
                cur_pending.put(e, p);
                System.out.println(e + ": " + p);

                layers.get(e).input.addInput(output);
                if (p == 0){
                    ready.add(e);
                }
            }
        }
        return meta;
    }
}
