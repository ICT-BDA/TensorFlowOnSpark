package bda.tensorflow.nn.LayerFactory;

import bda.tensorflow.jni_11.Input;
import bda.tensorflow.jni_11.Operation;
import bda.tensorflow.jni_11.Scope;
import bda.tensorflow.nn.CellFactory.RNNCellFactory;
import bda.tensorflow.nn.Config;
import bda.tensorflow.nn.LayerInfo.*;
import bda.tensorflow.util.Type;

public class BasicLayerFactory extends LayerFactory{
    public static String LossLayer = "createLossLayer";
    public static String BasicLayer = "createBasicLayer";
    public static String MaxPooolLayer = "createMaxPooolLayer";
    public static String ExpandLayers = "createExpandLayer";
    public static String ConvLayer = "createConvLayer";
    public static String FlatLayer = "createFlatLayer";

    static {
        try {
            LayerFactory.register("createLossLayer", BasicLayerFactory.class.getMethod("createLossLayer",
                    LossLayerInput.class, GraphMetaInfo.class, boolean.class));
            LayerFactory.register("createBasicLayer", BasicLayerFactory.class.getMethod("createBasicLayer",
                    LayerInput.class, GraphMetaInfo.class));
            LayerFactory.register("createMaxPooolLayer", BasicLayerFactory.class.getMethod("createMaxPooolLayer",
                    PoolLayerInput.class, GraphMetaInfo.class));
            LayerFactory.register("createExpandLayer", BasicLayerFactory.class.getMethod("createExpandLayer",
                    LayerInput.class, GraphMetaInfo.class));
            LayerFactory.register("createConvLayer", BasicLayerFactory.class.getMethod("createConvLayer",
                    ConvLayerInput.class, GraphMetaInfo.class));
            LayerFactory.register("createFlatLayer", BasicLayerFactory.class.getMethod("createFlatLayer",
                    LayerInput.class, GraphMetaInfo.class));
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }
    }

    public static LossLayerOutput createLossLayer(LossLayerInput input, GraphMetaInfo metaInfo, boolean ispredict){
        int len = input.input.length;
        Scope scope = metaInfo.scope;

        if (ispredict){
            Input[] result = new Input[len];
            Input[] ret = new Input[2];
            Input[] equals = new Input[len];
            for (int i = 0; i < len; i++) {
                Input pre = Operation.ArgMax(input.input[i], Operation.Const(0, scope), scope);
                result[i] = Operation.Cast(pre, Type.DT_INT32, scope, Config.PREDICT + i);
                equals[i] = Operation.Cast(Operation.Equal(result[i], input.y[i], scope), Type.DT_FLOAT, scope);
            }

            Input concat;
            if (len > 1) {
                concat = Operation.Concat(Operation.Const(0, scope), equals, scope);
            } else {
                concat = equals[0];
            }

            metaInfo.meta.accuracy = Config.ACCURACY;
            metaInfo.meta.accu = Operation.Mean(Operation.Reshape(concat, Operation.Const(new int[]{-1}, scope), scope), Operation.Const(0, scope), scope, Config.ACCURACY);

            ret[0] = concat;
            ret[1] = metaInfo.meta.accu;

            return new LossLayerOutput(ret);

        } else {
            Input[] loss = new Input[len];

            for (int i = 0; i < len; i++) {
                Input tran = Operation.Transpose(input.input[i], Operation.Const(new int[]{1, 0}, scope), scope);
                loss[i] = Operation.Reshape(Operation.SparseSoftmaxCrossEntropyWithLogits(tran, input.y[i]
                        , scope), Operation.Const(new int[]{-1}, scope), scope);
            }

            Input concat;
            if (len == 1) {
                concat = loss[0];
            } else {
                concat = Operation.Concat(Operation.Const(0, scope), loss, scope);
            }
            Input mean_loss = Operation.Mean(Operation.Reshape(concat, Operation.Const(new int[]{-1}, scope), scope), Operation.Const(new int[]{0}, scope), scope);

            String lossName = Config.LOSS;
            Input l = Operation.Identity(mean_loss, scope, lossName);
            String lossGradName = Config.LOSSGRAD;
            Input lossgrad = Operation.Const(new float[]{1.0F}, scope, lossGradName);

            return new LossLayerOutput(input.input, l, lossgrad);
        }
    }

    public static LayerOutput createBasicLayer(LayerInput input, GraphMetaInfo metaInfo){
        int len = input.input.length;
        Input[] node = new Input[len];
        Scope scope = metaInfo.scope;

        for(int i = 0; i < len; i++){
            String wkey = "BasicLayer-w" + i;
            String bkey = "BasicLayer-b" + i;
            int type = Type.DT_FLOAT;
            int[] bshape = new int[]{input.shape[0], 1};

            Input w = RNNCellFactory.createNode(wkey, type, input.shape, metaInfo.meta, scope, metaInfo.appId, metaInfo.ps);
            Input mul = Operation.MatMul(w, input.input[i], scope);
            Input b = RNNCellFactory.createNode(bkey, type, bshape, metaInfo.meta, scope, metaInfo.appId, metaInfo.ps);
            Input add = Operation.Add(mul, b, scope);
            node[i] = Operation.Relu(add, scope);
        }

        return new LayerOutput(node);
    }

    public static LayerOutput createExpandLayer(LayerInput input, GraphMetaInfo metaInfo){
        int len = input.input.length;
        Input[] node = new Input[len];
        Scope scope = metaInfo.scope;

        for(int i = 0; i < len; i++){
            node[i] = Operation.ExpandDims(input.input[i], Operation.Const(input.shape[0], scope), scope);
        }
        return new LayerOutput(node);
    }

    public static LayerOutput createMaxPooolLayer(PoolLayerInput input, GraphMetaInfo metaInfo){
        int len = input.input.length;
        Input[] node = new Input[len];
        Scope scope = metaInfo.scope;

        for(int i = 0; i < len; i++){
            Input n = Operation.Transpose(input.input[i], Operation.Const(new int[]{2,0,1,3}, scope), scope);
            Input r = Operation.MaxPool(n, input.ksize, input.strides, input.padding, scope);
            node[i] = Operation.Transpose(r, Operation.Const(new int[]{1,2,3,0}, scope), scope);
        }

        return new LayerOutput(node);
    }

    public static LayerOutput createConvLayer(ConvLayerInput input, GraphMetaInfo metaInfo){
        int len = input.input.length;
        Input[] node = new Input[len];
        Scope scope = metaInfo.scope;

        for(int i = 0; i < len; i++){
            String wkey = "Conv-w";
            // we need to put batchsize in first dimension
            Input n = Operation.Transpose(input.input[i], Operation.Const(new int[]{2,0,1,3}, scope), scope);
            Input w = RNNCellFactory.createNode(wkey, Type.DT_FLOAT, input.filter, metaInfo.meta, scope, metaInfo.appId, metaInfo.ps);

            Input r = Operation.Conv2D(n, w, input.strides, input.padding, scope);

            node[i] = Operation.Transpose(r, Operation.Const(new int[]{1,2,3,0}, scope), scope);
        }

        return new LayerOutput(node);
    }

    public static LayerOutput createFlatLayer(LayerInput input, GraphMetaInfo metaInfo){
        int len = input.input.length;
        Input[] node = new Input[len];
        Scope scope = metaInfo.scope;

        for(int i = 0; i < len; i++){
            node[i] = Operation.Reshape(input.input[i], Operation.Const(input.shape, scope), scope);
        }

        return new LayerOutput(node);
    }
}
