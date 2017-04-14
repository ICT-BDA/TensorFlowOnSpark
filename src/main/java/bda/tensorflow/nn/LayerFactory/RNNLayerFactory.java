package bda.tensorflow.nn.LayerFactory;

import bda.tensorflow.jni_11.Operation;
import bda.tensorflow.jni_11.Input;
import bda.tensorflow.jni_11.Scope;
import bda.tensorflow.nn.CellFactory.RNNCellFactory;
import bda.tensorflow.nn.LayerInfo.*;
import bda.tensorflow.run.RNN.RNNCellOutput;
import bda.tensorflow.util.Type;

import java.lang.reflect.InvocationTargetException;

public class RNNLayerFactory extends LayerFactory{
    public static String BasicRNNLayer = "createBasicRNNLayer";
    public static String BasicLayerWithSeq = "createBasicLayerWithSeq";
    static {
        try {
            LayerFactory.register("createBasicRNNLayer", RNNLayerFactory.class.getMethod("createBasicRNNLayer",
                    RNNLayerInput.class, GraphMetaInfo.class));
            LayerFactory.register("createBasicLayerWithSeq", RNNLayerFactory.class.getMethod("createBasicLayerWithSeq",
                    LayerInputWithSeq.class, GraphMetaInfo.class));
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }
    }

    public static LayerOutput createBasicRNNLayer(RNNLayerInput input, GraphMetaInfo metaInfo)
            throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        int len = input.input.length;
        Input state = null;
        Scope scope = metaInfo.scope;
        int[] wShape = {input.shape[0], input.shape[1] + 1};

        Input[] out = new Input[len];

        for(int i = 0; i < len; i++){
            RNNCellOutput rnnCellOutput = RNNCellFactory.createBasicRNNCell(input.input[i], state, scope, metaInfo.meta, "Relu", wShape, i, input.seq, metaInfo.appId, metaInfo.ps);
            state = rnnCellOutput.state;
            out[i] = rnnCellOutput.output;
        }
        return new LayerOutput(out);
    }

    public static LayerOutput createBasicLayerWithSeq(LayerInputWithSeq input, GraphMetaInfo metaInfo){
        int len = input.input.length;
        Input[] node = new Input[len];
        Scope scope = metaInfo.scope;

        int[] shape = new int[]{input.shape[0], input.shape[1]+1};
        for(int i = 0; i < len; i++){
            String wkey = "BasicLayer-seq-w" + i;
            int type = Type.DT_FLOAT;

            Input b = Operation.Slice(input.seq, Operation.Const(new int[]{i, 0}, scope), Operation.Const(new int[]{1, -1}, scope), scope);
            Input x = Operation.Concat(Operation.Const(0, scope), new Input[]{b, input.input[i]}, scope);

            Input w = RNNCellFactory.createNode(wkey, type, shape, metaInfo.meta, scope, metaInfo.appId, metaInfo.ps);
            node[i] = Operation.MatMul(w, x, scope);
        }

        return new LayerOutput(node);
    }

}
