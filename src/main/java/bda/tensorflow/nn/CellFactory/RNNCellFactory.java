package bda.tensorflow.nn.CellFactory;

import bda.tensorflow.jni_11.Operation;
import bda.tensorflow.jni_11.Scope;
import bda.tensorflow.jni_11.Input;
import bda.tensorflow.run.InitValueConfig;
import bda.tensorflow.run.Item;
import bda.tensorflow.run.RNN.RNNCellOutput;
import bda.tensorflow.run.RNN.RNNMeta;
import bda.tensorflow.util.Type;
import bda.tensorflow.util.Util;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.Map;

public class RNNCellFactory {
    /**
     * create variable node which need to update
     * @param key node name
     * @param type node type
     * @param shape node shape
     * @param meta meta
     * @param scope scope
     * @return return created Node
     */
    public static Input createNode(String key, int type, int[] shape, RNNMeta meta, Scope scope, String container, String[] ps) {
        Map<String, Input> map = meta.shareNodeMap;
        Map<String, Item> vMap = meta.variableMeta;
        Input node;

        // RNN need to share variable, so we need to check map and return if exists, else create one
        if (map.containsKey(key)) {
            node = map.get(key);
        } else {
            Input n;
            if (ps == null) {
                n = Operation.Variable(shape, type, scope, key, container);
            } else {
                int len = ps.length;
                //TODO: this implentation is quiet simply, leave some space to improve
                n = Operation.Variable(shape, type, scope, key, ps[meta.variable.size() % len]);
            }

            node = Operation.Identity(n, scope, Util.getIdentityName(key));

            map.put(key, node);
            meta.varMap.put(key, n);
            meta.variable.add(n);

            Item item = new Item(key, type, shape, new InitValueConfig(key, false, 0.0F, 0.0F, 0.1F, type, shape));

            meta.items.add(item);

            vMap.put(key, item);
        }

        return node;
    }

    /**
     * Create op w*[x 0/1] + u*s, [x 0/1] is augmented matrix. Because we still need fake input when it's no input
     * so we need to change 1 to 0 sometimes
     *
     * @param wKey w name
     * @param uKey u name
     * @param wType w tye
     * @param uType u type
     * @param wShape w shape
     * @param uShape u shape
     * @param scope scope
     * @param meta meta
     * @param input x node
     * @param state s node
     * @param seq seq node, which store value we used to create augmented matrix
     * @param time current node number, used to create augmented matrix
     * @return result node
     */
    private static Input createBasicOp(String wKey, String uKey, int wType, int uType, int[] wShape, int[] uShape,
                                      Scope scope, RNNMeta meta, Input input, Input state, Input seq, int time,
                                       String container, String[] ps) {
        Input w = createNode(wKey, wType, wShape, meta, scope, container, ps);
        Input bias = Operation.Slice(seq, Operation.Const(new int[]{time, 0}, scope), Operation.Const(new int[]{1, -1}, scope), scope);
//        Node bias = Operation.Slice(seq, Operation.Const(new int[]{12, 21}, scope), Operation.Const(new int[]{1, -1}, scope), scope);
        Input x = Operation.Concat(Operation.Const(0, scope), new Input[]{bias, input}, scope);

        Input wMul = Operation.MatMul(w, Operation.Identity(x, scope), scope);

        if (state != null) {
            Input u = createNode(uKey, uType, uShape, meta, scope, container, ps);
            Input uMul = Operation.MatMul(u, Operation.Identity(state, scope), scope);
            return Operation.Add(wMul, uMul, scope);
        }

        return wMul;
    }

    /**
     * create a basic rnn
     * @param input input node
     * @param state state node
     * @param scope scope
     * @param meta mete
     * @param activation active function, current support relu and tanh
     * @param shape int array
     * @param layer current number
     * @param seq seq node
     * @return rnn cell output
     * @throws NoSuchMethodException
     * @throws InvocationTargetException
     * @throws IllegalAccessException
     */
    public static RNNCellOutput createBasicRNNCell(Input input, Input state, Scope scope, RNNMeta meta,
                                                   String activation, int[] shape, int layer, Input seq,
                                                   String container, String[] ps)
            throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        Method activeFunc = Operation.class.getMethod(activation, Input.class, Scope.class);

        String prefix = "rnn-basic-";
        String wKey = prefix + "w" + 0;
        String uKey = prefix + "u" + 0;

        int[] ushape = {shape[0], shape[0]};
        int type = Type.DT_FLOAT;

        Input bias_add = createBasicOp(wKey, uKey, type, type, shape, ushape, scope, meta, input, state, seq, layer, container, ps);


        Input active = (Input) activeFunc.invoke(new Operation(), bias_add, scope);
        return new RNNCellOutput(active, active);
    }

    public static RNNCellOutput createLSTMRNNCell(Input input, Input state, Input c_state, Scope scope, RNNMeta meta,
                                                  String activation, int[] shape, int layer, Input seq, String container, String[] ps)
            throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        Method activeFunc = Operation.class.getMethod(activation, Input.class, Scope.class, String.class);

        String prefix = "rnn-LSTM-";
        String[] type = {"i", "f", "o", "g"};
        int types = type.length;
        Input[] nodes = new Input[types];

        for (int j = 0; j < types; j++) {
            String t = type[j];
            String wKey = prefix + "w" + t + layer;
            String uKey = prefix + "u" + t + layer;

            int[] ushape = {shape[0], shape[0]};

            int wType = Type.DT_FLOAT;
            int uType = Type.DT_FLOAT;

            Input bias_add = createBasicOp(wKey, uKey, wType, uType, shape, ushape, scope, meta, input, state, seq, layer, container, ps);

            Input active;
            if (j != 4)
                active = (Input) activeFunc.invoke(new Operation(), bias_add, scope, t + "active" + t + layer);
            else
                active = Operation.Tanh(bias_add, scope);
            nodes[j] = active;
        }

        Input giMul = Operation.Mul(nodes[0], nodes[3], scope);
        Input cfMul = Operation.Mul(c_state, nodes[1], scope);
        Input c_add = Operation.Add(giMul, cfMul, scope);
        Input tanh = Operation.Tanh(c_add, scope);
        Input s = Operation.Mul(tanh, nodes[2], scope);

        return new RNNCellOutput(s, c_add);
    }

    public static RNNCellOutput createGRURNNCell(Input input, Input state, Scope scope, RNNMeta meta, String activation,
                                                 int[] shape, int layer, Input seq, String container, String[] ps)
            throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        Method activeFunc = new Operation().getClass().getMethod(activation, Input.class, Scope.class, String.class);

        String prefix = "rnn-LSTM-";
        String[] type = {"z", "r"};
        int types = type.length;
        Input[] nodes = new Input[types];

        int[] ushape = {shape[0], shape[0]};

        for (int j = 0; j < types; j++) {
            String t = type[j];
            String wKey = prefix + "w" + t + layer;
            String uKey = prefix + "u" + t + layer;

            int wType = Type.DT_FLOAT;
            int uType = Type.DT_FLOAT;

            Input bias_add = createBasicOp(wKey, uKey, wType, uType, shape, ushape, scope, meta, input, state, seq, layer, container, ps);

            Input active = (Input) activeFunc.invoke(new Operation(), bias_add, scope, t + "active");
            nodes[j] = active;
        }

        int wType = Type.DT_FLOAT;
        int uType = Type.DT_FLOAT;


        String t = "h";
        String wKey = prefix + "w" + t + layer;
        String uKey = prefix + "u" + t + layer;
        Input u;
        Input w;
        if (meta.shareNodeMap.containsKey(wKey)) {
            w = meta.shareNodeMap.get(wKey);
            u = meta.shareNodeMap.get(uKey);
        } else {
            w = Operation.Variable(shape, wType, scope, wKey, container);
            u = Operation.Variable(ushape, uType, scope, uKey, container);

            meta.shareNodeMap.put(wKey, w);
            meta.shareNodeMap.put(uKey, u);

            meta.variableMeta.put(wKey, new Item(wKey, wType, shape, new InitValueConfig(wKey, false, 0.0F, 0.0F, 0.1F, wType, shape)));
            meta.variableMeta.put(uKey, new Item(uKey, uType, shape, new InitValueConfig(uKey, false, 0.0F, 0.0F, 0.1F, uType, ushape)));
        }

        Input uxMul = Operation.MatMul(input, u, scope, "uxMul");
        Input srMul = Operation.Mul(state, nodes[1], scope, "srMul");
        Input wMul = Operation.MatMul(srMul, w, scope, "wMul");
        Input uwAdd = Operation.Add(uxMul, wMul, scope, "uwAdd");
        Input h = Operation.Tanh(uwAdd, scope, "hActive");
        Input zsMul = Operation.Mul(nodes[0], state, scope, "zsMul");

        int[] all_one = new int[shape[0]];
        Arrays.fill(all_one, 1);
        Input cutZ = Operation.Sub(Operation.Const(all_one, scope, ""), nodes[0], scope, "1-z");
        Input zhMul = Operation.Mul(zsMul, cutZ, scope, "zhMul");
        Input s = Operation.Add(zsMul, zhMul, scope, "final_add");

        return new RNNCellOutput(s, s);
    }
}
