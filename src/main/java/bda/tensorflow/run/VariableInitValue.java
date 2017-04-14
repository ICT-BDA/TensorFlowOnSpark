package bda.tensorflow.run;

import bda.tensorflow.jni.GraphDef;
import bda.tensorflow.jni.Session;
import bda.tensorflow.jni.Status;
import bda.tensorflow.jni.Tensor;
import bda.tensorflow.jni_11.Input;
import bda.tensorflow.jni_11.Operation;
import bda.tensorflow.jni_11.Scope;

public class VariableInitValue {
    static {
        System.loadLibrary("jt");
    }

    public static float[] getValueFromConfig(InitValueConfig config) {
        int[] shape = config.shape;
        if (config.isConstant) {
            int num = 1;
            for (int s : shape) {
                num *= s;
            }
            float[] array = new float[num];
            float constant = config.constant;
            java.util.Arrays.fill(array, constant);
            return array;
        }
        float mean = config.mean;
        float dev = config.dev;
        String name = config.name;
        Scope scope = new Scope();
        Input const_node = Operation.Const(shape, scope);
        Input truncated_normal = Operation.TruncatedNormal(const_node, scope);
        Input const_stddev = Operation.Const(new float[]{dev}, scope);
        Input const_mean = Operation.Const(new float[]{mean}, scope);
        Input mul = Operation.Mul(truncated_normal, const_stddev, scope);
        Operation.Add(mul, const_mean, scope, name);

        Session session = new Session();
        GraphDef gdf = new GraphDef();
        Status s = new Status();
        scope.toGraphDef(gdf, s);
        session.create(gdf);
        Tensor[] outTensor = new Tensor[1];
        session.run(new String[0], new Tensor[0], new String[]{name}, new String[0], outTensor, s);
        assert (s.ok());
        float[] ret = outTensor[0].toFloatArray();
        return ret;
    }
}