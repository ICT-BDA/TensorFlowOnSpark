package bda.tensorflow.jni_11;

import bda.tensorflow.jni.Graph;
import bda.tensorflow.jni.TensorShape;

/**
 * Created by yixuanhe on 17/10/2016.
 */
public class Operation {
    public static native Input Variable(int[] shape,
                                        int type,
                                        Scope scope,
                                        String name,
                                        String container);

    public static native Input Variable(int[] shape,
                                        int type,
                                        Scope scope,
                                        String name,
                                        String container,
                                        String device);

    public static native Input Identity(Input nodeOut,
                                        Scope scope,
                                        String name);

    public static native Input Placeholder(int type,
                                           Scope scope,
                                           String name);

    public static native Input Transpose(Input tensor,
                                         Input permution,
                                         Scope scope,
                                         String name);

    public static native Input Conv2D(Input input,
                                      Input filter,
                                      int[] strides,
                                      String padding,
                                      Scope scope,
                                      String name);

    public static native Input BiasAdd(Input value,
                                       Input bias,
                                       Scope scope,
                                       String name);

    public static native Input Relu(Input features,
                                    Scope scope,
                                    String name);

    public static native Input MaxPool(Input input,
                                       int[] ksize,
                                       int[] strides,
                                       String padding,
                                       Scope scope,
                                       String name);

    public static native Input MatMul(Input a,
                                      Input b,
                                      Scope scope,
                                      String name);

    public static native Input Add(Input x,
                                   Input y,
                                   Scope scope,
                                   String name);

    public static native Input Mul(Input x,
                                   Input y,
                                   Scope scope,
                                   String name);

    public static native Input Sub(Input x,
                                   Input y,
                                   Scope scope,
                                   String name);

    public static native Input L2Loss(Input t,
                                      Scope scope,
                                      String name);

    public static native Input Mean(Input input,
                                    Input reductionIndice,
                                    Scope scope,
                                    String name);

    public static native Input SparseSoftmaxCrossEntropyWithLogits(Input features,
                                                                   Input labels,
                                                                   Scope scope,
                                                                   String name);

    public static native Input Reshape(Input tensor,
                                       Input shape,
                                       Scope scope,
                                       String name);

    public static native Input Const(int[] t,
                                     TensorShape ts,
                                     Scope scope,
                                     String name);

    public static native Input Const(float[] t,
                                     TensorShape ts,
                                     Scope scope,
                                     String name);

    //the float version
    public static native Input TruncatedNormal(Input shape,
                                               Scope scope,
                                               String name);

    public static native Input Assign(Input ref,
                                      Input value,
                                      Scope scope,
                                      String name);

    public static native Input AssignSub(Input ref,
                                         Input value,
                                         Scope scope,
                                         String name);

    public static native Input[] AddSymbolicGradients(Input loss, Input grad, Input[] inputs, Scope scope);

    public static native Input InTopK(Input predictions,
                                      Input targets,
                                      int k,
                                      Scope scope,
                                      String name);

    public static native Input Gather(Input params,
                                      Input indices,
                                      boolean validate_indices,
                                      Scope scope,
                                      String name);

    public static native Input ExpandDims(Input input,
                                          Input dim,
                                          Scope scope,
                                          String name);

    public static native Input Concat(Input concat_dim,
                                      Input[] values,
                                      Scope scope,
                                      String name);

    public static native Input Equal(Input x,
                                     Input y,
                                     Scope scope,
                                     String name);

    public static native Input ArgMax(Input input,
                                      Input dimension,
                                      Scope scope,
                                      String name);

    public static native Input Cast(Input input,
                                    int type,
                                    Scope scope,
                                    String name);

    public static native Input Tanh(Input features,
                                    Scope scope,
                                    String name);

    public static native Input Slice(Input input,
                                     Input begin,
                                     Input size,
                                     Scope scope,
                                     String name);

    public static native Input ApplyGradientDescent(Input input,
                                                    Input alpha,
                                                    Input delta,
                                                    Scope scope,
                                                    boolean usingLock,
                                                    String name);

    public static Input ApplyGradientDescent(Input input,
                                             Input alpha,
                                             Input delta,
                                             Scope scope,
                                             boolean usingLock) {
        return ApplyGradientDescent(input, alpha, delta, scope, usingLock, "ApplyGradientDescent");
    }

    public static Input ApplyGradientDescent(Input input,
                                             Input alpha,
                                             Input delta,
                                             Scope scope) {
        return ApplyGradientDescent(input, alpha, delta, scope, false, "ApplyGradientDescent");
    }

    public static Input Identity(Input nodeOut,
                                 Scope scope) {
        return Identity(nodeOut, scope, "MyIdentity");
    }

    public static Input Placeholder(int type,
                                    Scope scope) {
        return Placeholder(type, scope, "Placeholder");
    }

    public static Input Transpose(Input tensor,
                                  Input permution,
                                  Scope scope) {
        return Transpose(tensor, permution, scope, "Transpose");
    }

    public static Input Conv2D(Input input,
                               Input filter,
                               int[] strides,
                               String padding,
                               Scope scope) {
        return Conv2D(input, filter, strides, padding, scope, "Conv2D");
    }

    public static Input BiasAdd(Input value,
                                Input bias,
                                Scope scope) {
        return BiasAdd(value, bias, scope, "BiasAdd");
    }

    public static Input Relu(Input features,
                             Scope scope) {
        return Relu(features, scope, "Relu");
    }

    public static Input MaxPool(Input input,
                                int[] ksize,
                                int[] strides,
                                String padding,
                                Scope scope) {
        return MaxPool(input, ksize, strides, padding, scope, "MaxPool");
    }

    public static Input MatMul(Input a,
                               Input b,
                               Scope scope) {
        return MatMul(a, b, scope, "MyMatMul");
    }

    public static Input Add(Input x,
                            Input y,
                            Scope scope) {
        return Add(x, y, scope, "Add");
    }

    public static Input Mul(Input x,
                            Input y,
                            Scope scope) {
        return Mul(x, y, scope, "Mul");
    }

    public static Input Sub(Input x,
                            Input y,
                            Scope scope) {
        return Sub(x, y, scope, "Sub");
    }

    public static Input L2Loss(Input t,
                               Scope scope) {
        return L2Loss(t, scope, "L2Loss");
    }

    public static Input Mean(Input input,
                             Input reductionIndice,
                             Scope scope) {
        return Mean(input, reductionIndice, scope, "Mean");
    }

    public static Input SparseSoftmaxCrossEntropyWithLogits(Input features,
                                                            Input labels,
                                                            Scope scope) {
        return SparseSoftmaxCrossEntropyWithLogits(features, labels, scope, "SparseSoftmaxCrossEntropyWithLogits");
    }

    public static Input Reshape(Input tensor,
                                Input shape,
                                Scope scope) {
        return Reshape(tensor, shape, scope, "MyReshape");
    }

    public static Input Const(int[] t,
                              TensorShape ts,
                              Scope scope) {
        return Const(t, ts, scope, "Const");
    }

    public static Input Const(int[] t,
                              Scope scope) {
        TensorShape ts = new TensorShape(new int[]{t.length});
        return Const(t, ts, scope, "Const");
    }

    public static Input Const(int[] t,
                              Scope scope,
                              String name) {
        TensorShape ts = new TensorShape(new int[]{t.length});
        return Const(t, ts, scope, name);
    }

    public static Input Const(int t,
                              Scope scope) {
        TensorShape ts = new TensorShape(new int[]{});
        return Const(new int[]{t}, ts, scope, "Const");
    }

    public static Input Const(float t,
                              Scope scope) {
        TensorShape ts = new TensorShape(new int[]{});
        return Const(new float[]{t}, ts, scope, "Const");
    }

    public static Input Const(float[] t,
                              Scope scope,
                              String name) {
        TensorShape ts = new TensorShape(new int[]{t.length});
        return Const(t, ts, scope, name);
    }

    public static Input Const(float[] t,
                              TensorShape ts,
                              Scope scope) {
        return Const(t, ts, scope, "Const");
    }

    public static Input Const(float[] t,
                              Scope scope) {
        TensorShape ts = new TensorShape(new int[]{t.length});
        return Const(t, ts, scope, "Const");
    }

    //the float version
    public static Input TruncatedNormal(Input shape,
                                        Scope scope) {
        return TruncatedNormal(shape, scope, "TruncatedNormal");
    }

    public static Input Assign(Input ref,
                               Input value,
                               Scope scope) {
        return Assign(ref, value, scope, "Assign");
    }

    public static Input AssignSub(Input ref,
                                  Input value,
                                  Scope scope) {
        return AssignSub(ref, value, scope, "AssignSub");
    }

    public static Input InTopK(Input predictions,
                               Input targets,
                               int k,
                               Scope scope) {
        return InTopK(predictions, targets, k, scope, "InTopK");
    }

    public static Input Gather(Input params,
                               Input indices,
                               boolean validate_indices,
                               Scope scope) {
        return Gather(params, indices, validate_indices, scope, "Gather");
    }

    public static Input ExpandDims(Input input,
                                   Input dim,
                                   Scope scope) {
        return ExpandDims(input, dim, scope, "ExpandDims");
    }

    public static Input Concat(Input concat_dim,
                               Input[] values,
                               Scope scope) {
        return Concat(concat_dim, values, scope, "Concat");
    }

    public static Input Equal(Input x,
                              Input y,
                              Scope scope) {
        return Equal(x, y, scope, "Equal");
    }

    public static Input ArgMax(Input input,
                               Input dimension,
                               Scope scope) {
        return ArgMax(input, dimension, scope, "ArgMax");
    }

    public static Input Cast(Input input,
                             int type,
                             Scope scope) {
        return Cast(input, type, scope, "Cast");
    }

    public static Input Tanh(Input features,
                             Scope scope) {
        return Tanh(features, scope, "Tanh");
    }

    public static Input Slice(Input input,
                              Input begin,
                              Input size,
                              Scope scope) {
        return Slice(input, begin, size, scope, "Slice");
    }
}
