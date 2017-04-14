package bda.tensorflow.jni_11;

import bda.tensorflow.jni.BaseObject;
import bda.tensorflow.jni.Tensor;

/**
 * Created by yixuanhe on 10/11/2016.
 */
public class ClientSession extends BaseObject {
    public ClientSession(Scope scope, String master){
        allocate(scope, master);
    }

    public native void allocate(Scope scope, String master);

    @Override
    protected void deallocate(long paramLong) {
        deallocateMemory(paramLong);
    }

    private native void deallocateMemory(long paramLong);

    // run without tensorboard
    public native void run(Input[] input, Tensor[] t, Input[] fetch, Tensor[] output);

    // run with tensorboard
    public native void run(Input[] input, Tensor[] t, Input[] fetch, Tensor[] output, Input summary);
}
