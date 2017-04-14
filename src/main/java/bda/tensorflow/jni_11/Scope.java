package bda.tensorflow.jni_11;

import bda.tensorflow.jni.BaseObject;
import bda.tensorflow.jni.GraphDef;
import bda.tensorflow.jni.Status;

import java.io.Serializable;

/**
 * Created by yixuanhe on 17/10/2016.
 */
public class Scope  extends BaseObject implements Serializable{
    public Scope(){
        allocate();
    }

    @Override
    protected void deallocate(long address) {
        deallocateMemory(address);
    }

    private native void deallocateMemory(long paramLong);

    private native void allocate();

    public native void toGraphDef(GraphDef gdef, Status status);
}
