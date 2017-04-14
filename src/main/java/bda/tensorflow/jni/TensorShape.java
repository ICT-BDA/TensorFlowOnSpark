package bda.tensorflow.jni;

public class TensorShape extends BaseObject {
    public TensorShape(int[] ts) {
        allocate(ts);
    }

    private native void allocate(int[] paramArrayOfInt);

    protected void deallocate(long address) {
        deallocateMemory(address);
    }

    private native void deallocateMemory(long paramLong);
}
