package bda.tensorflow.jni;

public class Tensor extends BaseObject {
    public Tensor(int type, TensorShape ts) {
        allocate(type, ts);
    }

    public Tensor() {
    }

    private native void allocate(int paramInt, TensorShape paramTensorShape);

    protected void deallocate(long address) {
        deallocateMemory(address);
    }

    private native void deallocateMemory(long paramLong);

    public native void showContent();

    public native float[] toFloatArray();

    public native void initFromFloatArray(float[] paramArrayOfFloat);

    public native long[] toLongArray();

    public native void initFromLongArray(long[] paramArrayOfLong);

    public native int[] getTensorShape();

    public native boolean[] toBooleanArray();

    public native void initFromIntArray(int[] paramArrayOfInt);
}
