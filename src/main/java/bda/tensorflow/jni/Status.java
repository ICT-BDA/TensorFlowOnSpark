package bda.tensorflow.jni;

public class Status extends BaseObject {
    public Status() {
        allocate();
    }

    private native void allocate();

    public native boolean ok();

    public native String errorMessage();

    protected void deallocate(long address) {
        deallocateMemory(address);
    }

    private native void deallocateMemory(long paramLong);
}