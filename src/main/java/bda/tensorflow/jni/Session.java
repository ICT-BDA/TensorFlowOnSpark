package bda.tensorflow.jni;

public class Session extends BaseObject {
    public Session() {
        allocate();
    }

    private native void allocate();

    protected void deallocate(long address) {
        deallocateMemory(address);
    }

    private native void deallocateMemory(long paramLong);

    public native void create(GraphDef gdef);

    public native void run(String[] input_names, Tensor[] input_tensors, String[] output_names, String[] target_names, Tensor[] outputs, Status status);
}
