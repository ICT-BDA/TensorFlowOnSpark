package bda.tensorflow.jni;

public class Graph extends BaseObject {
    public Graph(GraphDef gd) {
        allocate(gd);
    }

    private native void allocate(GraphDef paramGraphDef);
  
    protected void deallocate(long address) {
        deallocateMemory(address);
    }
  
    private native void deallocateMemory(long paramLong);
}
