package bda.tensorflow.jni;

public class GraphDef extends BaseObject {
    public GraphDef(){
        allocate();
    }
  
    private native void allocate();
  
    protected void deallocate(long address){
        deallocateMemory(address);
    }
  
    private native void deallocateMemory(long paramLong);
}