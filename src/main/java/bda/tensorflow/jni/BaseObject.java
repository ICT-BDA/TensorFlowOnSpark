package bda.tensorflow.jni;

public abstract class BaseObject {
    protected long address = 0L;

    public BaseObject() {
        this(0L);
    }

    public BaseObject(long native_addr) {
        this.address = native_addr;
    }

    void init(long allocatedAddress) {
        this.address = allocatedAddress;
    }

    protected void finalize() {
        deallocate();
    }

    public synchronized void deallocate() {
        if (this.address != 0L) {
            deallocate(this.address);
            this.address = 0L;
        }
    }

    protected abstract void deallocate(long paramLong);

}
