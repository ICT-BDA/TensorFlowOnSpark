package bda.tensorflow.exception;

/** Exception when tensorflow running error **/
public class TensorflowRuntimeException extends Throwable{
    String message;

    public TensorflowRuntimeException(String message){
        this.message = message;
    }

    public String getMessage(){
        return "TensorFlow Runtime Error : " + this.message;
    }
}
