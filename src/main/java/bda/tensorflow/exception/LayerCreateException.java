package bda.tensorflow.exception;

/**
 * Created by yixuanhe on 9/19/16.
 */
public class LayerCreateException extends Throwable {
    String message;

    public LayerCreateException(String message){
        this.message = message;
    }

    public String getMessage(){
        return "Layer Create Error : " + message;
    }
}
