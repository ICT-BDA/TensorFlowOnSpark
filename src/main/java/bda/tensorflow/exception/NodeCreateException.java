package bda.tensorflow.exception;

/** Exception when create tensorflow node  **/
public class NodeCreateException extends Throwable {
    String message;

    public NodeCreateException(String message){
        this.message = message;
    }

    public String getMessage(){
        return "Node Create Error : " + message;
    }
}
