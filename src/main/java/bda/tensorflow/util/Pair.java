package bda.tensorflow.util;

import java.io.Serializable;


public class Pair<T, U> implements Serializable {
    public T first;
    public U second;
  
    public Pair(T first, U second) {
        this.first = first;
        this.second = second;
    }
}
