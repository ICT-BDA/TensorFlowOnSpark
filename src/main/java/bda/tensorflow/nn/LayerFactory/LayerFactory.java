package bda.tensorflow.nn.LayerFactory;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by yixuanhe on 9/20/16.
 */
public class LayerFactory {
    static Map<String, Method> nameToMethod = new HashMap<>();
    public static void register(String s, Method method){
        nameToMethod.put(s, method);
    }
    public static Method getMethod(String s){
        return nameToMethod.get(s);
    }
}
