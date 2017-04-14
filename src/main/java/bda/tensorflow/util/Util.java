package bda.tensorflow.util;

import bda.tensorflow.jni.Graph;
import bda.tensorflow.jni_11.Input;
import bda.tensorflow.jni_11.Operation;
import bda.tensorflow.jni_11.Scope;
import bda.tensorflow.run.Item;
import bda.tensorflow.run.RNN.RNNMeta;
//import org.apache.commons.cli.Options;
//import org.apache.commons.cli.ParseException;

import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class Util {
    public static String getIdentityName(String variableName) {
        return "identity_" + variableName;
    }

    public static void addAssign(RNNMeta meta, String variableName, int type){
        Scope scope = meta.scope;

        Input place = Operation.Placeholder(type, scope);
        Input var = meta.varMap.get(variableName);
        Operation.Assign(var, place, scope);
    }

    public static void addAssign(RNNMeta meta, int i, int type){
        Scope scope = meta.scope;

        meta.placeholder[i] = Operation.Placeholder(type, scope);
        Input var = meta.variable.get(i);
        meta.assign[i] = Operation.Assign(var, meta.placeholder[i], scope);
    }


    private static native void addAssignNodeAccordingVariableName(Graph paramGraph, String paramString1, String paramString2, String paramString3, int paramInt);
 }
