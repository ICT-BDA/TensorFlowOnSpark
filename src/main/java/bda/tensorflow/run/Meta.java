package bda.tensorflow.run;

import bda.tensorflow.jni.*;
import bda.tensorflow.jni_11.ClientSession;
import bda.tensorflow.jni_11.Input;
import bda.tensorflow.jni_11.Scope;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

public class Meta implements Serializable {
    public Session session = new Session();
    public Map<String, Item> variableMeta;
    public Map<String, String> grad2name;
    public String lossName;
    public String[] predictName;
    public String accuracy;

    public Map<String, Input> varMap;
    public Scope scope;

    public ClientSession client;

    public Input loss;
    public Input lossgrad;
    public Input accu;
    public Input seq;

    public Input lr;

    public Input[] predict;

    // all variable node
    public List<Input> variable;
    public Input[] assign;
    public Input[] placeholder;

    // all x node
    public List<Input> data;
    // all y node
    public List<Input> label;
    // add grad node
    public Input[] gradient;

    public List<Item> items;

    // used for tensorboard
    public Input summary;

    // used for async network
    public Input[] applys;

    public boolean async = false;
}