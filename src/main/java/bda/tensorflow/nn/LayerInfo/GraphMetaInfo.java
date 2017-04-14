package bda.tensorflow.nn.LayerInfo;

import bda.tensorflow.jni_11.Scope;
import bda.tensorflow.run.RNN.RNNMeta;

/**
 * Created by yixuanhe on 9/19/16.
 */
public class GraphMetaInfo {
    public Scope scope;
    public RNNMeta meta;
    public int batchSize;
    public String appId;
    public String[] ps;

    public GraphMetaInfo(Scope scope, RNNMeta meta, int batchSize, String appId) {
        this.scope = scope;
        this.meta = meta;
        this.batchSize = batchSize;
        this.appId = appId;
        this.ps = null;
    }

    public GraphMetaInfo(Scope scope, RNNMeta meta, int batchSize, String appId, String[] ps) {
        this.scope = scope;
        this.meta = meta;
        this.batchSize = batchSize;
        this.appId = appId;
        this.ps = ps;
    }
}
