package bda.tensorflow.jni_11;

/**
 * Created by yixuanhe on 12/12/2016.
 */
public class SummaryOperation {
    public static native Input HistogramSummary(String tag, Input values, Scope scope);

    public static native Input MergeSummary(Input[] summarys, Scope scope);

    public static native Input ScalarSummary(String tag, Input values, Scope scope);
}
