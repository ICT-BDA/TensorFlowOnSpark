#include "include/bda_tensorflow_jni_11_SummaryOperation.h"
#include "util/util.h"

#include "tensorflow/cc/ops/logging_ops.h"

/*
 * Class:     bda_tensorflow_jni_11_SummaryOperation
 * Method:    HistogramSummary
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_SummaryOperation_HistogramSummary
(JNIEnv *env, jclass jc, jstring jtag, jobject jvalue, jobject jscope){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* tag = env->GetStringUTFChars(jtag, 0);

    Input t = Const<string>(scope, string(tag));
    Input* value = (Input*)GetNativeAddress(env, jvalue);

    Input i = HistogramSummary(*scope, t, *value);

     env->ReleaseStringUTFChars(jtag, tag);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_SummaryOperation
 * Method:    MergeSummary
 * Signature: ([Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_SummaryOperation_MergeSummary
(JNIEnv *env, jclass jc, jobjectArray jsummarys, jobject jscope){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);

    jsize size = env->GetArrayLength(jsummarys);
    std::vector<Input> vec_values;
    for(int i = 0; i < size; i++){
        jobject jinput_i = env->GetObjectArrayElement(jsummarys, i);
        vec_values.push_back(*(Input*)GetNativeAddress(env, jinput_i));
    }

    Input i = MergeSummary(*scope, InputList(gtl::ArraySlice<Input>(vec_values)));

    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_SummaryOperation
 * Method:    ScalarSummary
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_SummaryOperation_ScalarSummary
(JNIEnv *env, jclass jc, jstring jtag, jobject jvalue, jobject jscope){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* tag = env->GetStringUTFChars(jtag, 0);

    Input t = Const<string>(scope, string(tag));
    Input* value = (Input*)GetNativeAddress(env, jvalue);

    Input i = ScalarSummary(*scope, t, *value);

    env->ReleaseStringUTFChars(jtag, tag);

    return createInput(env, new Input(i));
}
