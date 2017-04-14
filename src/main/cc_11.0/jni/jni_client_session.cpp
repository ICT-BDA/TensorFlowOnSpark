#include "include/bda_tensorflow_jni_11_ClientSession.h"
#include "util/util.h"
#include "tensorflow/cc/client/client_session.h"

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/util/event.pb.h"
#include "tensorflow/core/util/events_writer.h"

/*
 * Class:     bda_tensorflow_jni_11_ClienSession
 * Method:    allocate
 * Signature: (Lbda/tensorflow/jni_11/Scope;)V
 */
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_111_ClientSession_allocate
(JNIEnv *env, jobject jo, jobject jscope, jstring jtarget){
    Scope* s = (Scope*)GetNativeAddress(env, jscope);
    const char* target = env->GetStringUTFChars(jtarget, 0);

    ClientSession *c = new ClientSession(*s, target);
    SetNativeAddress(env, jo, c);

    env->ReleaseStringUTFChars(jtarget, target);
}

/*
 * Class:     bda_tensorflow_jni_11_ClientSession
 * Method:    deallocateMemory
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_111_ClientSession_deallocateMemory
(JNIEnv *env, jobject jo, jlong jl){
    delete (ClientSession*)jl ;
}

/*
 * Class:     bda_tensorflow_jni_11_ClientSession
 * Method:    run
 * Signature: ([Lbda/tensorflow/jni_11/Input;[Lbda/tensorflow/jni/Tensor;[Lbda/tensorflow/jni_11/Input;[Lbda/tensorflow/jni/Tensor;)V
 */
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_111_ClientSession_run___3Lbda_tensorflow_jni_111_Input_2_3Lbda_tensorflow_jni_Tensor_2_3Lbda_tensorflow_jni_111_Input_2_3Lbda_tensorflow_jni_Tensor_2
(JNIEnv *env, jobject jc, jobjectArray jinput, jobjectArray jtensor, jobjectArray jfetch , jobjectArray joutput){
    ClientSession* c = (ClientSession*)GetNativeAddress(env, jc);

    jsize size = env->GetArrayLength(jinput);
    ClientSession::FeedType feeds;
    std::vector<Output> inputs;
    for (int i = 0; i < size; i++){
        jobject jinput_i = env->GetObjectArrayElement(jinput, i);
        jobject jtensor_i = env->GetObjectArrayElement(jtensor, i);

        Input* input = (Input*)GetNativeAddress(env, jinput_i);
        Tensor* tensor = (Tensor*)GetNativeAddress(env, jtensor_i);

        feeds.insert(std::make_pair<Output,Input::Initializer>(Output(input->node()), Input::Initializer(*tensor)));
    }

    std::vector<Output> fetchs;
    size = env->GetArrayLength(jfetch);
    for (int i = 0; i < size; i++){
        jobject jinput_i = env->GetObjectArrayElement(jfetch, i);
        fetchs.push_back(Output(((Input*)GetNativeAddress(env, jinput_i))->node()));
    }

    std::vector<Tensor> outputs;

    c->Run(feeds, fetchs, &outputs);

    size = env->GetArrayLength(joutput);
    for (int i = 0; i < size; i++){
        jclass tensor_class = env->FindClass("bda/tensorflow/jni/Tensor");
 		jmethodID constructor = env->GetMethodID(tensor_class, "<init>", "()V");
 		jobject ten = env->NewObject(tensor_class, constructor);
 		Tensor* t_p = new Tensor(outputs[i]);
 		SetNativeAddress(env, ten, t_p);
 		env->SetObjectArrayElement(joutput, i, ten);
    }
}

/*
 * Class:     bda_tensorflow_jni_11_ClientSession
 * Method:    run
 * Signature: ([Lbda/tensorflow/jni_11/Input;[Lbda/tensorflow/jni/Tensor;[Lbda/tensorflow/jni_11/Input;[Lbda/tensorflow/jni/Tensor;Lbda/tensorflow/jni_11/Input;)V
 */
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_111_ClientSession_run___3Lbda_tensorflow_jni_111_Input_2_3Lbda_tensorflow_jni_Tensor_2_3Lbda_tensorflow_jni_111_Input_2_3Lbda_tensorflow_jni_Tensor_2Lbda_tensorflow_jni_111_Input_2
(JNIEnv *env, jobject jc, jobjectArray jinput, jobjectArray jtensor, jobjectArray jfetch , jobjectArray joutput, jobject jsummary){
    ClientSession* c = (ClientSession*)GetNativeAddress(env, jc);

    jsize size = env->GetArrayLength(jinput);
    ClientSession::FeedType feeds;
    std::vector<Output> inputs;
    for (int i = 0; i < size; i++){
        jobject jinput_i = env->GetObjectArrayElement(jinput, i);
        jobject jtensor_i = env->GetObjectArrayElement(jtensor, i);

        Input* input = (Input*)GetNativeAddress(env, jinput_i);
        Tensor* tensor = (Tensor*)GetNativeAddress(env, jtensor_i);

        feeds.insert(std::make_pair<Output,Input::Initializer>(Output(input->node()), Input::Initializer(*tensor)));
    }

    std::vector<Output> fetchs;
    size = env->GetArrayLength(jfetch);
    for (int i = 0; i < size; i++){
        jobject jinput_i = env->GetObjectArrayElement(jfetch, i);
        fetchs.push_back(Output(((Input*)GetNativeAddress(env, jinput_i))->node()));
    }

    // add summary output
    fetchs.push_back(Output(((Input*)GetNativeAddress(env, jsummary))->node()));

    std::vector<Tensor> outputs;
    c->Run(feeds, fetchs, &outputs);

    //get summary and write to file
    Tensor sum = outputs.back();
    outputs.pop_back();
    const string* sum_str = sum.flat<string>().data();
    Summary s;
    s.ParseFromString(sum_str);
    Event event;
    *event.mutable_summary() = s;
    EventsWriter writer("/home/cgx/temp/");
    writer.WriteEvent(event);
    writer.Close();

    size = env->GetArrayLength(joutput);
    for (int i = 0; i < size; i++){
        jclass tensor_class = env->FindClass("bda/tensorflow/jni/Tensor");
 		jmethodID constructor = env->GetMethodID(tensor_class, "<init>", "()V");
 		jobject ten = env->NewObject(tensor_class, constructor);
 		Tensor* t_p = new Tensor(outputs[i]);
 		SetNativeAddress(env, ten, t_p);
 		env->SetObjectArrayElement(joutput, i, ten);
    }
}
