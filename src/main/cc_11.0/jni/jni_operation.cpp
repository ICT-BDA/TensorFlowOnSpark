#include "util/util.h"
#include "include/bda_tensorflow_jni_11_Operation.h"
/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Variable
 * Signature: ([IILbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Variable___3IILbda_tensorflow_jni_111_Scope_2Ljava_lang_String_2Ljava_lang_String_2
(JNIEnv *env, jclass jc, jintArray shape, jint type, jobject jscope, jstring jname, jstring jcontainer){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);
    const char* container = env->GetStringUTFChars(jcontainer, 0);

    std::vector<int64> vi;
    std::vector<int> v;
    putJintArrIntoVector(env, shape, v);
    for (int i : v) {
    	vi.push_back((int64)i);
    }

    Input i = Variable(scope->WithOpName(name), TensorShape(gtl::ArraySlice<int64>(vi)), DataType(type), Variable::Container(container));

    env->ReleaseStringUTFChars(jname, name);
    env->ReleaseStringUTFChars(jcontainer, container);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Variable
 * Signature: ([IILbda/tensorflow/jni_11/Scope;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Variable___3IILbda_tensorflow_jni_111_Scope_2Ljava_lang_String_2Ljava_lang_String_2Ljava_lang_String_2
(JNIEnv *env, jclass jc, jintArray shape, jint type, jobject jscope, jstring jname, jstring jcontainer, jstring jdevice){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);
    const char* container = env->GetStringUTFChars(jcontainer, 0);
    const char* device = env->GetStringUTFChars(jdevice, 0);

    std::vector<int64> vi;
    std::vector<int> v;
    putJintArrIntoVector(env, shape, v);
    for (int i : v) {
    	vi.push_back((int64)i);
    }

    Input i = Variable(scope->WithOpName(name).WithDevice(device), TensorShape(gtl::ArraySlice<int64>(vi)), DataType(type), Variable::Container(container));

    env->ReleaseStringUTFChars(jname, name);
    env->ReleaseStringUTFChars(jcontainer, container);
    env->ReleaseStringUTFChars(jdevice, device);
    return createInput(env, new Input(i));
}


/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Identity
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Identity
(JNIEnv *env, jclass jc, jobject jinput, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);
    Input *input = (Input*)GetNativeAddress(env, jinput);

    Input i = Identity(scope->WithOpName(name), *input);
    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Placeholder
 * Signature: (ILbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Placeholder
(JNIEnv *env, jclass jc, jint type, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input i = Placeholder(scope->WithOpName(name), DataType(type));
    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Transpose
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Transpose
(JNIEnv *env, jclass jc, jobject jtensor, jobject jpermution, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *tensor = (Input*)GetNativeAddress(env, jtensor);
    Input *permution = (Input*)GetNativeAddress(env, jpermution);

    Input i = Transpose(scope->WithOpName(name), *tensor, *permution);

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Conv2D
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;[ILjava/lang/String;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Conv2D
(JNIEnv *env, jclass jc, jobject jinput, jobject jfilter, jintArray jstrides, jstring jpadding, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *input = (Input*)GetNativeAddress(env, jinput);
    Input *filter = (Input*)GetNativeAddress(env, jfilter);

    std::vector<int> strides;
    putJintArrIntoVector(env, jstrides, strides);

    const char* padding = env->GetStringUTFChars(jpadding, 0);

    Input i = Conv2D(scope->WithOpName(name), *input, *filter, gtl::ArraySlice<int>(strides), padding);
    

    env->ReleaseStringUTFChars(jname, name);
    env->ReleaseStringUTFChars(jpadding, padding);

    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    BiasAdd
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_BiasAdd
(JNIEnv *env, jclass jc, jobject jvalue, jobject jbias, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *value = (Input*)GetNativeAddress(env, jvalue);
    Input *bias = (Input*)GetNativeAddress(env, jbias);

    Input i = BiasAdd(scope->WithOpName(name), *value, *bias);

    env->ReleaseStringUTFChars(jname, name);

    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Relu
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Relu
(JNIEnv *env, jclass jc, jobject jfeature, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *feature = (Input*)GetNativeAddress(env, jfeature);

    Input i = Relu(scope->WithOpName(name), *feature);
    env->ReleaseStringUTFChars(jname, name);

    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    MaxPool
 * Signature: (Lbda/tensorflow/jni_11/Input;[I[ILjava/lang/String;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_MaxPool
(JNIEnv *env, jclass jc, jobject jinput, jintArray jksize, jintArray jstrides, jstring jpadding, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *input = (Input*)GetNativeAddress(env, jinput);

    std::vector<int> ksize;
    putJintArrIntoVector(env, jksize, ksize);
    std::vector<int> strides;
    putJintArrIntoVector(env, jstrides, strides);

    const char* padding = env->GetStringUTFChars(jpadding, 0);

    Input i = MaxPool(scope->WithOpName(name), *input, ksize, strides, padding);

    env->ReleaseStringUTFChars(jname, name);
    env->ReleaseStringUTFChars(jpadding, padding);

    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    MatMul
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_MatMul
(JNIEnv *env, jclass jc, jobject ja, jobject jb, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *a = (Input*)GetNativeAddress(env, ja);
    Input *b = (Input*)GetNativeAddress(env, jb);

    Input i = MatMul(scope->WithOpName(name), *a, *b);
    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Add
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Add
(JNIEnv *env, jclass jc, jobject jx, jobject jy, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *x = (Input*)GetNativeAddress(env, jx);
    Input *y = (Input*)GetNativeAddress(env, jy);

    Input i = Add(scope->WithOpName(name), *x, *y);
    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Mul
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Mul
(JNIEnv *env, jclass jc, jobject jx, jobject jy, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *x = (Input*)GetNativeAddress(env, jx);
    Input *y = (Input*)GetNativeAddress(env, jy);

    Input i = Mul(scope->WithOpName(name), *x, *y);
    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Sub
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Sub
(JNIEnv *env, jclass jc, jobject jx, jobject jy, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *x = (Input*)GetNativeAddress(env, jx);
    Input *y = (Input*)GetNativeAddress(env, jy);

    Input i = Sub(scope->WithOpName(name), *x, *y);
    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    L2Loss
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_L2Loss
(JNIEnv *env, jclass jc, jobject jt, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *t = (Input*)GetNativeAddress(env, jt);

    Input i = L2Loss(scope->WithOpName(name), *t);
    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Mean
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Mean
(JNIEnv *env, jclass jc, jobject jinput, jobject jreduce, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *input = (Input*)GetNativeAddress(env, jinput);
    Input *reduce = (Input*)GetNativeAddress(env, jreduce);

    Input i = Mean(scope->WithOpName(name), *input, *reduce);
    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    SparseSoftmaxCrossEntropyWithLogits
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_SparseSoftmaxCrossEntropyWithLogits
(JNIEnv *env, jclass jc, jobject jfeatures, jobject jlabel, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *features = (Input*)GetNativeAddress(env, jfeatures);
    Input *label = (Input*)GetNativeAddress(env, jlabel);

    SparseSoftmaxCrossEntropyWithLogits i(scope->WithOpName(name), *features, *label);

    env->ReleaseStringUTFChars(jname, name);

    return createInput(env, new Input(i.loss));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Reshape
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Reshape
(JNIEnv *env, jclass jc, jobject jtensor, jobject jshape, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *tensor = (Input*)GetNativeAddress(env, jtensor);
    Input *shape = (Input*)GetNativeAddress(env, jshape);

    Input i = Reshape(scope->WithOpName(name), *tensor, *shape);

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Const
 * Signature: ([ILbda/tensorflow/jni/TensorShape;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Const___3ILbda_tensorflow_jni_TensorShape_2Lbda_tensorflow_jni_111_Scope_2Ljava_lang_String_2
(JNIEnv *env, jclass jc, jintArray jt, jobject jts, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    TensorShape* ts = (TensorShape*)GetNativeAddress(env, jts);


    std::vector<int> t;
    putJintArrIntoVector(env, jt, t);

    Tensor tensor(DataType::DT_INT32, *ts);
    std::copy_n(t.begin(), t.size(), tensor.flat<int>().data());

    Input i = Const(scope->WithOpName(name), Input::Initializer(tensor));

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));

}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Const
 * Signature: ([FLbda/tensorflow/jni/TensorShape;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Const___3FLbda_tensorflow_jni_TensorShape_2Lbda_tensorflow_jni_111_Scope_2Ljava_lang_String_2
(JNIEnv *env, jclass jc, jfloatArray jt, jobject jts, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    TensorShape* ts = (TensorShape*)GetNativeAddress(env, jts);

    std::vector<float> t;
    putJfloatArrIntoVector(env, jt, t);

    Tensor tensor(DataType::DT_FLOAT, *ts);
    std::copy_n(t.begin(), t.size(), tensor.flat<float>().data());

    Input i = Const(scope->WithOpName(name), Input::Initializer(tensor));

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));

}


/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    TruncatedNormal
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_TruncatedNormal
(JNIEnv *env, jclass jc, jobject jshape, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *shape = (Input*)GetNativeAddress(env, jshape);

    Input i = TruncatedNormal(scope->WithOpName(name), *shape, DataType::DT_FLOAT);

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Assign
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Assign
(JNIEnv *env, jclass jc, jobject jref, jobject jvalue, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *ref = (Input*)GetNativeAddress(env, jref);
    Input *value = (Input*)GetNativeAddress(env, jvalue);

    Input i = Assign(scope->WithOpName(name), *ref, *value);

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    AssignSub
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_AssignSub
(JNIEnv *env, jclass jc, jobject jref, jobject jvalue, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *ref = (Input*)GetNativeAddress(env, jref);
    Input *value = (Input*)GetNativeAddress(env, jvalue);

    Input i = AssignSub(scope->WithOpName(name), *ref, *value);

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    AddSymbolicGradients
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;[Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;)[Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobjectArray JNICALL Java_bda_tensorflow_jni_111_Operation_AddSymbolicGradients
(JNIEnv *env, jclass jc, jobject jloss, jobject jgrad, jobjectArray jinputs, jobject jscope){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    Input *loss = (Input*)GetNativeAddress(env, jloss);
    Input *gradient = (Input*)GetNativeAddress(env, jgrad);

    int length = env->GetArrayLength(jinputs);
    std::vector<Output> inputs;
    for(int i = 0; i < length; i++){
        jobject element = env->GetObjectArrayElement(jinputs, i);
        inputs.push_back(Output(((Input*)GetNativeAddress(env, element))->node()));
    }

    std::vector<Output> grads;
    Status sta = AddSymbolicGradients(*scope, { Output(loss->node()) }, inputs, { Output(gradient->node()) }, &grads);
    std::cout << sta.error_message() << std::endl;

    length = grads.size();

    jclass string_class = env->FindClass("bda/tensorflow/jni_11/Input");
    jobjectArray ret = env->NewObjectArray(length, string_class, NULL);
    int i = 0;
    for(Output g : grads){
        jobject o = createInput(env, new Input(g));
        env->SetObjectArrayElement(ret, i, o);
        i += 1;
    }
    return ret;
}


/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    InTopK
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;ILbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_InTopK
(JNIEnv *env, jclass jc, jobject jpre, jobject jtarget, jint jk, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *pre = (Input*)GetNativeAddress(env, jpre);
    Input *target = (Input*)GetNativeAddress(env, jtarget);

    Input i = InTopK(scope->WithOpName(name), *pre, *target, jk);
    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Gather
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;ZLbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Gather
(JNIEnv *env, jclass jc, jobject jparams, jobject jindices, jboolean jval, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *params = (Input*)GetNativeAddress(env, jparams);
    Input *indices = (Input*)GetNativeAddress(env, jindices);

    Input i = Gather(scope->WithOpName(name), *params, *indices);

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    ExpandDims
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_ExpandDims
(JNIEnv *env, jclass jc, jobject jinput, jobject jdim, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *input = (Input*)GetNativeAddress(env, jinput);
    Input *dim = (Input*)GetNativeAddress(env, jdim);

    Input i = ExpandDims(scope->WithOpName(name), *input, *dim);

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Concat
 * Signature: (Lbda/tensorflow/jni_11/Input;[Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Concat
(JNIEnv *env, jclass jc, jobject jdim, jobjectArray jvalues, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *dim = (Input*)GetNativeAddress(env, jdim);

    jsize size = env->GetArrayLength(jvalues);
    std::vector<Input> vec_values;
    for(int i = 0; i < size; i++){
        jobject jinput_i = env->GetObjectArrayElement(jvalues, i);
        vec_values.push_back(*(Input*)GetNativeAddress(env, jinput_i));
    }

    Input i = Concat(scope->WithOpName(name), *dim, InputList(gtl::ArraySlice<Input>(vec_values)));

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Equal
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Equal
(JNIEnv *env, jclass jc, jobject jx, jobject jy, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *x = (Input*)GetNativeAddress(env, jx);
    Input *y = (Input*)GetNativeAddress(env, jy);

    Input i = Equal(scope->WithOpName(name), *x, *y);

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    ArgMax
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_ArgMax
(JNIEnv *env, jclass jc, jobject jinput, jobject jdim, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *dim = (Input*)GetNativeAddress(env, jdim);
    Input *input = (Input*)GetNativeAddress(env, jinput);

    Input i = ArgMax(scope->WithOpName(name), *input, *dim);

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Cast
 * Signature: (Lbda/tensorflow/jni_11/Input;ILbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Cast
(JNIEnv *env, jclass jc, jobject jinput, jint jtype, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *input = (Input*)GetNativeAddress(env, jinput);

    Input i = Cast(scope->WithOpName(name), *input, DataType(jtype));

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Tanh
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Tanh
(JNIEnv *env, jclass jc, jobject jfeatures, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *features = (Input*)GetNativeAddress(env, jfeatures);

    Input i = Tanh(scope->WithOpName(name), *features);

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    Slice
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Scope;Ljava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_Slice
(JNIEnv *env, jclass jc, jobject jinput, jobject jbegin, jobject jsize, jobject jscope, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *input = (Input*)GetNativeAddress(env, jinput);
    Input *begin = (Input*)GetNativeAddress(env, jbegin);
    Input *size = (Input*)GetNativeAddress(env, jsize);

    Input i = Slice(scope->WithOpName(name), *input, *begin, *size);

    
    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}

/*
 * Class:     bda_tensorflow_jni_11_Operation
 * Method:    ApplyGradientDescent
 * Signature: (Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;Lbda/tensorflow/jni_11/Input;ZLjava/lang/String;)Lbda/tensorflow/jni_11/Input;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_111_Operation_ApplyGradientDescent
(JNIEnv *env, jclass jc, jobject jinput, jobject jalpha, jobject jdelta, jobject jscope, jboolean jusingLock, jstring jname){
    Scope* scope = (Scope*)GetNativeAddress(env, jscope);
    const char* name = env->GetStringUTFChars(jname, 0);

    Input *input = (Input*)GetNativeAddress(env, jinput);
    Input *alpha = (Input*)GetNativeAddress(env, jalpha);
    Input *delta = (Input*)GetNativeAddress(env, jdelta);

    ApplyGradientDescent::Attrs attr;

    Input i = ApplyGradientDescent(scope->WithOpName(name), *input, *alpha, *delta, attr.UseLocking(jusingLock));

    env->ReleaseStringUTFChars(jname, name);
    return createInput(env, new Input(i));
}