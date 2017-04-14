#include "include/bda_tensorflow_jni_Operation.h"
#include "util/util.h"

/*
* Class:     tensorflow_jni_Operation
* Method:    Variable
* Signature: ([IILtensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Variable
(JNIEnv * env, jclass cls, jintArray jiarr, jint ji, jobject jo, jstring js) {

	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jo);
	const char* name = env->GetStringUTFChars(js, 0);
	std::vector<int64> vi;
	std::vector<int> v;
	putJintArrIntoVector(env, jiarr, v);
	for (int i : v) {
		vi.push_back((int64)i);
	}
	Node* node = Variable(TensorShape(gtl::ArraySlice<int64>(vi)), DataType(ji), gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(js, name);
	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Identity
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Identity
(JNIEnv * env, jclass cls, jobject jinput, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd = env->GetObjectField(jinput, node_id);
	jint ni = env->GetIntField(jinput, index_id);
	Node* input = (Node*)GetNativeAddress(env, nd);
	Node* node = Identity(tensorflow::ops::NodeOut(input, ni), gdb->opts().WithName(name));

    if (node == nullptr)
            throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);
	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Placeholder
* Signature: (ILtensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Placeholder
(JNIEnv * env, jclass cls, jint jtype, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	Node* node = Placeholder(DataType(jtype), gdb->opts().WithName(name));

    if (node == nullptr)
            throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);
	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Transpose
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Transpose
(JNIEnv * env, jclass cls, jobject jx, jobject jperm, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(jx, node_id);
	jint ni1 = env->GetIntField(jx, index_id);

	jobject nd2 = env->GetObjectField(jperm, node_id);
	jint ni2 = env->GetIntField(jperm, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);
	Node* node = Transpose(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));
        
    if (node == nullptr)
            throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);
	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Conv2D
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;[ILjava/lang/String;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Conv2D
(JNIEnv * env, jclass cls, jobject input, jobject filter, jintArray jstrides, jstring jpadding,
	jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	const char* padding = env->GetStringUTFChars(jpadding, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(input, node_id);
	jint ni1 = env->GetIntField(input, index_id);

	jobject nd2 = env->GetObjectField(filter, node_id);
	jint ni2 = env->GetIntField(filter, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);

	std::vector<int> vi;
	putJintArrIntoVector(env, jstrides, vi);
	Node* node = Conv2D(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gtl::ArraySlice<int>(vi),
		padding,
		gdb->opts().WithName(name));

    if (node == nullptr)
            throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);
	env->ReleaseStringUTFChars(jpadding, padding);
	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    BiasAdd
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_BiasAdd
(JNIEnv * env, jclass cls, jobject value, jobject bias, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(value, node_id);
	jint ni1 = env->GetIntField(value, index_id);

	jobject nd2 = env->GetObjectField(bias, node_id);
	jint ni2 = env->GetIntField(bias, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);


	Node* node = BiasAdd(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));

    if (node == nullptr)
            throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    AssignSub
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;ILtensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_InTopK
(JNIEnv *env, jclass cls, jobject jpredictions, jobject jtargets, jint jk, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(jpredictions, node_id);
	jint ni1 = env->GetIntField(jpredictions, index_id);

	jobject nd2 = env->GetObjectField(jtargets, node_id);
	jint ni2 = env->GetIntField(jtargets, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);


	Node* node = InTopK(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		jk,
		gdb->opts().WithName(name));

    if (node == nullptr)
            throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Relu
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Relu
(JNIEnv * env, jclass cls, jobject features, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(features, node_id);
	jint ni1 = env->GetIntField(features, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);

	Node* node = Relu(tensorflow::ops::NodeOut(input1, ni1),
		gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);
	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Tanh
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Tanh
(JNIEnv * env, jclass cls, jobject features, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(features, node_id);
	jint ni1 = env->GetIntField(features, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);

	Node* node = Tanh(tensorflow::ops::NodeOut(input1, ni1),
		gdb->opts().WithName(name));

        if (node == nullptr)
                throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);
	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    MaxPool
* Signature: (Ltensorflow/util/NodeOut;[I[ILjava/lang/String;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_MaxPool
(JNIEnv * env,
	jclass cls,
	jobject input,
	jintArray jksize,
	jintArray jstrides,
	jstring jpadding,
	jobject jgdb,
	jstring jname) {

	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	const char* padding = env->GetStringUTFChars(jpadding, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(input, node_id);
	jint ni1 = env->GetIntField(input, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);

	std::vector<int> ksize;
	std::vector<int> strides;
	putJintArrIntoVector(env, jksize, ksize);
	putJintArrIntoVector(env, jstrides, strides);
	Node* node = MaxPool(tensorflow::ops::NodeOut(input1, ni1),
		gtl::ArraySlice<int>(ksize),
		gtl::ArraySlice<int>(strides),
		padding,
		gdb->opts().WithName(name));

        if (node == nullptr)
                throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);
	env->ReleaseStringUTFChars(jpadding, padding);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    MatMul
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_MatMul
(JNIEnv* env, jclass cls, jobject a, jobject b, jobject jgdb, jstring jname) {

	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(a, node_id);
	jint ni1 = env->GetIntField(a, index_id);

	jobject nd2 = env->GetObjectField(b, node_id);
	jint ni2 = env->GetIntField(b, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);


	Node* node = MatMul(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());
	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Add
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Add
(JNIEnv* env, jclass cls, jobject x, jobject y, jobject jgdb, jstring jname) {

	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(x, node_id);
	jint ni1 = env->GetIntField(x, index_id);

	jobject nd2 = env->GetObjectField(y, node_id);
	jint ni2 = env->GetIntField(y, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);


	Node* node = Add(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Mul
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Mul
(JNIEnv* env, jclass cls, jobject x, jobject y, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(x, node_id);
	jint ni1 = env->GetIntField(x, index_id);

	jobject nd2 = env->GetObjectField(y, node_id);
	jint ni2 = env->GetIntField(y, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);


	Node* node = Mul(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Sub
* Signature: (Lbda/tensorflow/rnn/NodeOut;Lbda/tensorflow/rnn/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Sub
(JNIEnv* env, jclass cls, jobject x, jobject y, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(x, node_id);
	jint ni1 = env->GetIntField(x, index_id);

	jobject nd2 = env->GetObjectField(y, node_id);
	jint ni2 = env->GetIntField(y, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);


	Node* node = Sub(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    L2Loss
* Signature: (Lbda/tensorflow/rnn/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_L2Loss
(JNIEnv* env, jclass cls, jobject t, jobject jgdb, jstring jname) {

	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");

	jobject nd1 = env->GetObjectField(t, node_id);
	jint ni1 = env->GetIntField(t, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);

	Node* node = L2Loss(tensorflow::ops::NodeOut(input1, ni1),
		gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Mean
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Mean
(JNIEnv* env, jclass cls, jobject input, jobject reduction_indices, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(input, node_id);
	jint ni1 = env->GetIntField(input, index_id);

	jobject nd2 = env->GetObjectField(reduction_indices, node_id);
	jint ni2 = env->GetIntField(reduction_indices, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);

	Node* node = Mean(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    SparseSoftmaxCrossEntropyWithLogits
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_SparseSoftmaxCrossEntropyWithLogits
(JNIEnv* env, jclass cls, jobject features, jobject labels, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(features, node_id);
	jint ni1 = env->GetIntField(features, index_id);

	jobject nd2 = env->GetObjectField(labels, node_id);
	jint ni2 = env->GetIntField(labels, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);


	Node* node = SparseSoftmaxCrossEntropyWithLogits(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Reshape
* Signature: (Lbda/tensorflow/rnn/NodeOut;Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Reshape
(JNIEnv* env, jclass cls, jobject tensor, jobject shape, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(tensor, node_id);
	jint ni1 = env->GetIntField(tensor, index_id);

	jobject nd2 = env->GetObjectField(shape, node_id);
	jint ni2 = env->GetIntField(shape, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);

	Node* node = Reshape(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Const
* Signature: ([ILtensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Const___3ILbda_tensorflow_jni_GraphDefBuilder_2Ljava_lang_String_2
(JNIEnv* env, jclass cls, jintArray t, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);

	std::vector<int> vi;
	putJintArrIntoVector(env, t, vi);

	Node* node = Const(gtl::ArraySlice<int>(vi),
		gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Const
* Signature: ([FLtensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Const___3FLbda_tensorflow_jni_GraphDefBuilder_2Ljava_lang_String_2
(JNIEnv* env, jclass cls, jfloatArray t, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);

	std::vector<float> vf;
	putJfloatArrIntoVector(env, t, vf);

	Node* node = Const(gtl::ArraySlice<float>(vf),
		gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Const
* Signature: ([ILtensorflow/jni/TensorShape;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Const___3ILbda_tensorflow_jni_TensorShape_2Lbda_tensorflow_jni_GraphDefBuilder_2Ljava_lang_String_2
(JNIEnv* env, jclass cls, jintArray t, jobject shape, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);

	std::vector<int> vi;
	putJintArrIntoVector(env, t, vi);

	TensorShape* ts = (TensorShape*)GetNativeAddress(env, shape);

	Node* node = Const(gtl::ArraySlice<int>(vi),
		*ts,
		gdb->opts().WithName(name)
		);

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Const
* Signature: ([FLtensorflow/jni/TensorShape;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Const___3FLbda_tensorflow_jni_TensorShape_2Lbda_tensorflow_jni_GraphDefBuilder_2Ljava_lang_String_2
(JNIEnv* env, jclass cls, jfloatArray t, jobject shape, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);

	std::vector<float> vf;
	putJfloatArrIntoVector(env, t, vf);

	TensorShape* ts = (TensorShape*)GetNativeAddress(env, shape);

	Node* node = Const(gtl::ArraySlice<float>(vf),
		*ts,
		gdb->opts().WithName(name)
		);

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Const
* Signature: (ILtensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/jni/Node;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Const__ILbda_tensorflow_jni_GraphDefBuilder_2Ljava_lang_String_2
(JNIEnv * env, jclass jcls, jint jt, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	int t = jt;
	Node* node = Const(t,
		gdb->opts().WithName(name)
		);

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Const
* Signature: (FLtensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/jni/Node;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Const__FLbda_tensorflow_jni_GraphDefBuilder_2Ljava_lang_String_2
(JNIEnv * env, jclass jcls, jfloat jt, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	float t = jt;
	Node* node = Const(t,
		gdb->opts().WithName(name)
		);

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    TruncatedNormal
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_TruncatedNormal
(JNIEnv* env, jclass cls, jobject shape, jobject jgdb, jstring jname) {
	//the float version
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(shape, node_id);
	jint ni1 = env->GetIntField(shape, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);

	Node* node = TruncatedNormal(tensorflow::ops::NodeOut(input1, ni1),
		DataType::DT_FLOAT,
		gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Assign
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Assign
(JNIEnv* env, jclass cls, jobject ref, jobject value, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(ref, node_id);
	jint ni1 = env->GetIntField(ref, index_id);

	jobject nd2 = env->GetObjectField(value, node_id);
	jint ni2 = env->GetIntField(value, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);

	Node* node = Assign(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    AssignSub
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/util/NodeOut;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_AssignSub
(JNIEnv* env, jclass cls, jobject ref, jobject value, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	//println("in jni operation.cpp lin 575");
	//println(name);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	//println("in jni operation.cpp lin 579");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(ref, node_id);
	//println("in jni operation.cpp lin 582");
	jint ni1 = env->GetIntField(ref, index_id);

	jobject nd2 = env->GetObjectField(value, node_id);
	jint ni2 = env->GetIntField(value, index_id);

	//println("in jni operation.cpp lin 588");
	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	//println(input1->name());
	Node* input2 = (Node*)GetNativeAddress(env, nd2);
	//println(input2->name());

	Node* node = AssignSub(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));
	//println(gdb->)

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    AddSymbolicGradients
* Signature: (Ljava/lang/String;Ljava/lang/String;[Ljava/lang/String;Ltensorflow/jni/Graph;)[Ljava/lang/String;
*/
JNIEXPORT jobjectArray JNICALL JNICALL Java_bda_tensorflow_jni_Operation_AddSymbolicGradients__Ljava_lang_String_2Ljava_lang_String_2_3Ljava_lang_String_2Lbda_tensorflow_jni_Graph_2
(JNIEnv *env, jclass cls, jstring jy_name, jstring jy_grad_name, jobjectArray jx_names, jobject jgraph)
{
	Graph* g = (Graph*)GetNativeAddress(env, jgraph);

	const char* y_name = env->GetStringUTFChars(jy_name, 0);
	const char* y_grad_name = env->GetStringUTFChars(jy_grad_name, 0);
	std::vector<std::string> x_names;
	GetStringVector(env, x_names, jx_names);
	/*for (std::string str : x_names) {
		std::cout <<"x_names contains "<< str << std::endl;
	}*/

	//y_gradient
	tensorflow::NodeOut y_grad;
	//y_node_output
	tensorflow::NodeOut y_node_output;
	//x_node_output
	std::vector<::tensorflow::NodeOut> x_node_output;
	//std::cout << "x_names size :" << x_names.size() << std::endl;
	x_node_output.resize(x_names.size());
	for (Node* node : g->nodes()) {
		std::string name = node->name();
		//std::cout << "the name is " << name << std::endl;
		if (name == y_name) {
			y_node_output.node = node;
			y_node_output.index = 0;
			//std::cout << "in loss_identity" << std::endl;
		}
		else if (name == y_grad_name) {
			y_grad.node = node;
			y_grad.index = 0;
		}
		else {
			auto iter = std::find(x_names.begin(), x_names.end(), name);
			if (iter != x_names.end()) {
				uint16 index = iter - x_names.begin();
				x_node_output[index].node = node;
				x_node_output[index].index = 0;
			}
		}
	}

	std::vector<::tensorflow::NodeOut> x_grad;
	Status sta = AddSymbolicGradients({ y_node_output }, x_node_output, { y_grad }, &x_grad, g);

	std::cout << sta.error_message() << std::endl;

	for (::tensorflow::NodeOut& nodeout : x_grad) {
		x_grad_name.push_back(nodeout.name());
	}
	jobjectArray joa = GetJStringFromVector(env, x_grad_name);
	env->ReleaseStringUTFChars(jy_grad_name, y_grad_name);
	env->ReleaseStringUTFChars(jy_name, y_name);
	return joa;
}

/*
 * Class:     bda_tensorflow_jni_Operation
 * Method:    AddSymbolicGradients
 * Signature: ([Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;Lbda/tensorflow/jni/Graph;)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_bda_tensorflow_jni_Operation_AddSymbolicGradients___3Ljava_lang_String_2_3Ljava_lang_String_2_3Ljava_lang_String_2Lbda_tensorflow_jni_Graph_2
  (JNIEnv *env, jclass jcls, jobjectArray jy_names, jobjectArray jy_grad_names, jobjectArray jx_names, jobject jgdb){
        std::cout<<"1"<<std::endl;
	Graph* g = (Graph*)GetNativeAddress(env, jgdb);

	std::cout<<"2"<<std::endl;
        std::vector<std::string> y_names;
        GetStringVector(env, y_names, jy_names);
        std::vector<std::string> y_grad_names;
        GetStringVector(env, y_grad_names, jy_grad_names);
	std::vector<std::string> x_names;
	GetStringVector(env, x_names, jx_names);

	std::cout<<"3"<<std::endl;
	std::vector<::tensorflow::NodeOut> y_node_output;
	y_node_output.resize(y_names.size());
	std::vector<::tensorflow::NodeOut> y_grad_output;
	y_grad_output.resize(y_grad_names.size());
	std::vector<::tensorflow::NodeOut> x_node_output;
        x_node_output.resize(x_names.size());

	std::cout<<"4"<<std::endl;
	for (Node* node : g->nodes()) {
		std::string name = node->name();

		auto iter = std::find(x_names.begin(), x_names.end(), name);
		if (iter != x_names.end()) {
			uint16 index = iter - x_names.begin();
			x_node_output[index].node = node;
			x_node_output[index].index = 0;
			continue;
		}		

		iter = std::find(y_names.begin(), y_names.end(), name);
                if (iter != y_names.end()) {
                        uint16 index = iter - y_names.begin();
                        y_node_output[index].node = node;
                        y_node_output[index].index = 0;
                        continue;
                }	
		
		iter = std::find(y_grad_names.begin(), y_grad_names.end(), name);
                if (iter != x_names.end()) {
                        uint16 index = iter - y_grad_names.begin();
                        y_grad_output[index].node = node;
                        y_grad_output[index].index = 0;
                        continue;
                }	
	}
	std::vector<::tensorflow::NodeOut> x_grad;
	Status sta = AddSymbolicGradients(y_node_output, x_node_output, y_grad_output, &x_grad, g);


	if (!sta.ok()) 
		std::cout << sta.error_message() << std::endl;

	std::vector<std::string> x_grad_name;
	for (::tensorflow::NodeOut& nodeout : x_grad) {
                //println(nodeout.name());
                x_grad_name.push_back(nodeout.name());
        }
	jobjectArray joa = GetJStringFromVector(env, x_grad_name);
    return joa;
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Gather
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;ZLtensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/jni/Node;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Gather
(JNIEnv * env, jclass jcls, 
	jobject jparams,
	jobject jindices, 
	jboolean jvalidate_indices, 
	jobject jgdb,
	jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");

	jobject nd1 = env->GetObjectField(jparams, node_id);
	jint ni1 = env->GetIntField(jparams, index_id);

	jobject nd2 = env->GetObjectField(jindices, node_id);
	jint ni2 = env->GetIntField(jindices, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);

	Node* node = Gather(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));
	//println(gdb->)

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    ExpandDims
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/jni/Node;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_ExpandDims
(JNIEnv * env,
	jclass jcls,
	jobject jinput,
	jobject jdim,
	jobject jgdb,
	jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	//println("in jni operation.cpp lin 575");
	//println(name);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");

	jobject nd1 = env->GetObjectField(jinput, node_id);
	jint ni1 = env->GetIntField(jinput, index_id);

	jobject nd2 = env->GetObjectField(jdim, node_id);
	jint ni2 = env->GetIntField(jdim, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);

	Node* node = ExpandDims(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));
	//println(gdb->)

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Concat
* Signature: (Ltensorflow/util/NodeOut;[Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/jni/Node;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Concat
(JNIEnv * env,
	jclass jcls,
	jobject jconcat_dim,
	jobjectArray jvalues,
	jobject jgdb,
	jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);

	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");

	jobject nd1 = env->GetObjectField(jconcat_dim, node_id);
	jint ni1 = env->GetIntField(jconcat_dim, index_id);
	Node* input1 = (Node*)GetNativeAddress(env, nd1);

	jsize size = env->GetArrayLength(jvalues);
	std::vector<tensorflow::ops::NodeOut> vec_values;
	for (int i = 0; i < size; i++) {
		//the java version NodeOut
		jobject jnodeout_i = env->GetObjectArrayElement(jvalues, i);
		jobject jnode_i = env->GetObjectField(jnodeout_i, node_id);
		jint j_i = env->GetIntField(jnodeout_i, index_id);
		Node* node = (Node*)GetNativeAddress(env, jnode_i);
		vec_values.push_back(tensorflow::ops::NodeOut(node, j_i));
	}
	Node* node = Concat(tensorflow::ops::NodeOut(input1, ni1),
		gtl::ArraySlice<tensorflow::ops::NodeOut>(vec_values),
		gdb->opts().WithName(name));
	
    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);

}


/*
* Class:     tensorflow_jni_Operation
* Method:    Equal
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/jni/Node;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Equal
(JNIEnv * env,
	jclass jcls,
	jobject jx, 
	jobject jy,
	jobject jgdb, 
	jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");

	jobject nd1 = env->GetObjectField(jx, node_id);
	jint ni1 = env->GetIntField(jx, index_id);

	jobject nd2 = env->GetObjectField(jy, node_id);
	jint ni2 = env->GetIntField(jy, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);

	Node* node = Equal(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));
	//println(gdb->)

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());
	
	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    ArgMax
* Signature: (Ltensorflow/util/NodeOut;Ltensorflow/util/NodeOut;Ltensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/jni/Node;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_ArgMax
(JNIEnv * env,
	jclass jcls, 
	jobject jinput, 
	jobject jdimension,
	jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	//println("in jni operation.cpp lin 575");
	//println(name);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");

	jobject nd1 = env->GetObjectField(jinput, node_id);
	jint ni1 = env->GetIntField(jinput, index_id);

	jobject nd2 = env->GetObjectField(jdimension, node_id);
	jint ni2 = env->GetIntField(jdimension, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);
	Node* input2 = (Node*)GetNativeAddress(env, nd2);

	Node* node = ArgMax(tensorflow::ops::NodeOut(input1, ni1),
		tensorflow::ops::NodeOut(input2, ni2),
		gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
* Class:     tensorflow_jni_Operation
* Method:    Cast
* Signature: (Ltensorflow/util/NodeOut;ILtensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/jni/Node;
*/
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Cast
(JNIEnv * env, jclass jcls, jobject jinput, jint jtype, jobject jgdb, jstring jname) {
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	//println("in jni operation.cpp lin 575");
	//println(name);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");

	jobject nd1 = env->GetObjectField(jinput, node_id);
	jint ni1 = env->GetIntField(jinput, index_id);

	Node* input1 = (Node*)GetNativeAddress(env, nd1);

	Node* node = Cast(tensorflow::ops::NodeOut(input1, ni1),
		(DataType)jtype,
		gdb->opts().WithName(name));
	//println(gdb->)
    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());

	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
 *  Class:     tensorflow_jni_Operation
 *  Method:    Cast
 *  Signature: (Ltensorflow/util/NodeOut;ILtensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Ltensorflow/jni/Node;
 */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Select
(JNIEnv * env, jclass cls, jobject ref, jobjectArray array, jobject jgdb, jstring jname)
{
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");
	jobject nd1 = env->GetObjectField(ref, node_id);
	jint pred_id = env->GetIntField(ref, index_id);
	Node* pred = (Node*)GetNativeAddress(env, nd1);

	std::vector<tensorflow::NodeBuilder::NodeOut> input;
	jsize length = env->GetArrayLength(array);
	input.resize(length);
	for(int i = 0; i < length; i++){
                jobject jnodeout_i = env->GetObjectArrayElement(array, i);
		jobject jnode_i = env->GetObjectField(jnodeout_i, node_id);
		jint id = env->GetIntField(jnodeout_i, index_id);
		Node* node = (Node*)GetNativeAddress(env, jnode_i);
		input[i].node = node;
                input[i].index = id;
	}
        gtl::ArraySlice<tensorflow::NodeBuilder::NodeOut> inputs(input);
	Node* node = RefSelect(tensorflow::ops::NodeOut(pred, pred_id), inputs, gdb->opts().WithName(name));

    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());
	env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

/*
 *  * Class:     bda_tensorflow_jni_Operation
 *   * Method:    Slice
 *    * Signature: (Lbda/tensorflow/rnn/NodeOut;Lbda/tensorflow/rnn/NodeOut;Lbda/tensorflow/rnn/NodeOut;Lbda/tensorflow/jni/GraphDefBuilder;Ljava/lang/String;)Lbda/tensorflow/jni/Node;
 *     */
JNIEXPORT jobject JNICALL Java_bda_tensorflow_jni_Operation_Slice
  (JNIEnv *env, jclass cls, jobject jinput, jobject jbegin, jobject jsize, jobject jgdb, jstring jname){
	GraphDefBuilder* gdb = (GraphDefBuilder*)GetNativeAddress(env, jgdb);
	const char* name = env->GetStringUTFChars(jname, 0);
	jclass nodeout_class = env->FindClass("bda/tensorflow/rnn/NodeOut");
	jfieldID node_id = env->GetFieldID(nodeout_class, "node", "Lbda/tensorflow/jni/Node;");
	jfieldID index_id = env->GetFieldID(nodeout_class, "index", "I");

	jobject input_obj = env->GetObjectField(jinput, node_id);
	jint input_id = env->GetIntField(jinput, index_id);
        Node* input_node = (Node*)GetNativeAddress(env, input_obj);

        jobject begin_obj = env->GetObjectField(jbegin, node_id);
        jint begin_id = env->GetIntField(jbegin, index_id);
        Node* begin_node = (Node*)GetNativeAddress(env, begin_obj);

        jobject size_obj = env->GetObjectField(jsize, node_id);
        jint size_id = env->GetIntField(jsize, index_id);
        Node* size_node = (Node*)GetNativeAddress(env, size_obj);
 
	Node* node = Slice(tensorflow::ops::NodeOut(input_node, input_id), tensorflow::ops::NodeOut(begin_node, begin_id), tensorflow::ops::NodeOut(size_node, size_id), gdb->opts().WithName(name));
    if (node == nullptr)
        throwNodeCreateException(env, gdb->opts().ErrorMessage());
    env->ReleaseStringUTFChars(jname, name);

	return createNode(env, node);
}

