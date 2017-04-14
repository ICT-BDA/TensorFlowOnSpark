#include "include/bda_tensorflow_jni_Session.h"
#include "util/util.h"


/*
* Class:     tensorflow_jni_Session
* Method:    allocate
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Session_allocate
(JNIEnv * env, jobject obj) {
	//println("in session cpp line 12");
	SessionOptions options;
	Session* session = NewSession(options);
	SetNativeAddress(env, obj, session);
	//println("in session cpp line 16");
}

/*
* Class:     tensorflow_jni_Session
* Method:    deallocateMemory
* Signature: (J)V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Session_deallocateMemory
(JNIEnv * env, jobject obj, jlong address) {
	Session* session = (Session*)address;
	session->Close();
	delete session;
}

/*
* Class:     tensorflow_jni_Session
* Method:    create
* Signature: (Ltensorflow/jni/Graph;)V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Session_create
(JNIEnv *env, jobject obj, jobject jg) {
	Graph * g = (Graph*)GetNativeAddress(env,jg);
	GraphDef gd;
	g->ToGraphDef(&gd);

	Session* sess = (Session*)GetNativeAddress(env, obj);
	Status status = sess->Create(gd);
	if (!status.ok())
		std::cout << status.error_message() << std::endl;

}

/*
* Class:     tensorflow_jni_Session
* Method:    run
* Signature: ([Ljava/lang/String;[Ltensorflow/jni/Tensor;[Ljava/lang/String;[Ljava/lang/String;[Ltensorflow/jni/Tensor;Ltensorflow/jni/Status;)V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Session_run
(JNIEnv * env,
	jobject obj, 
	jobjectArray input_names,  // a string array
	jobjectArray input_tensors, // a Tensor array
	jobjectArray output_names,// a string array
	jobjectArray target_names, // a string array
	jobjectArray outputs, // a tensor array
	jobject status) {

	std::vector<std::string> in_n;
	std::vector<std::string> out_n;
	std::vector<std::string> t_n;
	GetStringVector(env, in_n, input_names);
	GetStringVector(env, out_n, output_names);
	GetStringVector(env, t_n, target_names);

	Status* st = (Status*) GetNativeAddress(env, status);

	std::vector<Tensor> vec_tensor;
	putJTensorArrayIntoVector(env, input_tensors, vec_tensor);

	if (in_n.size() != vec_tensor.size()) {
		std::cout << "there must be an error in session's run because the number of names and tensors don't match!!!" << std::endl;
		return;
	}
	std::vector<std::pair<std::string, Tensor>> input_names_tensors;
	int length = in_n.size();
	for (int i = 0; i < length; i++) {
		input_names_tensors.push_back({ in_n[i],vec_tensor[i] });
	}
	Session* sess = (Session*)GetNativeAddress(env, obj);
	std::vector<Tensor> vec_out;
	*st = sess->Run(input_names_tensors, out_n, t_n, &vec_out);

	jsize size = env->GetArrayLength(outputs);
	if (size != vec_out.size()) {
		std::cout << "your output dimension " << size << " and the actual outputs dimension " << vec_out.size() << " mismatch" << std::endl;
		return;
	}
	for (int i = 0; i < size; i++) {
		jclass tensor_class = env->FindClass("bda/tensorflow/jni/Tensor");
		jmethodID constructor = env->GetMethodID(tensor_class, "<init>", "()V");
		jobject ten = env->NewObject(tensor_class, constructor);
		Tensor* t_p = new Tensor(vec_out[i]);
		SetNativeAddress(env, ten, t_p);
		env->SetObjectArrayElement(outputs, i, ten);
	}
	//std::cout << "finished run" << std::endl;
}
